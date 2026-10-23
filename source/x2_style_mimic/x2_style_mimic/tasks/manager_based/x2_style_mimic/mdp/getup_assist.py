"""起身辅助力：按回合采样，按时间平滑撤除。"""

from collections import deque
import torch


class GetUpAssist:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]

        self.levels = tuple(env.cfg.assist_force_levels)    # 读取按阶段递减的辅助力上限，单位牛顿。
        self.stage = int(env.cfg.assist_initial_stage)      # 读取初始阶段索引，必须在 [0, len(levels)) 范围内。
        self.zero_probability = float(env.cfg.assist_zero_probability)  # 读取零辅助概率，范围 [0, 1]。
        self.full_until = float(env.cfg.assist_full_until_s)            # 读取最大辅助力保持时间，单位秒
        self.zero_after = float(env.cfg.assist_zero_after_s)            # 读取辅助力完全撤除时间，单位秒。

        if (
            not self.levels
            or not all(
                0.0 <= force < float("inf")
                for force in self.levels
            )
            or self.levels[-1] != 0.0
            or any(
                a < b
                for a, b in zip(
                    self.levels, self.levels[1:]
                )
            )
        ):
            raise ValueError("辅助力阶段必须非负、递减，最后为零")
        if not 0 <= self.stage < len(self.levels):
            raise ValueError("辅助力初始阶段无效")
        if not 0.0 <= self.zero_probability <= 1.0:
            raise ValueError("零辅助概率必须在 [0, 1]")
        if not 0.0 <= self.full_until < self.zero_after:
            raise ValueError("辅助力保持/撤除时间无效")

        body_ids, _ = self.robot.find_bodies("torso_link")      # 只允许一个躯干刚体，辅助力作用于其质心。
        if len(body_ids) != 1:
            raise RuntimeError("必须匹配一个 torso_link")

        self.body_ids = torch.tensor(           # 保存躯干刚体索引，便于后续施加外力。
            body_ids,
            dtype=torch.long,
            device=env.device,
        )
        self.episode_force = torch.zeros(       # 保存每个环境的本回合辅助力上限，单位牛顿。
            env.num_envs,
            device=env.device
        )
        self.episode_stage = torch.zeros(       # 保存每个环境的本回合辅助力阶段，单位为索引。
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
        self.applied_force = torch.zeros(       # 保存每个环境的当前施加辅助力，单位牛顿。
            env.num_envs, device=env.device
        )
        self.forces = torch.zeros(              # 保存每个环境的当前施加辅助力向量，单位牛顿，形状为 [num_envs, 1, 3]。
            (env.num_envs, 1, 3),
            device=env.device,
        )
        self.torques = torch.zeros_like(self.forces)        # 保存每个环境的当前施加辅助力矩，单位牛顿·米，形状为 [num_envs, 1, 3]。
        self.curriculum_enabled = bool(         # 读取是否启用课程学习，若启用则根据最近若干回合的成功率决定是否降低辅助力。
            env.cfg.assist_curriculum_enabled
        )
        self.window_size = int(                 # 读取课程统计窗口大小，单位为回合数。
            env.cfg.assist_curriculum_window
        )
        self.threshold = float(                 # 读取课程成功率阈值，范围 (0, 1]，若最近若干回合的成功率超过该阈值则降低辅助力。
            env.cfg.assist_curriculum_threshold
        )

        if self.window_size <= 0:
            raise ValueError("课程统计窗口必须大于零")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("课程阈值必须在 (0, 1]")

        dataset = env._getup_motion_dataset_cache[1]        # 获取动作数据集，便于课程学习统计每个动作的成功率。
        self.active_motion_ids = (
            (dataset.weights > 0)
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        self.results = {                                    # 为每条动作创建结果队列
            motion_id: deque(maxlen=self.window_size)
            for motion_id in self.active_motion_ids
        }

        self.post_assist_succeeded = torch.zeros(           # 记录每个环境是否曾经在“完全无辅助力期间”成功保持稳定。
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        self.max_post_assist_hold = torch.zeros(            # 记录每个环境在无辅助力期间达到的最大连续稳定时间。
            env.num_envs,
            device=env.device,
        )

    @torch.no_grad()
    def reset(self, env_ids):
        """新回合采样辅助力，回合中不改变其上限。"""
        ids = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.env.device,
        )
        count = ids.numel()
        if count == 0:
            return

        force = self.levels[self.stage]                 # 采样当前阶段的辅助力上限，单位牛顿。
        zero_mask = (
            torch.rand(count, device=self.env.device)
            < self.zero_probability
        )

        self.episode_force[ids] = torch.where(
            zero_mask,
            torch.zeros(count, device=self.env.device),
            torch.full(
                (count,),               # size
                float(force),           # fill_value
                device=self.env.device,
            ),
        )
        self.episode_stage[ids] = self.stage
        self.applied_force[ids] = 0.0

        self.post_assist_succeeded[ids] = False
        self.max_post_assist_hold[ids] = 0.0

    @torch.no_grad()
    def apply(self):
        """每个控制步执行一次，外力持续作用于物理子步。"""
        # 当前回合已过去时间，单位秒
        elapsed = (
            self.env.episode_length_buf
            * self.env.step_dt
        )
        # 当前回合的进度，范围 [0, 1]
        progress = (
            (elapsed - self.full_until) / (self.zero_after - self.full_until)
        ).clamp(0.0, 1.0)

        # 从 1 平滑下降到 0。
        fade = 1.0 - progress.square() * (3.0 - 2.0 * progress)

        self.applied_force.copy_(self.episode_force * fade)         # 计算当前施加的辅助力，单位牛顿。
        self.forces.zero_()                                         # 先把所有环境的力清零。
        self.forces[:, 0, 2] = self.applied_force                   # 施加的辅助力方向为全局 Z 轴向上，单位牛顿。

        # 此控制器独占 permanent 外力缓冲。
        # 后续若增加其他外力，必须统一组合，不能互相清空。
        composer = self.robot.permanent_wrench_composer
        composer.reset()
        composer.add_forces_and_torques_index(
            forces=self.forces,
            torques=self.torques,
            body_ids=self.body_ids,             # 施加在躯干刚体上
            is_global=True,                     # 施加在全局坐标系下
        )

    @torch.no_grad()
    def observe_stability(self, current_hold, required_hold):
        """
        每步更新：只计入完全无辅助期间的连续稳定。
        current_hold：当前已经连续稳定了多久；
        required_hold：要求至少稳定多久才算成功。
        """
        elapsed = (self.env.episode_length_buf * self.env.step_dt)

        # 确定无辅助阶段的开始时间
        # 零辅助回合从 t=0 开始；
        # 有辅助回合从撤力完成时开始。
        zero_started = torch.where(
            self.episode_force > 0,
            torch.full_like(elapsed, self.zero_after),
            torch.zeros_like(elapsed),
        )
        # 计算每个环境的无辅助阶段已经持续了多久，单位秒
        zero_duration = (
            elapsed - zero_started
        ).clamp_min(0.0)

        # 连续稳定可能早于撤力开始，只截取无辅助部分。
        post_hold = torch.minimum(
            current_hold, zero_duration
        )

        # 更新每个环境的最大连续稳定时间，并判断是否成功。
        self.max_post_assist_hold.copy_(
            torch.maximum(
                self.max_post_assist_hold,
                post_hold,
            )
        )
        # 判断是否在无辅助阶段成功保持稳定，允许微小误差。
        self.post_assist_succeeded |= (
            post_hold + 1e-6 >= required_hold
        )

    def finish_episodes(self, env_ids, motion_ids):
        """
        收集这些 episode 是否成功；
        按动作类别分别统计成功率；
        如果所有动作都达标，就降低辅助力阶段。
        """
        # 只在课程学习启用且当前阶段不是最后阶段时才统计。
        if (
            not self.curriculum_enabled
            or self.stage >= len(self.levels) - 1
        ):
            return

        ids = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.env.device,
        )

        # 排除零辅助回合和仍使用旧阶段力的回合。
        eligible = (
            (self.episode_stage[ids] == self.stage)
            & (self.episode_force[ids] > 0)
        )

        # 取出有效 episode 对应的动作编号
        selected_motion_ids = motion_ids[eligible].tolist()
        # 取出这些 episode 是否在无辅助期间成功保持稳定。
        selected_successes = (
            self.post_assist_succeeded[ids][eligible]
            .tolist()
        )

        # 统计每条动作的成功率，若所有动作都达标就降低辅助力阶段。
        for motion_id, succeeded in zip(
            selected_motion_ids, selected_successes
        ):
            # 把结果放入对应动作的队列。
            self.results[motion_id].append(bool(succeeded))

        ready = all(
            len(values) == self.window_size
            and sum(values) / len(values) >= self.threshold
            for values in self.results.values()
        )
        if not ready:
            return

        # 所有动作都达标，降低辅助力阶段。
        old_force = self.levels[self.stage]
        self.stage += 1

        # 新阶段重新收集数据，旧阶段成绩不能继续推动降级。
        for values in self.results.values():
            values.clear()

        print(
            f"[辅助力课程] {old_force:.1f} N → "
            f"{self.levels[self.stage]:.1f} N；"
            "新回合开始采用新阶段。",
            flush=True,
        )

    def _curriculum_settings(self):
        return {
            "levels": self.levels,
            "zero_probability": self.zero_probability,
            "full_until": self.full_until,
            "zero_after": self.zero_after,
            "window_size": self.window_size,
            "threshold": self.threshold,
            "active_motion_ids": self.active_motion_ids,
        }

    def state_dict(self):
        """只保存课程进度，不保存物理状态或当前回合历史。"""
        return {
            "version": 1,
            "settings": self._curriculum_settings(),
            "stage": self.stage,
            "results": {
                motion_id: list(values)
                for motion_id, values in self.results.items()
            },
        }

    def load_state_dict(self, state):
        if state.get("version") != 1:
            raise ValueError("辅助力课程存档版本不匹配")

        if state["settings"] != self._curriculum_settings():
            raise ValueError("辅助力课程配置与存档不一致")

        stage = state["stage"]
        if (
            not isinstance(stage, int)
            or not 0 <= stage < len(self.levels)
        ):
            raise ValueError("存档中的课程阶段无效")

        saved_results = state["results"]
        if set(saved_results) != set(self.results):
            raise ValueError("课程统计中的动作编号不一致")

        for values in saved_results.values():
            if (
                len(values) > self.window_size
                or any(type(value) is not bool for value in values)
            ):
                raise ValueError("课程统计窗口数据无效")

        self.stage = stage
        for motion_id, values in saved_results.items():
            self.results[motion_id].clear()
            self.results[motion_id].extend(values)

        # 仿真状态没有恢复，所有环境按新回合重新采样辅助力。
        self.reset(
            torch.arange(
                self.env.num_envs,
                device=self.env.device,
            )
        )
