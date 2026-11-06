"""起身辅助力：按回合采样，按骨盆高度衰减，完全撤力后锁定。"""

from collections import deque
import torch


class GetUpAssist:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]

        self.levels = tuple(env.cfg.assist_force_levels)    # 读取按阶段递减的辅助力上限，单位牛顿。
        self.stage = int(env.cfg.assist_initial_stage)      # 读取初始阶段索引，必须在 [0, len(levels)) 范围内。
        self.zero_probability = float(env.cfg.assist_zero_probability)  # 读取零辅助概率，范围 [0, 1]。
        self.full_below_height = float(env.cfg.assist_full_below_height)# 读取最大辅助力保持高度，单位米
        self.zero_above_height = float(env.cfg.assist_zero_above_height)# 读取辅助力完全撤除高度，单位米。

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
        if not 0.0 <= self.full_below_height < self.zero_above_height < float("inf"):
            raise ValueError("辅助力保持/撤除高度无效")

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
        self.withdrawn = torch.zeros(           # 达到撤力高度后，本回合不再恢复辅助。
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        self.withdraw_time = torch.full(        # 完全撤力开始时刻；尚未撤力时为正无穷。
            (env.num_envs,),
            float("inf"),
            device=env.device,
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
        self.final_threshold = float(                 # 读取课程最终成功率阈值，范围 (0, 1]，若最近若干回合的成功率超过该阈值则降低辅助力。
            env.cfg.assist_curriculum_final_threshold
        )
        self.required_hold = float(             # 与正式成功判定共用连续稳定时间要求。
            env.cfg.terminations.progress.params["hold_time"]
        )

        if not 0.0 < self.final_threshold <= 1.0:
            raise ValueError("课程末态稳定阈值必须在 (0, 1]")
        if not 0.0 < self.required_hold < float("inf"):
            raise ValueError("连续稳定时间必须为有限正数")
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
        self.episode_stage[ids] = self.stage                # 记录当前阶段索引，便于课程学习统计每条动作的成功率。
        self.applied_force[ids] = 0.0                       # 新回合开始时，当前施加的辅助力为零。
        no_assist = self.episode_force[ids] == 0.0          # 标记本回合是否为零辅助力。
        self.withdrawn[ids] = no_assist
        self.withdraw_time[ids] = torch.where(
            no_assist,
            torch.zeros_like(self.episode_force[ids]),        # 已经撤力的回合，撤力时间为 0
            torch.full_like(self.episode_force[ids], float("inf")),          # 尚未撤力的回合
        )

        self.post_assist_succeeded[ids] = False
        self.max_post_assist_hold[ids] = 0.0

    @torch.no_grad()
    def apply(self):
        """控制步开始时，根据骨盆高度更新向上辅助力。"""
        elapsed = (
            self.env.episode_length_buf
            * self.env.step_dt
        )
        height = self.robot.data.root_pos_w.torch[:, 2] - self.env.scene.env_origins[:, 2]
        # 第一次达到撤力高度时，记录时刻并锁定
        newly_withdrawn = ~self.withdrawn & (height >= self.zero_above_height)
        self.withdraw_time[newly_withdrawn] = elapsed[newly_withdrawn]
        self.withdrawn |= newly_withdrawn
        # 计算当前骨盆高度对应的辅助力衰减比例，范围 [0, 1]，在 full_below_height 以下为 1，在 zero_above_height 以上为 0。
        progress = (
            (height - self.full_below_height) / (self.zero_above_height - self.full_below_height)
        ).clamp(0.0, 1.0)
        # 从 1 平滑下降到 0。
        fade = 1.0 - progress.square() * (3.0 - 2.0 * progress)
        self.applied_force.copy_(
            torch.where(
                self.withdrawn,
                torch.zeros_like(self.applied_force),
                self.episode_force * fade,
            )
        )
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

        # 仅从实际完全撤力的时刻开始计时。
        zero_duration = torch.where(
            self.withdrawn,
            (elapsed - self.withdraw_time).clamp_min(0.0),
            torch.zeros_like(elapsed),
        )

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

    def finish_episodes(self, env_ids, motion_ids, final_hold):
        """仅用首帧出生、整回合零辅助的表现推动课程降档。"""
        # 只在课程学习启用且当前阶段不是最后阶段时才统计。
        if (not self.curriculum_enabled or self.stage >= len(self.levels) - 1):
            return

        ids = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.env.device,
        )
        frame_ids = self.env._getup_reference_frame_ids[ids]        # 取出每个环境的出生帧编号，便于判断是否为首帧出生。
        # 只统计当前阶段、首帧出生、整回合零辅助的回合。
        eligible = (
            (self.episode_stage[ids] == self.stage)
            & (self.episode_force[ids] == 0)
            & (frame_ids == 0)
        )
        # 取出有效 episode 对应的动作编号
        selected_motion_ids = motion_ids[eligible].tolist()
        # 取出这些 episode 是否在无辅助期间成功保持稳定。
        selected_successes = (
            self.post_assist_succeeded[ids][eligible]
            .tolist()
        )
        # 取出这些 episode 在最终阶段是否成功保持稳定。
        selected_final_successes = (
            final_hold[eligible] + 1e-6 >= self.required_hold
        ).tolist()
        # 统计每条动作的成功率，若所有动作都达标就降低辅助力阶段。
        for motion_id, succeeded, final_succeeded in zip(
            selected_motion_ids, selected_successes, selected_final_successes
        ):
            # 把结果放入对应动作的队列。
            self.results[motion_id].append(
                (bool(succeeded), bool(final_succeeded))
            )
        ready = all(
            len(values) == self.window_size                                         # 队列已满，才有资格推动课程进度
            and sum(row[0] for row in values) / len(values) >= self.threshold       # 队列中“完全无辅助力期间成功保持稳定”的比例达标
            and sum(row[1] for row in values) / len(values) >= self.final_threshold # 队列中“最终阶段成功保持稳定”的比例达标
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
            "full_height": self.full_below_height,
            "zero_height": self.zero_above_height,
            "window_size": self.window_size,
            "threshold": self.threshold,
            "final_threshold": self.final_threshold,
            "required_hold": self.required_hold,
            "evaluation": "initial_frame_zero_assist",
            "active_motion_ids": self.active_motion_ids,
        }

    def state_dict(self):
        """只保存课程进度，不保存物理状态或当前回合历史。"""
        return {
            "version": 2,
            "settings": self._curriculum_settings(),
            "stage": self.stage,
            "results": {
                motion_id: list(values)
                for motion_id, values in self.results.items()
            },
        }

    def load_state_dict(self, state):
        if state.get("version") != 2:
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
            if len(values) > self.window_size:
                raise ValueError("课程统计窗口长度无效")

            for row in values:
                if (
                    not isinstance(row, (tuple, list))
                    or len(row) != 2
                    or any(type(value) is not bool for value in row)
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
