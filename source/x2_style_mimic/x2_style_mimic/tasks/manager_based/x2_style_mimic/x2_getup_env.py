"""新版多动作环境：保留 AMP 终止状态，并输出独立成功统计。"""

import torch
import math

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .mdp.amp import (
    AMP_KEY_BODY_NAMES,
    AMP_STATE_DIM,
    amp_observation,
)
from .mdp.getup_assist import GetUpAssist
from .x2_style_mimic_env_cfg import POLICY_JOINT_NAMES
from .mdp.self_contact import (
    reset_ground_contact_sensors,
)
from .mdp.getup import ground_contact_force

class X2AMPGetUpEnv(ManagerBasedRLEnv):
    """保留普通环境返回值，额外返回 AMP 的真实步后状态。"""

    def step(self, action):
        # 势函数奖励需要本步动作执行前的状态。
        # 放在共同父类里，X2MultiMotionGetUpEnv 和 X2SuperMimicEnv 都会经过。
        for term_name in self.reward_manager.active_terms:
            term_cfg = self.reward_manager.get_term_cfg(term_name)

            capture = getattr(
                term_cfg.func,
                "capture_before_step",
                None,
            )

            if term_cfg.weight != 0.0 and capture is not None:
                capture(self)
        if not hasattr(self, "_amp_asset_cfg"):
            self._amp_asset_cfg = SceneEntityCfg(
                "robot",
                joint_names=POLICY_JOINT_NAMES,
                body_names=list(AMP_KEY_BODY_NAMES),
                preserve_order=True,
            )
            self._amp_asset_cfg.resolve(self.scene)     # 把名称解析成当前场景对象中的实际索引。

            self._amp_terminal_state = torch.empty(
                self.num_envs,
                AMP_STATE_DIM,
                device=self.device,
            )

            self._amp_terminal_mask = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )

        self._amp_terminal_mask.zero_()                 # 每个环境步开始，清空本步捕获标志
        self._amp_capture_active = True

        try:
            result = super().step(action)
        finally:
            self._amp_capture_active = False            # 关闭捕获开关。这样如果父类执行中出错，也不会让环境一直残留

        observations, rewards, terminated, truncated, extras = result

        done = terminated | truncated           # 合并两种结束标志

        if not torch.equal(
            done,
            self._amp_terminal_mask,
        ):
            raise RuntimeError("AMP 终止状态捕获与环境重置不一致")

        # 普通环境使用步后观测。
        # 已终止的环境使用重置前保存的真实终止状态。
        following = observations["amp"].clone()
        following[done] = self._amp_terminal_state[done]

        extras["amp_next_state"] = following
        extras["amp_terminal_mask"] = (self._amp_terminal_mask.clone())

        return (
            observations,
            rewards,
            terminated,
            truncated,
            extras,
        )

    def _reset_idx(self, env_ids):
        # 只有动作执行过程中触发的自动重置才需要捕获。
        # 初始化和手动 reset 不属于动作转移。
        if getattr(self, "_amp_capture_active", False):
            state = amp_observation(
                self,
                self._amp_asset_cfg,
            )

            self._amp_terminal_state[env_ids] = state[env_ids]
            self._amp_terminal_mask[env_ids] = True

        # 捕获完成后，执行原有重置流程。
        super()._reset_idx(env_ids)
        # 旧回合终止状态已捕获。
        # 新回合不继承旧的过滤地面接触历史。
        reset_ground_contact_sensors(
            self,
            env_ids,
        )

class X2MultiMotionGetUpEnv(X2AMPGetUpEnv):
    def _ensure_amp_stage(self):
        """延迟初始化，兼容构造环境时调用 reset。"""
        if hasattr(self, "_amp_standing_stage"):
            return

        cfg = self.cfg

        values = (
            cfg.amp_stage_enter_height,
            cfg.amp_stage_exit_height,
            cfg.amp_stage_enter_tilt_deg,
            cfg.amp_stage_enter_duration_s,
            cfg.amp_stage_exit_duration_s,
            cfg.amp_stage_fade_s,
        )
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("AMP 阶段参数必须为有限数")
        if not (
            0.0 < cfg.amp_stage_exit_height
            < cfg.amp_stage_enter_height
        ):
            raise ValueError("AMP 退出高度必须低于进入高度")
        if not 0.0 < cfg.amp_stage_enter_tilt_deg < 90.0:
            raise ValueError("AMP 进入倾角必须在 (0, 90) 度")
        if min(
            cfg.amp_stage_enter_duration_s,
            cfg.amp_stage_exit_duration_s,
            cfg.amp_stage_fade_s,
        ) <= 0.0:
            raise ValueError("AMP 阶段持续时间必须大于零")

        self._amp_stage_torso_cfg = SceneEntityCfg(
            "robot",
            body_names=["torso_link"],
            preserve_order=True,
        )
        self._amp_stage_torso_cfg.resolve(self.scene)

        self._amp_stage_foot_cfgs = (
            SceneEntityCfg("left_foot_ground_contact"),
            SceneEntityCfg("right_foot_ground_contact"),
        )

        with torch.inference_mode(False):
            self._amp_standing_stage = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
            self._amp_stage_enter_s = torch.zeros(
                self.num_envs, device=self.device
            )
            self._amp_stage_exit_s = torch.zeros(
                self.num_envs, device=self.device
            )
            self._amp_stage_blend = torch.zeros(
                self.num_envs, device=self.device
            )

    @torch.no_grad()
    def _update_amp_stage(self, env_ids):
        """每个环境每个控制步调用一次，返回当前奖励门控。"""
        self._ensure_amp_stage()

        if len(env_ids) == 0:
            return self._amp_stage_blend[env_ids]

        robot = self.scene["robot"]
        cfg = self.cfg
        dt = float(self.step_dt)

        height = (
            robot.data.root_pos_w.torch[env_ids, 2]
            - self.scene.env_origins[env_ids, 2]
        )

        root_up = (
            -robot.data.projected_gravity_b.torch[env_ids, 2]
        )

        torso_q = robot.data.body_link_pose_w.torch[
            :, self._amp_stage_torso_cfg.body_ids, 3:7
        ][env_ids]
        torso_q = torso_q / torso_q.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        # 当前项目四元数为 xyzw。
        torso_up = (
            1.0 - 2.0 * torso_q[..., :2].square().sum(dim=-1)
        ).amin(dim=-1)

        both_feet = torch.stack(
            [
                ground_contact_force(self, foot_cfg)[env_ids] > 5.0
                for foot_cfg in self._amp_stage_foot_cfgs
            ],
            dim=-1,
        ).all(dim=-1)

        up_threshold = math.cos(
            math.radians(cfg.amp_stage_enter_tilt_deg)
        )

        enter_condition = (
            (height >= cfg.amp_stage_enter_height)
            & (root_up >= up_threshold)
            & (torso_up >= up_threshold)
            & both_feet
        )

        # 退出不依赖速度或短暂接触变化。
        exit_condition = height < cfg.amp_stage_exit_height

        was_standing = self._amp_standing_stage[env_ids]

        enter_s = torch.where(
            (~was_standing) & enter_condition,
            self._amp_stage_enter_s[env_ids] + dt,
            0.0,
        )
        exit_s = torch.where(
            was_standing & exit_condition,
            self._amp_stage_exit_s[env_ids] + dt,
            0.0,
        )

        standing = (
            was_standing
            | (enter_s + 1e-6 >= cfg.amp_stage_enter_duration_s)
        )
        standing &= ~(
            exit_s + 1e-6 >= cfg.amp_stage_exit_duration_s
        )

        self._amp_standing_stage[env_ids] = standing
        self._amp_stage_enter_s[env_ids] = torch.where(
            standing, 0.0, enter_s
        )
        self._amp_stage_exit_s[env_ids] = torch.where(
            standing, exit_s, 0.0
        )

        # blend=0 对应起身阶段，blend=1 对应站立阶段。
        direction = torch.where(standing, 1.0, -1.0)
        blend = (
            self._amp_stage_blend[env_ids]
            + direction * dt / cfg.amp_stage_fade_s
        ).clamp(0.0, 1.0)

        self._amp_stage_blend[env_ids] = blend
        smooth_blend = blend.square() * (3.0 - 2.0 * blend)

        return 1.0 - smooth_blend

    def step(self, action):
        self._ensure_amp_stage()

        # 自动重置前保存终止步结果，避免使用下一回合的姿态。
        self._amp_stage_terminal_gate = torch.ones(
            self.num_envs, device=self.device
        )
        self._amp_stage_terminal_standing = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        # 必须在 step 前保存：自动重置会换成下一回合的动作编号。
        motion_ids = (
            self._getup_reference_motion_ids.clone()
        )
        birth_types = self._getup_birth_types.clone()
        # 必须在 step 前保存，自动重置后这些值属于下一回合。
        episode_force_cap = self._getup_assist.episode_force.clone()
        bank_row_ids = self._getup_bank_row_ids.clone()
        recovery_row_ids = self._getup_recovery_row_ids.clone()

        monitor = self.scene.sensors["self_contact"]
        if self._physics_handles_decimation:
            raise RuntimeError("当前监测实现需要逐物理子步 scene.update；请先处理后端代管 decimation 的兼容性")

        monitor.begin_control_window()

        # 保存自动重置前的终止步监测值。
        self._self_contact_terminal = {
            name: torch.zeros_like(value)
            for name, value in monitor.snapshot().items()
        }
        self._self_contact_terminal_mask = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._getup_assist.apply()

        try:
            result = super().step(action)
        finally:
            monitor.end_control_window()

        observations, rewards, terminated, truncated, extras = result
        contact_metrics = monitor.snapshot()

        # 已自动重置的环境，其传感器缓存已经清零。
        # 用 _reset_idx 中保存的终止步数据替换。
        terminal_mask = self._self_contact_terminal_mask

        for name, value in contact_metrics.items():
            value[terminal_mask] = (
                self._self_contact_terminal[name][terminal_mask]
            )

        updates = contact_metrics["updates_per_control_step"]

        if not bool((updates == self.cfg.decimation).all()):
            raise RuntimeError("自接触采样次数不等于 decimation，存在漏采或重复采样")

        extras["self_contact"] = contact_metrics

        log = extras.setdefault("log", {})
        for name, value in contact_metrics.items():
            log[f"SelfContact/{name}"] = value.mean().item()


        # 计时器在终止判断阶段保存了重置前的快照。
        snapshot = self._getup_progress_snapshot
        extras["getup_progress"] = {
            key: value.clone()
            for key, value in snapshot.items()
        }
        extras["getup_progress"]["motion_ids"] = motion_ids
        extras["getup_progress"]["birth_types"] = birth_types
        extras["getup_progress"]["bank_row_ids"] = bank_row_ids
        extras["getup_progress"]["recovery_row_ids"] = recovery_row_ids

        progress_cfg = self.termination_manager.get_term_cfg("progress")
        required_hold = float(progress_cfg.params["hold_time"])
        extras["getup_progress"]["zero_assist_episode"] = (episode_force_cap == 0.0)
        extras["getup_progress"]["stable_at_end"] = (
            snapshot["hold_seconds"] + 1e-6 >= required_hold
        )

        # 已结束环境的姿态已经重置，使用重置前保存的门控。
        done = terminated | truncated
        live_ids = (~done).nonzero(as_tuple=False).flatten()

        gate = self._amp_stage_terminal_gate.clone()
        standing_stage = self._amp_stage_terminal_standing.clone()

        gate[live_ids] = self._update_amp_stage(live_ids)
        standing_stage[live_ids] = (
            self._amp_standing_stage[live_ids]
        )

        extras["amp_reward_gate"] = gate

        # 本轮仅改奖励门控，判别器样本门控保持原样。
        sample_max_hold = float(
            self.cfg.amp_policy_sample_max_hold_s
        )
        if not (
            math.isfinite(sample_max_hold)
            and sample_max_hold >= 0.0
        ):
            raise ValueError("AMP 样本门控时长必须为非负有限数")

        hold = snapshot["hold_seconds"]
        extras["amp_policy_sample_mask"] = (
            hold <= sample_max_hold
        )

        log["AMPStage/standing_fraction"] = (
            standing_stage.float().mean().item()
        )
        log["AMPStage/reward_gate_mean"] = gate.mean().item()

        return (
            observations,
            rewards,
            terminated,
            truncated,
            extras,
        )

    def _reset_idx(self, env_ids):
        self._ensure_amp_stage()

        if (
            getattr(self, "_amp_capture_active", False)
            and len(env_ids) > 0
        ):
            self._amp_stage_terminal_gate[env_ids] = (
                self._update_amp_stage(env_ids)
            )
            self._amp_stage_terminal_standing[env_ids] = (
                self._amp_standing_stage[env_ids]
            )

        episode_stats = None

        if (
            getattr(self, "_amp_capture_active", False)
            and hasattr(self, "_self_contact_terminal")
            and len(env_ids) > 0
        ):
            terminal_contact = (
                self.scene.sensors["self_contact"].snapshot(env_ids)
            )
            for name, value in terminal_contact.items():
                self._self_contact_terminal[name][env_ids] = value
            self._self_contact_terminal_mask[env_ids] = True

        # 初始化或手动 reset 不算已完成的训练回合。
        # 只在真实 step 引起自动重置时统计。
        if (
            getattr(self, "_amp_capture_active", False)
            and len(env_ids) > 0
            and hasattr(self, "_getup_progress_snapshot")
        ):
            snapshot = self._getup_progress_snapshot

            episode_stats = {
                key: value[env_ids].clone()
                for key, value in snapshot.items()
            }
            episode_stats["motion_ids"] = (
                self._getup_reference_motion_ids[
                    env_ids
                ].clone()
            )
            episode_stats["birth_types"] = (
                self._getup_birth_types[env_ids].clone()
            )
            episode_stats["bank_row_ids"] = (
                self._getup_bank_row_ids[env_ids].clone()
            )
            episode_stats["recovery_row_ids"] = (
                self._getup_recovery_row_ids[env_ids].clone()
            )

        if episode_stats is not None:
            self._getup_assist.finish_episodes(
                env_ids,
                episode_stats["motion_ids"],
                episode_stats["hold_seconds"],
            )

        # 父类先捕获 AMP 终止状态，再执行实际重置。
        super()._reset_idx(env_ids)

        # 新回合不能继承上一回合的阶段和淡出进度。
        self._amp_standing_stage[env_ids] = False
        self._amp_stage_enter_s[env_ids] = 0.0
        self._amp_stage_exit_s[env_ids] = 0.0
        self._amp_stage_blend[env_ids] = 0.0

        # 初始化时也会执行 reset，因此在这里延迟创建。
        if not hasattr(self, "_getup_assist"):
            with torch.inference_mode(False):
                self._getup_assist = GetUpAssist(self)

        self._getup_assist.reset(env_ids)

        if episode_stats is None:
            return

        # 父类重置会重建 extras["log"]，所以在它之后写入。
        log = self.extras.setdefault("log", {})
        log["Assist/stage"] = float(
            self._getup_assist.stage
        )
        log["Assist/force_cap_n"] = float(
            self._getup_assist.levels[
                self._getup_assist.stage
            ]
        )

        if "post_assist_succeeded" in episode_stats:
            post_success = episode_stats[
                "post_assist_succeeded"
            ]
            log["Assist/post_withdraw_success_rate"] = (
                post_success.float().mean().item()
            )
            log["Assist/max_post_withdraw_hold"] = (
                episode_stats[
                    "max_post_assist_hold"
                ].mean().item()
            )

            zero = (
                episode_stats["assist_episode_force_n"] == 0
            )
            if bool(zero.any()):
                log["Assist/zero_force_success_rate"] = (
                    post_success[zero].float().mean().item()
                )
        succeeded = episode_stats["succeeded"]
        first_time = episode_stats["first_success_time"]
        max_hold = episode_stats["max_hold_seconds"]
        final_hold = episode_stats["hold_seconds"]
        motion_ids = episode_stats["motion_ids"]

        log["GetUp/success_rate"] = (
            succeeded.float().mean().item()
        )
        log["GetUp/max_hold_seconds"] = (
            max_hold.mean().item()
        )
        log["GetUp/final_hold_seconds"] = (
            final_hold.mean().item()
        )

        required_hold = float(self.cfg.terminations.progress.params["hold_time"])
        birth_names = (
            "reference_initial",
            "perturbed_initial",
            "reference_rsi",
            "simulated_fall",
            "pre_entrapment",
        )
        for birth_id, birth_name in enumerate(birth_names):
            selected = episode_stats["birth_types"] == birth_id
            if not bool(selected.any()):
                continue
            prefix = f"GetUp/{birth_name}"
            log[f"{prefix}/success_rate"] = (
                succeeded[selected].float().mean().item()
            )

            log[f"{prefix}/stable_at_end_rate"] = (
                (final_hold[selected] + 1e-6 >= required_hold)
                .float()
                .mean()
                .item()
            )

            log[f"{prefix}/max_hold_seconds"] = (
                max_hold[selected].mean().item()
            )

            log[f"{prefix}/final_hold_seconds"] = (
                final_hold[selected].mean().item()
            )

        if "assist_episode_force_n" in episode_stats:
            dynamic = episode_stats["birth_types"] == 4
            force_cap = episode_stats["assist_episode_force_n"]

            for label, selected in (
                ("zero_assist", dynamic & (force_cap == 0.0)),
                ("assist_enabled", dynamic & (force_cap > 0.0)),
            ):
                if not bool(selected.any()):
                    continue

                prefix = f"GetUp/pre_entrapment/{label}"
                log[f"{prefix}/success_rate"] = (
                    succeeded[selected].float().mean().item()
                )
                log[f"{prefix}/stable_at_end_rate"] = (
                    (final_hold[selected] + 1e-6 >= required_hold)
                    .float()
                    .mean()
                    .item()
                )

        # 只对成功回合计算首次成功时间。
        # 本批没有成功回合时不写该指标。
        if bool(succeeded.any()):
            log["GetUp/first_success_seconds"] = (
                first_time[succeeded].mean().item()
            )

        # 按参考动作分别统计，避免只学会其中某一条。
        for motion_id in range(
            len(self._getup_reference_motion_paths)
        ):
            selected = motion_ids == motion_id

            if not bool(selected.any()):
                continue

            prefix = f"GetUp/motion_{motion_id}"
            log[f"{prefix}/success_rate"] = (
                succeeded[selected].float().mean().item()
            )
            log[f"{prefix}/max_hold_seconds"] = (
                max_hold[selected].mean().item()
            )