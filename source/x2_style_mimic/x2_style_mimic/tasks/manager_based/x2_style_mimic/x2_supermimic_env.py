"""SuperMimic 起身环境及真实转移采集。"""

import torch

from x2_style_mimic.algorithms.supermimic.state import (
    StateSpec,
    pack_state,
)

from .x2_getup_env import (
    X2AMPGetUpEnv,
)

from .mdp.supermimic.transition_data import (
    read_task_memory,
)


class X2SuperMimicEnv(
    X2AMPGetUpEnv
):
    """
    复用原起身执行流程，额外保存世界模型转移。
    保留：
    自动重置前的真实终止状态。
    AMP 终止状态。
    起身成功统计。
    出生类型分组统计。
    """
    transition_schema_version = 4

    @torch.no_grad()
    def read_world_state(self):
        """读取环境局部坐标下的完整世界模型状态。"""

        robot = self.scene["robot"]
        action_term = (
            self.action_manager.get_term(
                "joint_pos"
            )
        )

        root_pose = (
            robot.data.root_link_pose_w
            .torch.clone()
        )
        root_pose[:, :3] -= (
            self.scene.env_origins
        )

        return pack_state(
            root_pose=root_pose,
            root_velocity=(
                robot.data.root_link_vel_w.torch
            ),
            joint_position=(
                robot.data.joint_pos.torch
            ),
            joint_velocity=(
                robot.data.joint_vel.torch
            ),
            previous_action=(
                action_term.raw_actions
            ),
            previous_target=(
                action_term._previous_target
            ),
        ).detach().clone()

    @property
    def world_state_spec(self):
        robot = self.scene["robot"]
        action_term = (
            self.action_manager.get_term(
                "joint_pos"
            )
        )

        return StateSpec(
            num_joints=(
                robot.data.joint_pos.shape[1]
            ),
            num_actions=(
                action_term.raw_actions.shape[1]
            ),
        )

    @torch.no_grad()
    def _capture_transition_auxiliary(self):
        return {
            "task_memory": read_task_memory(self),
        }

    def step(self, action):
        sent_action = action.detach().clone()

        if sent_action.shape != (
            self.num_envs,
            self.world_state_spec.num_actions,
        ):
            raise ValueError("动作形状必须为 [环境数量, 控制关节数量]")
        if not bool(torch.isfinite(sent_action).all()):
            raise ValueError("动作包含 NaN 或无穷值")

        before = self.read_world_state()
        task_before = read_task_memory(self)

        # 必须在自动重置之前保存出生分组。
        birth_metadata = {
            "motion_ids": (self._getup_reference_motion_ids.clone()),
            "birth_types": self._getup_birth_types.clone(),
            "bank_row_ids": self._getup_bank_row_ids.clone(),
            "recovery_row_ids": (self._getup_recovery_row_ids.clone()),
        }

        self._wm_terminal_state = torch.empty_like(before)
        self._wm_terminal_mask = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._wm_terminal_auxiliary = None

        self._wm_capture_active = True
        try:
            result = super().step(sent_action)
        finally:
            self._wm_capture_active = False

        (
            observations,
            rewards,
            terminated,
            truncated,
            extras,
        ) = result

        done = terminated | truncated

        if not torch.equal(done, self._wm_terminal_mask):
            raise RuntimeError("世界模型终止状态捕获与自动重置不一致")

        following = self.read_world_state()
        following[done] = self._wm_terminal_state[done]

        auxiliary = self._capture_transition_auxiliary()

        if bool(done.any()):
            if self._wm_terminal_auxiliary is None:
                raise RuntimeError("终止回合缺少任务状态快照")

            for group, fields in auxiliary.items():
                for name, value in fields.items():
                    value[done] = (
                        self._wm_terminal_auxiliary[
                            group
                        ][name][done]
                    )

        # GetUpProgress 已经在终止判断阶段保存真实步后数据。
        progress = {
            name: value.clone()
            for name, value
            in self._getup_progress_snapshot.items()
        }
        progress.update(birth_metadata)
        extras["getup_progress"] = progress

        fade_start = float(self.cfg.amp_gate_fade_start_s)
        fade_end = float(self.cfg.amp_gate_fade_end_s)

        if not 0.0 <= fade_start < fade_end:
            raise ValueError("AMP 门控时间范围无效")

        hold = progress["hold_seconds"]
        blend = (
            (hold - fade_start)
            / (fade_end - fade_start)
        ).clamp(0.0, 1.0)

        extras["amp_reward_gate"] = (
            1.0 - blend.square() * (3.0 - 2.0 * blend)
        )
        extras["amp_policy_sample_mask"] = (
            hold <= self.cfg.amp_policy_sample_max_hold_s
        )

        extras["world_model"] = {
            "schema_version": self.transition_schema_version,
            "state": before,
            "action": sent_action,
            "next_state": following,
            "reward": rewards.detach().clone(),
            "terminated": terminated.detach().clone(),
            "truncated": truncated.detach().clone(),
            "task_memory": task_before,
            "next_task_memory": auxiliary["task_memory"],
            "amp_reward_gate": (
                extras["amp_reward_gate"].detach().clone()
            ),
            "amp_policy_sample_mask": (
                extras["amp_policy_sample_mask"]
                .detach()
                .clone()
            ),
        }

        return (
            observations,
            rewards,
            terminated,
            truncated,
            extras,
        )

    def _reset_idx(self, env_ids):
        episode_stats = None

        if (
            getattr(self, "_wm_capture_active", False)
            and len(env_ids) > 0
        ):
            terminal_state = self.read_world_state()
            auxiliary = self._capture_transition_auxiliary()

            self._wm_terminal_state[env_ids] = (
                terminal_state[env_ids]
            )
            self._wm_terminal_mask[env_ids] = True

            if self._wm_terminal_auxiliary is None:
                self._wm_terminal_auxiliary = {
                    group: {
                        name: torch.empty_like(value)
                        for name, value in fields.items()
                    }
                    for group, fields in auxiliary.items()
                }

            for group, fields in auxiliary.items():
                for name, value in fields.items():
                    self._wm_terminal_auxiliary[
                        group
                    ][name][env_ids] = value[env_ids]

            episode_stats = {
                name: value[env_ids].clone()
                for name, value
                in self._getup_progress_snapshot.items()
            }
            episode_stats["birth_types"] = (
                self._getup_birth_types[env_ids].clone()
            )

        # X2AMPGetUpEnv 继续捕获 AMP 终止状态，
        # 然后由 ManagerBasedRLEnv 执行实际重置。
        super()._reset_idx(env_ids)

        if episode_stats is None:
            return

        required_hold = float(
            self.cfg.terminations.progress.params["hold_time"]
        )
        log = self.extras.setdefault("log", {})

        values = {
            "success_rate": (
                episode_stats["succeeded"].float()
            ),
            "stable_at_end_rate": (
                episode_stats["hold_seconds"]
                >= required_hold
            ).float(),
            "max_hold_seconds": (
                episode_stats["max_hold_seconds"]
            ),
            "final_hold_seconds": (
                episode_stats["hold_seconds"]
            ),
        }

        for name, value in values.items():
            log[f"GetUp/{name}"] = value.mean().item()

        birth_names = {
            0: "reference_initial",
            1: "perturbed_initial",
            2: "reference_rsi",
            3: "simulated_fall",
            4: "dynamic_recovery",
        }

        for birth_type, birth_name in birth_names.items():
            mask = (
                episode_stats["birth_types"] == birth_type
            )
            if bool(mask.any()):
                for name, value in values.items():
                    log[
                        f"GetUp/{birth_name}/{name}"
                    ] = value[mask].mean().item()