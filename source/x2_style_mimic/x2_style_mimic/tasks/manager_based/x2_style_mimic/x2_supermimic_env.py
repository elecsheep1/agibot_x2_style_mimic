"""SuperMimic 起身环境及真实转移采集。"""

import torch

from x2_style_mimic.algorithms.supermimic.state import (
    StateSpec,
    pack_state,
)

from .x2_getup_env import (
    X2MultiMotionGetUpEnv,
)

from .mdp.supermimic.transition_data import (
    read_task_memory,
    read_contact_targets,
)


class X2SuperMimicEnv(
    X2MultiMotionGetUpEnv
):
    """复用原起身执行流程，额外保存世界模型转移。"""

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
            "contact_targets": read_contact_targets(self),
        }

    def step(self, action):
        before = self.read_world_state()
        sent_action = action.detach().clone()

        task_before = read_task_memory(self)
        # interval 事件在本控制步物理积分之后执行。
        # 记录外部事件造成的速度变化
        self._wm_interval_delta_velocity = torch.zeros(
            self.num_envs,
            6,      # [Δvx, Δvy, Δvz, Δωx, Δωy, Δωz]
            device=self.device,
        )
        # 标记哪些环境发生了推扰
        self._wm_interval_push_mask = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        # 终止环境的辅助任务与接触数据在 reset 前填入。
        self._wm_terminal_auxiliary = None

        if sent_action.shape != (
            self.num_envs,
            self.world_state_spec.num_actions,
        ):
            raise ValueError("动作形状必须为 [环境数量, 控制关节数量]")
        if not torch.isfinite(
            sent_action
        ).all():
            raise ValueError("动作包含 NaN 或无穷值")

        self._wm_terminal_state = (
            torch.empty_like(before)
        )
        self._wm_terminal_mask = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self._wm_capture_active = True

        try:
            result = super().step(
                sent_action
            )
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

        if not torch.equal(
            done,
            self._wm_terminal_mask,
        ):
            raise RuntimeError("世界模型终止状态捕获与环境自动重置不一致")

        following = self.read_world_state()

        # 已结束环境必须使用重置前的最终状态，
        # 不能使用新回合的出生状态。
        following[done] = (
            self._wm_terminal_state[done]
        )

        auxiliary = (
            self._capture_transition_auxiliary()
        )
        # 自动重置已经清除了旧回合任务与接触缓存，
        # 必须换回重置前保存的数据。
        if bool(done.any()):
            if self._wm_terminal_auxiliary is None:
                raise RuntimeError("终止回合缺少任务与接触监督快照")

            # 把已经结束的环境的辅助数据，替换成自动重置之前保存的终止数据。未结束的环境保持不变。
            for group, fields in auxiliary.items():
                    for name, value in fields.items():
                        value[done] = (
                            self._wm_terminal_auxiliary[
                                group
                            ][name][done]
                        )

        # 本步真正参与物理积分的辅助力。
        # 原起身环境已经保留了重置前的逐步快照。
        applied_force = (
            extras["getup_progress"][
                "assist_applied_force_n"
            ]
            .detach()
            .clone()
        )

        # 世界坐标，施加在 torso 质心的力／力矩。
        assist_wrench = torch.zeros(
            self.num_envs,
            6,          # [fx, fy, fz, τx, τy, τz]
            device=self.device,
        )
        assist_wrench[:, 2] = applied_force

        # 自动重置后发生的推扰属于新回合，
        # 不得归到刚结束的旧回合转移。
        delta_velocity = (
            self._wm_interval_delta_velocity
            .detach()
            .clone()
        )
        push_mask = (
            self._wm_interval_push_mask
            .detach()
            .clone()
        )

        delta_velocity[done] = 0.0
        push_mask[done] = False

        # following 是 interval 事件后的状态。
        # 奖励和成功判定使用的是事件前状态。
        reward_state = following.clone()

        # 状态布局：
        # 根位姿 [0:7]，根线／角速度 [7:13]。
        reward_state[:, 7:13] -= delta_velocity

        extras["world_model"] = {
            "state": before,
            "action": sent_action,
            "next_state": following,
            # 原起身 RewardManager 的真实任务奖励。
            # 不包含算法侧额外计算的 AMP 奖励。
            "reward": (
                rewards.detach().clone()
            ),
            "terminated": (
                terminated.detach().clone()
            ),
            "truncated": (
                truncated.detach().clone()
            ),
        }

        extras["world_model"].update(
            {
                "schema_version": 2,
                "task_memory": task_before,
                "next_task_memory": (
                    auxiliary["task_memory"]
                ),

                "contact_targets": (
                    auxiliary["contact_targets"]
                ),

                "assist_wrench_w": assist_wrench,

                "interval_delta_velocity_w": (
                    delta_velocity
                ),
                "interval_push_mask": push_mask,

                "reward_state": reward_state,

                "amp_reward_gate": (
                    extras["amp_reward_gate"]
                    .detach()
                    .clone()
                ),
                "amp_policy_sample_mask": (
                    extras["amp_policy_sample_mask"]
                    .detach()
                    .clone()
                ),
            }
        )

        return (
            observations,
            rewards,
            terminated,
            truncated,
            extras,
        )

    def _reset_idx(self, env_ids):
        if getattr(
            self,
            "_wm_capture_active",
            False,
        ):
            terminal_state = (
                self.read_world_state()
            )

            auxiliary = (
                self._capture_transition_auxiliary()
            )

            if self._wm_terminal_auxiliary is None:
                self._wm_terminal_auxiliary = {
                    group: {
                        name: torch.empty_like(value)
                        for name, value in fields.items()
                    }
                    for group, fields
                    in auxiliary.items()
                }

            for group, fields in auxiliary.items():
                for name, value in fields.items():
                    self._wm_terminal_auxiliary[
                        group
                    ][name][env_ids] = value[env_ids]

            self._wm_terminal_state[
                env_ids
            ] = terminal_state[env_ids]

            self._wm_terminal_mask[
                env_ids
            ] = True

        # 继续执行原起身环境的终止状态捕获、
        # 回合统计、参考重置和辅助状态重置。
        super()._reset_idx(env_ids)