import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class LimitedJointPositionAction(JointPositionAction):
    """限制最终位置目标的范围和每个控制步的变化量。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        if not 0.0 < cfg.target_velocity_fraction <= 1.0:
            raise ValueError("target_velocity_fraction 必须在 (0, 1] 内")

        self._control_dt = env.step_dt

        self._previous_target = (
            self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
        )

        self._processed_actions.copy_(self._previous_target)

    def process_actions(self, actions):
        # 保留原始动作，并执行 scale 和默认姿态偏移。
        super().process_actions(actions)

        limits = self._asset.data.joint_pos_limits.torch[
            :, self._joint_ids
        ]
        lower = limits[..., 0]
        upper = limits[..., 1]

        # 约束的是最终关节目标，单位 rad。
        desired = self._processed_actions.clamp(min=lower, max=upper)
        previous = self._previous_target.clamp(min=lower, max=upper)

        # 各关节分别使用自身速度上限，换算为每个控制步的最大变化量。
        # 速度上限 * 比例 * 控制周期
        max_delta = (
            self._asset.data.joint_vel_limits.torch[:, self._joint_ids]
            * self.cfg.target_velocity_fraction
            * self._control_dt
        )

        delta = (desired - previous).clamp(min=-max_delta, max=max_delta)

        self._processed_actions.copy_(
            (previous + delta).clamp(min=lower, max=upper)
        )
        self._previous_target.copy_(self._processed_actions)

    def reset(self, env_ids=None):
        super().reset(env_ids)

        ids = slice(None) if env_ids is None else env_ids

        position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.joint_pos_limits.torch[:, self._joint_ids]

        # 此时 RSI 已写入参考姿态，从新姿态开始限速。
        target = position[ids].clamp(
            min=limits[ids, :, 0],
            max=limits[ids, :, 1],
        )

        self._previous_target[ids] = target
        self._processed_actions[ids] = target


