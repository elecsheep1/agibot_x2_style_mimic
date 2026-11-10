from dataclasses import dataclass

import torch


@dataclass(forzen=True)
class StateSpec:
    """状态规范，定义了状态的形状和数据类型。"""
    num_joints: int
    num_actions: int

    @property
    def dim(self) -> int:
        # root pose: 7
        # root velocity: 6
        # joint position/velocity: 2 * J
        # previous action/target: 2 * A
        return (
            13
            + 2 * self.num_joints
            + 2 * self.num_actions
        )


def pack_state(
    root_pose,
    root_velocity,
    joint_position,
    joint_velocity,
    previous_action,
    previous_target,
):
    """所有字段均为 [Batch, Dim]；root 四元数使用 XYZW。"""
    batch_size = root_pose.shape[0]

    fields = (
        root_pose,
        root_velocity,
        joint_position,
        joint_velocity,
        previous_action,
        previous_target,
    )

    if any(
        value.ndim != 2 or value.shape[0] != batch_size
        for value in fields
    ):
        raise ValueError("状态字段必须具有一致的 batch 维度")
    if root_pose.shape[1] != 7:
        raise ValueError("root_pose 必须为 [B, 7]")
    if root_velocity.shape[1] != 6:
        raise ValueError("root_velocity 必须为 [B, 6]")
    if joint_position.shape != joint_velocity.shape:
        raise ValueError("关节位置和速度形状不一致")
    if previous_action.shape != previous_target.shape:
        raise ValueError("动作和历史目标形状不一致")

    return torch.cat(fields, dim=-1)

def unpack_state(state, spec: StateSpec):
    """将状态向量拆分为各个字段。"""
    if state.ndim != 2 or state.shape[1] != spec.dim:
        raise ValueError("state 形状与 StateSpec 不一致")

    sizes = (
        7,
        6,
        spec.num_joints,
        spec.num_joints,
        spec.num_actions,
        spec.num_actions,
    )

    values = torch.split(state, sizes, dim=-1)      # 拆分为各个字段

    return dict(zip(
        (
            "root_pose",
            "root_velocity",
            "joint_position",
            "joint_velocity",
            "previous_action",
            "previous_target",
        ),
        values,
    ))
