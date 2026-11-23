"""根据碰撞几何计算必要身体部位的离地距离。"""

import torch
from torch import nn

from .state import unpack_state
from ...utils.rotation import (
    inverse_rotate_vector,
    normalize_quaternion,
)


CONTACT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "head_yaw_link",
    "head_pitch_link",
)


class GroundContactGeometry(nn.Module):
    """输出 [B, 6] 离地距离，单位米。"""

    def __init__(self, spec, kinematics, vertices):
        super().__init__()

        if tuple(kinematics.body_names) != CONTACT_BODY_NAMES:
            raise ValueError("接触几何的身体顺序不一致")

        self.spec = spec
        self.kinematics = kinematics

        for index, name in enumerate(CONTACT_BODY_NAMES):
            points = torch.as_tensor(
                vertices[name],
                dtype=torch.float32,
            )

            if (
                points.ndim != 2
                or points.shape[1] != 3
                or points.shape[0] < 4
                or not bool(torch.isfinite(points).all())
            ):
                raise ValueError(
                    f"碰撞顶点必须为有限的 [V, 3]：{name}"
                )

            self.register_buffer(
                f"vertices_{index}",
                points.detach().clone(),
            )

    def forward(self, states):
        fields = unpack_state(states, self.spec)

        poses, _ = self.kinematics(
            root_pose=fields["root_pose"],
            root_velocity=fields["root_velocity"],
            joint_position=fields["joint_position"],
            joint_velocity=fields["joint_velocity"],
        )

        world_up = torch.zeros_like(poses[..., :3])
        world_up[..., 2] = 1.0

        local_up = inverse_rotate_vector(
            normalize_quaternion(poses[..., 3:7]),
            world_up,
        )

        clearances = []

        for index in range(len(CONTACT_BODY_NAMES)):
            vertices = getattr(
                self,
                f"vertices_{index}",
            ).to(poses)

            lowest_offset = (
                local_up[:, index] @ vertices.T
            ).amin(dim=-1)

            clearances.append(
                poses[:, index, 2] + lowest_offset
            )

        return torch.stack(clearances, dim=-1)