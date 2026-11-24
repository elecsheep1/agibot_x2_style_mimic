"""广义坐标动力学表示；完整状态和旋转工具统一使用 XYZW。"""

import math

import torch
from torch import nn

from .state import pack_state, unpack_state
from ...utils.rotation import (
    normalize_quaternion,
    quaternion_multiply,
    rotation_vector_to_quaternion,
    rotate_vector,
    inverse_rotate_vector,
)


class GenCoordRepresentation(nn.Module):
    """模型局部状态：

    高度、局部重力、局部线速度、局部角速度、
    关节位置、关节速度。

    根位置和根四元数通过辅助状态保存。
    历史动作、历史执行目标由外层解析维护。
    """

    def __init__(
        self,
        state_spec,
        dt,
        decimation,
    ):
        super().__init__()

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("控制步长 dt 必须为有限正数")
        if (
            isinstance(decimation, bool)
            or not isinstance(decimation, int)
            or decimation < 1
        ):
            raise ValueError("decimation 必须为正整数")

        self.spec = state_spec
        self.dt = float(dt)
        self.decimation = decimation

        # 计算物理子步积分系数，用于将子步速度积分到控制步。
        self.substep_coeff = (
            (decimation + 1)
            / (2 * decimation)
        )

        # 计算模型局部状态维度。
        self.rep_state_dim = (
            10 + 2 * state_spec.num_joints
        )
        # 计算网络预测输出的维数。
        self.prediction_dim = (
            6 + state_spec.num_joints
        )

    def represent(self, full_state):
        """完整状态转换为模型局部状态。"""

        fields = unpack_state(full_state, self.spec)

        quaternion = normalize_quaternion(fields["root_pose"][:, 3:7])
        velocity = fields["root_velocity"]
        gravity = torch.zeros_like(velocity[:, :3])
        gravity[:, 2] = -1.0

        return torch.cat(
            (
                fields["root_pose"][:, 2:3],
                inverse_rotate_vector(
                    quaternion,
                    gravity,
                ),
                inverse_rotate_vector(
                    quaternion,
                    velocity[:, :3],
                ),
                inverse_rotate_vector(
                    quaternion,
                    velocity[:, 3:6],
                ),
                fields["joint_position"],
                fields["joint_velocity"],
            ),
            dim=-1,
        )

    def extract_aux_state(self, full_state):
        """辅助状态保存根位置与 XYZW 四元数。"""

        fields = unpack_state(full_state, self.spec)

        return torch.cat(
            (
                fields["root_pose"][:, :3],
                normalize_quaternion(
                    fields["root_pose"][:, 3:7]
                ),
            ),
            dim=-1,
        )

    def decompose(self, rep_state):
        """将模型局部状态拆分为各个字段。"""
        if (
            rep_state.ndim != 2
            or rep_state.shape[-1]
            != self.rep_state_dim
        ):
            raise ValueError("模型局部状态必须为 [B, rep_state_dim]")

        return torch.split(
            rep_state,
            (
                1,
                3,
                3,
                3,
                self.spec.num_joints,
                self.spec.num_joints,
            ),
            dim=-1,
        )

    def apply_prediction(
        self,
        aux_state,          # 根部位置、根部姿态四元数
        rep_state,          # 模型局部状态
        prediction,
    ):
        """速度增量在当前根坐标系表达。

        积分位置和姿态后，将末端速度重新表达为
        下一时刻根坐标系下的速度。
        """

        (
            _,
            _,
            linear,
            angular,
            joint_position,
            joint_velocity,
        ) = self.decompose(rep_state)

        batch_size = rep_state.shape[0]

        if aux_state.shape != (batch_size, 7):
            raise ValueError("辅助根状态必须为 [B, 7]")
        if prediction.shape != (
            batch_size,
            self.prediction_dim,
        ):
            raise ValueError("速度增量必须为 [B, prediction_dim]")

        (
            delta_linear,
            delta_angular,
            delta_joint,
        ) = torch.split(
            prediction,
            (
                3,
                3,
                self.spec.num_joints,
            ),
            dim=-1,
        )

        quaternion = normalize_quaternion(aux_state[:, 3:7])
        coefficient = self.substep_coeff

        # 下一步末端速度：
        # 先从当前根坐标系转换到世界坐标系。
        world_linear = rotate_vector(quaternion, linear + delta_linear)
        world_angular = rotate_vector(quaternion, angular + delta_angular)
        next_joint_velocity = joint_velocity + delta_joint

        # 位姿积分使用子步加权速度。
        pose_linear = rotate_vector(
            quaternion,
            linear + coefficient * delta_linear,
        )
        pose_angular = rotate_vector(
            quaternion,
            angular + coefficient * delta_angular,
        )

        next_position = (
            aux_state[:, :3]
            + pose_linear * self.dt
        )

        # pose_angular 为世界角速度，因此左乘旋转增量。
        next_quaternion = normalize_quaternion(
            quaternion_multiply(
                rotation_vector_to_quaternion(
                    pose_angular * self.dt
                ),
                quaternion,
            )
        )

        next_joint_position = (
            joint_position
            + (
                joint_velocity
                + coefficient * delta_joint
            )
            * self.dt
        )

        gravity = torch.zeros_like(world_linear)
        gravity[:, 2] = -1.0

        next_aux = torch.cat(
            (
                next_position,
                next_quaternion,
            ),
            dim=-1,
        )

        next_rep = torch.cat(
            (
                next_position[:, 2:3],
                inverse_rotate_vector(
                    next_quaternion,
                    gravity,
                ),
                inverse_rotate_vector(
                    next_quaternion,
                    world_linear,
                ),
                inverse_rotate_vector(
                    next_quaternion,
                    world_angular,
                ),
                next_joint_position,
                next_joint_velocity,
            ),
            dim=-1,
        )

        return next_aux, next_rep

    def reconstruct_full_state(
        self,
        aux_state,
        rep_state,
        previous_action,
        previous_target,
    ):
        """重建现有完整状态，保持奖励和 AMP 的输入接口。"""
        (
            height,
            _,
            linear,
            angular,
            position,
            velocity,
        ) = self.decompose(rep_state)

        batch_size = rep_state.shape[0]

        if aux_state.shape != (batch_size, 7):
            raise ValueError("辅助根状态必须为 [B, 7]")
        if previous_action.shape != (
            batch_size,
            self.spec.num_actions,
        ):
            raise ValueError("历史动作必须为 [B, num_actions]")
        if (
            previous_target.shape
            != previous_action.shape
        ):
            raise ValueError("历史执行目标必须与历史动作同形状")

        quaternion = normalize_quaternion(
            aux_state[:, 3:7]
        )

        return pack_state(
            root_pose=torch.cat(
                (
                    aux_state[:, :2],
                    height,
                    quaternion,
                ),
                dim=-1,
            ),
            root_velocity=torch.cat(
                (
                    rotate_vector(
                        quaternion,
                        linear,
                    ),
                    rotate_vector(
                        quaternion,
                        angular,
                    ),
                ),
                dim=-1,
            ),
            joint_position=position,
            joint_velocity=velocity,
            previous_action=previous_action,
            previous_target=previous_target,
        )