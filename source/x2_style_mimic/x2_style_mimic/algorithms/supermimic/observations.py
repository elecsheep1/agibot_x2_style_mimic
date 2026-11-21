import torch
from torch import nn

from .state import unpack_state
from ...utils.rotation import (
    normalize_quaternion,
    inverse_rotate_vector,
)


class ObservationNormalizer(nn.Module):
    """单帧观测归一化；统计量只由真实采样显式更新。"""

    def __init__(
        self,
        dimension,
        minimum_std=0.01,
        clip=10.0,
    ):
        super().__init__()

        self.minimum_std = float(minimum_std)
        self.clip = float(clip)

        self.register_buffer(
            "mean",
            torch.zeros(dimension),
        )
        self.register_buffer(
            "variance",
            torch.ones(dimension),
        )
        self.register_buffer(
            "count",
            torch.zeros((), dtype=torch.float64),
        )

    @torch.no_grad()
    def update(self, values):
        if (
            values.ndim != 2
            or values.shape[1] != self.mean.numel()
        ):
            raise ValueError("观测统计输入必须为 [样本数, 单帧维度]")

        if values.shape[0] == 0:
            return

        values = values.detach().to(self.mean)

        if not bool(torch.isfinite(values).all()):
            raise ValueError("观测统计输入包含 NaN/Inf")

        batch_mean = values.mean(dim=0)
        batch_variance = values.var(dim=0, unbiased=False)
        # 把本批数据的样本数量转换成一个标量张量，用于更新归一化统计量。
        batch_count = self.count.new_tensor(
            values.shape[0]
        )

        total = self.count + batch_count
        ratio = (
            batch_count / total
        ).to(self.mean.dtype)

        delta = batch_mean - self.mean

        # 合并已有样本与当前批次的总体方差。
        self.variance.copy_(
            (1.0 - ratio) * self.variance
            + ratio * batch_variance
            + ratio * (1.0 - ratio) * delta.square()
        )

        self.mean.add_(ratio * delta)
        self.count.copy_(total)

    def forward(self, values):
        # 避免接近常量的观测产生过大的归一化数值。
        std = self.variance.clamp_min(
            self.minimum_std ** 2
        ).sqrt()

        return (
            (values - self.mean) / std
        ).clamp(
            -self.clip,
            self.clip,
        )


class Proprioception(nn.Module):
    """本体感知观测模块,将历史状态转换为本体感知观测。"""

    def __init__(
        self,
        spec,
        joint_ids,
        default_joint_position,
        default_joint_velocity,
        history_length=5,
    ):
        super().__init__()

        self.spec = spec
        self.history_length = int(history_length)

        self.register_buffer(
            "joint_ids",
            torch.as_tensor(
                joint_ids,
                dtype=torch.long
            ),
        )
        self.register_buffer(
            "default_joint_position",
            default_joint_position.detach().clone(),
        )
        self.register_buffer(
            "default_joint_velocity",
            default_joint_velocity.detach().clone(),
        )

        # 计算 actor 本体感知的维度
        # 单帧观测维度
        self.frame_dim = (
            6 + 3 * self.joint_ids.numel()
        )
        self.output_dim = (
            self.frame_dim * self.history_length
        )
        self.normalizer = ObservationNormalizer(
            self.frame_dim
        )

    @torch.no_grad()
    def update_normalization(self, real_states):
        """只接收真实采样状态，不接收想象状态。"""
        self.normalizer.update(
            self.frame(real_states)
        )

    def frame(self, state):
        """
        从单帧状态中提取本体感知观测,并和历史拼接。
        Args:
            state: 状态张量，形状为 [批量, 状态维度]。
        Returns:
            本体感知观测张量，形状为 [批量, 本体感知维度]。
        """
        fields = unpack_state(state, self.spec)

        quaternion = normalize_quaternion(
            fields["root_pose"][:, 3:7]
        )
        gravity = torch.zeros_like(
            fields["root_velocity"][:, :3]
        )
        # 这里表示单位重力方向，用于计算机器人姿态，所以不取9.81
        gravity[:, 2] = -1.0
        angular_velocity = inverse_rotate_vector(
            quaternion,
            fields["root_velocity"][:, 3:6],
        )
        projected_gravity = inverse_rotate_vector(
            quaternion,
            gravity,
        )

        joint_position = (
            fields["joint_position"][:, self.joint_ids]
            - self.default_joint_position
        )
        joint_velocity = (
            fields["joint_velocity"][:, self.joint_ids]
            - self.default_joint_velocity
        )

        return torch.cat(
            (
                angular_velocity,           # 3维机身角速度
                projected_gravity,          # 3维机身坐标系下的重力方向
                joint_position,             # 关节位置偏差
                joint_velocity,             # 关节速度偏差
                fields["previous_action"],  # 上一时刻动作
            ),
            dim=-1,
        )

    def from_states(self, history):
        """将一段历史状态转换成一段历史观测特征。"""
        batch_size, length, state_dim = history.shape

        frames = self.frame(
            history.reshape(-1, state_dim)
        )

        return frames.reshape(
            batch_size,
            length,
            self.frame_dim,
        )

    def flatten(self, frames):
        """
        每个观测项内部按时间拼接。
        为了把每个样本的历史信息整理成一个长向量，方便输入网络。
        后续网络需要按照固定位置来学习各帧之间的关系。
        """
        action_dim = self.spec.num_actions

        terms = torch.split(
            frames,
            (3, 3, action_dim, action_dim, action_dim),
            dim=-1,
        )

        return torch.cat(
            [
                term.flatten(start_dim=1)
                for term in terms       # 依次将每个观测项按时间维度展平并拼接。
            ],
            dim=-1,
        )

    def forward(self, history):
        frames = self.from_states(history)
        normalized_frames = self.normalizer(
            frames
        )
        return self.flatten(
            normalized_frames
        )
