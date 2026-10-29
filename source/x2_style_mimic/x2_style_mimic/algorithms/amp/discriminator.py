
"""AMP 动作风格判别器。"""

import torch
from torch import nn

from ...tasks.manager_based.x2_style_mimic.mdp.amp import (
    AMP_STATE_DIM,
)


class AMPDiscriminator(nn.Module):
    """独立于 Actor/Critic 的动作风格判别器。"""

    def __init__(
        self,
        reference_states,       # 整段参考动作提取出来的 AMP 状态
        learning_rate=1e-4,     # 判别器的学习率，默认 0.0001
        gradient_penalty=10.0,  # 梯度惩罚的强度，默认 10
        weight_decay=0.0,       # 权重衰减，默认 0
    ):
        super().__init__()

        if (reference_states.ndim != 2 or reference_states.shape[1] != AMP_STATE_DIM):
            raise ValueError(f"参考状态必须为 [帧数, {AMP_STATE_DIM}]")

        if (len(reference_states) < 2 or not torch.isfinite(reference_states).all().item()):
            raise ValueError("参考状态数量或数值错误")

        states = reference_states.detach().float()

        # v0 使用固定的参考统计量。
        # 参考样本和策略样本共用相同的均值、标准差。
        self.register_buffer(
            "state_mean",
            states.mean(dim=0).clone(),
        )
        self.register_buffer(
            "state_std",
            states.std(
                dim=0,
                unbiased=False,
            ).clamp_min(0.05).clone(),
        )

        # 相邻两帧输入
        self.network = nn.Sequential(
            nn.Linear(2 * AMP_STATE_DIM, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        ).to(states.device)

        self.gradient_penalty = gradient_penalty

        # 创建优化器，优化器负责根据梯度修改网络参数。
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def normalize(self, pairs):
        """相邻两帧分别使用同一套单帧统计量进行归一化。"""
        expected_dim = 2 * AMP_STATE_DIM
        if pairs.ndim != 2 or pairs.shape[1] != expected_dim:
            raise ValueError(f"判别器输入必须为 [批量, {expected_dim}]")

        # 单帧统计量都是 80 维，两帧输入则需要 160 维，所以重复两遍：
        mean = self.state_mean.repeat(2)
        std = self.state_std.repeat(2)

        return (pairs - mean) / std

    def forward(self, pairs):
        """输出判别分数，不使用 Sigmoid。"""
        normalized = self.normalize(pairs)

        return self.network(
            normalized
        ).squeeze(-1)

    @torch.no_grad()
    def style_reward(self, current, following):
        """将相邻状态的判别分数转换为 [0, 1] 风格奖励。"""
        # 拼接
        pairs = torch.cat(
            (current, following),
            dim=-1,
        )

        score = self(pairs)

        # amp奖励计算公式
        reward = (
            1.0
            - 0.25 * (score - 1.0).square()
        )

        return reward.clamp(0.0, 1.0)

    def update(self, reference_pairs, policy_pairs):
        """
        独立更新判别器；不能放在 inference_mode 内调用。
        reference_pairs：参考动作中的相邻状态。
        policy_pairs：机器人实际执行得到的相邻状态。
        """

        # 切断 reference_pairs 之前的计算图，但从新的 reference 开始重新记录梯度。
        reference = self.normalize(
            reference_pairs.detach()
        ).detach().clone()

        #为了计算后面的梯度惩罚，还需要知道“分数对输入有多敏感”，所以设置：
        reference.requires_grad_(True)

        policy = self.normalize(
            policy_pairs.detach()
        ).detach()

        reference_score = self.network(
            reference
        ).squeeze(-1)

        policy_score = self.network(
            policy
        ).squeeze(-1)

        # 最小二乘目标：参考样本 +1，策略样本 -1。
        classification_loss = (
            (reference_score - 1.0).square().mean()
            + (policy_score + 1.0).square().mean()
        )

        # 在归一化后的参考输入上施加零中心梯度惩罚。
        # torch.autograd.grad() 返回一个元组，这里只请求了一个输入的梯度，所以取第一个结果。
        gradient = torch.autograd.grad(
            outputs=reference_score.sum(),
            inputs=reference,
            create_graph=True,
        )[0]

        # 梯度惩罚计算公式
        penalty = (
            gradient.square()
            .sum(dim=-1)
            .mean()
        )

        # 最终损失函数计算公式
        loss = (
            classification_loss
            + 0.5 * self.gradient_penalty * penalty
        )

        if not torch.isfinite(loss).item():
            raise ValueError("判别器损失包含 NaN/Inf")

        self.optimizer.zero_grad(set_to_none=True)  # 清除上一次梯度
        loss.backward()                             # 反向传播

        # 参数梯度裁剪，限制整体参数梯度的大小，避免一次更新过猛。
        nn.utils.clip_grad_norm_(
            self.network.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()                       # 更新参数

        return {
            "loss": loss.item(),
            "classification_loss": classification_loss.item(),
            "gradient_penalty": penalty.item(),
            "reference_score": reference_score.detach().mean().item(),
            "policy_score": policy_score.detach().mean().item(),
        }

class AMPWindowDiscriminator(AMPDiscriminator):
    """使用滑动窗口的 AMP 判别器。"""

    def __init__(
        self,
        reference_states,
        num_steps=5,        # 滑动窗口长度
        learning_rate=1e-4,
        gradient_penalty=10.0,
        weight_decay=0.0,       # 权重衰减，默认 0
    ):
        if num_steps < 2:
            raise ValueError("滑动窗口长度必须至少为 2")

        # 复用单帧参考统计量和原有损失函数。
        super().__init__(
            reference_states=reference_states,
            learning_rate=learning_rate,
            gradient_penalty=gradient_penalty,
        )

        self.num_steps = num_steps
        self.state_dim = reference_states.shape[-1]

        # 窗口长度 × 单帧特征维度。
        self.network[0] = nn.Linear(
            self.num_steps * self.state_dim,
            256,
        ).to(reference_states.device)

        # 网络第一层被替换后，必须重新创建优化器。
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def normalize(self, windows):
        """所有时间步使用相同的单帧统计量，再展开时间维。"""
        if windows.ndim != 3 or windows.shape[1:] != (self.num_steps, self.state_dim):
            raise ValueError(f"判别器输入必须为 [批量, {self.num_steps}, {self.state_dim}]")

        normalized = (
            windows - self.state_mean[None, None, :]
        ) / self.state_std[None, None, :]

        return normalized.flatten(start_dim=1)  # 展开时间维，变为 [批量, num_steps * state_dim]

    @torch.no_grad()
    def style_reward(self, windows):
        """根据完整窗口计算 [0, 1] 风格奖励。"""
        score = self(windows)

        reward = (
            1.0
            - 0.25 * (score - 1.0).square()
        )

        return reward.clamp(0.0, 1.0)