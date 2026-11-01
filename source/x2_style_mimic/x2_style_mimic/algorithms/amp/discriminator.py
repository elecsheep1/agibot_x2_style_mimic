
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
        reference_states,               # 整段参考动作提取出来的 AMP 状态
        learning_rate=1e-4,             # 判别器的学习率，默认 0.0001
        gradient_penalty=10.0,          # 梯度惩罚的强度，默认 10
        trunk_weight_decay=1.0e-3,      # 主干网络权重衰减，默认 0.001
        head_weight_decay=1.0e-1,       # 输出网络权重衰减，默认 0.1
        normalization_clip=10.0,        # 归一化统计量裁剪，避免数值过大，默认 10
    ):
        super().__init__()

        if (reference_states.ndim != 2 or reference_states.shape[1] != AMP_STATE_DIM):
            raise ValueError(f"参考状态必须为 [帧数, {AMP_STATE_DIM}]")

        if (len(reference_states) < 2 or not torch.isfinite(reference_states).all().item()):
            raise ValueError("参考状态数量或数值错误")

        states = reference_states.detach().float()

        self.normalization_clip = float(normalization_clip)

        self.register_buffer(
            "state_mean",
            states.mean(dim=0).clone(),
        )
        self.register_buffer(
            "state_var",
            states.var(dim=0,unbiased=False,).clone(),
        )
        self.register_buffer(
            "state_count",
            torch.tensor(
                float(states.shape[0]),
                dtype=torch.float64,
                device=states.device,
            )
        )
        self.register_buffer(
            "state_std",
            self.state_var.sqrt().clamp_min(0.05).clone(),
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
        self.configure_optimizer(
            learning_rate,
            trunk_weight_decay,
            head_weight_decay,
        )

    def configure_optimizer(
        self,
        learning_rate,
        trunk_weight_decay,
        head_weight_decay,
    ):
        """重新创建优化器，支持不同的学习率和权重衰减。"""
        self.optimizer = torch.optim.Adam(
            [
                {
                    "name": "amp_trunk",
                    "params": list(self.network[:-1].parameters()),
                    "weight_decay": float(trunk_weight_decay),
                },
                {
                    "name": "amp_head",
                    "params": list(self.network[-1].parameters()),
                    "weight_decay": float(head_weight_decay),
                },
            ],
            lr=float(learning_rate),
        )

    @torch.no_grad()
    def update_normalization(self, reference, policy):
        """参考和策略共同更新同一个统计器；支持窗口或帧对。"""
        state_dim = self.state_mean.numel()
        samples = torch.cat(
            [
                reference.detach().reshape(-1, state_dim),
                policy.detach().reshape(-1, state_dim),
            ],
            dim=0,
        ).to(device=self.state_mean.device, dtype=self.state_mean.dtype)

        if samples.shape[0] == 0:
            return
        if not torch.isfinite(samples).all():
            raise ValueError("AMP 标准化样本包含 NaN/Inf")
        batch_mean = samples.mean(dim=0)
        batch_var = samples.var(dim=0, unbiased=False)
        batch_count = samples.shape[0]

        total_count = self.state_count + batch_count    # 累计样本数量
        delta = batch_mean - self.state_mean
        new_mean = self.state_mean + delta * (batch_count / total_count).to(self.state_mean.dtype)
        new_var = (
            self.state_var * self.state_count.to(self.state_var.dtype)
            + batch_var * batch_count
            + delta.square()
            * (self.state_count * batch_count / total_count).to(self.state_var.dtype)
        ) / total_count.to(self.state_var.dtype)
        self.state_mean.copy_(new_mean)
        self.state_var.copy_(new_var.clamp_min(0.0))
        self.state_std.copy_(self.state_var.sqrt().clamp_min(0.05))
        self.state_count.copy_(total_count)

    def normalize_states(self, states):
        normailzed = (states - self.state_mean) / self.state_std
        return normailzed.clamp(
            -self.normalization_clip,
            self.normalization_clip,
        )


    def normalize(self, pairs):
        """相邻两帧分别使用同一套单帧统计量进行归一化。"""
        expected_dim = 2 * AMP_STATE_DIM
        if pairs.ndim != 2 or pairs.shape[1] != expected_dim:
            raise ValueError(f"判别器输入必须为 [批量, {expected_dim}]")

        states = pairs.reshape(-1, 2, self.state_mean.numel())
        return self.normalize_states(states).flatten(start_dim=1)

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
        self.update_normalization(reference_pairs, policy_pairs)

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
        trunk_weight_decay=1.0e-3,
        head_weight_decay=1.0e-1,
        normalization_clip=10.0,
    ):
        if num_steps < 2:
            raise ValueError("滑动窗口长度必须至少为 2")

        # 复用单帧参考统计量和原有损失函数。
        super().__init__(
            reference_states=reference_states,
            learning_rate=learning_rate,
            gradient_penalty=gradient_penalty,
            trunk_weight_decay=trunk_weight_decay,
            head_weight_decay=head_weight_decay,
            normalization_clip=normalization_clip,
        )

        self.num_steps = num_steps
        self.state_dim = reference_states.shape[-1]

        # 窗口长度 × 单帧特征维度。
        self.network[0] = nn.Linear(
            self.num_steps * self.state_dim,
            256,
        ).to(reference_states.device)

        # 网络第一层被替换后，必须重新创建优化器。
        self.configure_optimizer(
            learning_rate,
            trunk_weight_decay,
            head_weight_decay,
        )

    def normalize(self, windows):
        """所有时间步使用相同的单帧统计量，再展开时间维。"""
        if windows.ndim != 3 or windows.shape[1:] != (self.num_steps, self.state_dim):
            raise ValueError(f"判别器输入必须为 [批量, {self.num_steps}, {self.state_dim}]")

        normalized = self.normalize_states(windows)
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