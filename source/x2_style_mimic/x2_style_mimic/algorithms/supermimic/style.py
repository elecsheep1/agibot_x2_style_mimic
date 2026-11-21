import torch
from torch import nn


class DifferentiableStyleReward(nn.Module):
    """可微风格奖励模块,计算想象轨迹的风格奖励。"""

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, windows):
        score = self.discriminator(windows)

        reward = (
            1.0
            - 0.25 * (score - 1.0).square()
        )

        return reward.clamp(0.0, 1.0)

    @torch.no_grad()
    def evaluate(self, windows):
        """用于真实轨迹评分和日志。"""
        return self(windows)    # 执行回调和forward操作

    def update_discriminator(
        self,
        reference_windows,
        policy_windows,
    ):
        """判别器只用真实运动窗口和参考窗口更新。"""
        return self.discriminator.update(
            reference_windows.detach(),
            policy_windows.detach(),
        )