# 改编自 CuboiLeo/SuperMimic-Reproduced，MIT；完整许可见 LICENSE.reproduced。
"""Dynamics MLP f_omega (paper Table A.4: MLP (2048, 2048, 2048)).
    世界模型中的动力学网络,
    它不直接预测完整的下一状态，而是预测一个状态修正量 delta，
    再由 StateSpec.integrate() 将修正量应用到当前状态上。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    参数含义：
    - in_dim：输入维度
    - hidden：隐藏层维度，例如 (2048, 2048, 2048)
    - out_dim：输出维度
    - act：激活函数，默认是 SiLU
    - zero_last：是否将最后一层初始化为全零
    - last_gain：如果不使用全零初始化，是否使用指定增益的正交初始化
    """
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.SiLU, zero_last: bool = False, last_gain: float | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicsMLP(nn.Module):
    """delta = f(canonical(s), a, xi_norm) -> (…, 74). Zero-initialised last layer (identity dynamics at init)."""

    def __init__(self, feat_dim: int = 75, act_dim: int = 29, xi_dim: int = 35, hidden=(2048, 2048, 2048), out_dim: int = 74):
        super().__init__()
        self.mlp = MLP(feat_dim + act_dim + xi_dim, tuple(hidden), out_dim, zero_last=True)

    def forward(self, feat: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, action, xi_norm], dim=-1))
