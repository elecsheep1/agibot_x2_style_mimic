"""AMP 判别器分数到风格奖励的映射。"""

import torch
import torch.nn.functional as F


def style_reward_from_score(
    score: torch.Tensor,
    mapping: str = "lsgan",
) -> torch.Tensor:
    if mapping == "lsgan":
        return (
            1.0 - 0.25 * (score - 1.0).square()
        ).clamp(0.0, 1.0)

    if mapping == "softplus":
        return F.softplus(score)

    raise ValueError(
        f"不支持的 AMP 奖励映射：{mapping}"
    )