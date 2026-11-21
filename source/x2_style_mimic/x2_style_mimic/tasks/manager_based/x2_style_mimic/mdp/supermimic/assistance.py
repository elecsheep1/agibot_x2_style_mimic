"""真实与想象环境共享的高度辅助规则。"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class HeightAssistanceCfg:
    full_below_height: float
    zero_above_height: float

    def __post_init__(self):
        if not all(
            math.isfinite(value)
            for value in (
                self.full_below_height,
                self.zero_above_height,
            )
        ):
            raise ValueError("辅助高度阈值必须为有限值")

        if (
            self.full_below_height
            >= self.zero_above_height
        ):
            raise ValueError("完全撤力高度必须大于满辅助高度")


def height_assistance(
    height,
    force_cap,
    withdrawn,
    cfg,
):
    """返回本步辅助力及更新后的撤力标记。

    输入形状均为 [B]。
    高度使用环境局部世界坐标，单位米。
    辅助力沿世界 +Z，单位牛顿。
    """

    if (
        height.shape != force_cap.shape
        or height.shape != withdrawn.shape
    ):
        raise ValueError("高度、辅助上限和撤力标记形状必须一致")
    if withdrawn.dtype != torch.bool:
        raise ValueError("撤力标记必须为布尔型")

    next_withdrawn = (
        withdrawn
        | (height >= cfg.zero_above_height)
    )

    progress = (
        (height - cfg.full_below_height)
        / (
            cfg.zero_above_height
            - cfg.full_below_height
        )
    ).clamp(0.0, 1.0)

    fade = (
        1.0
        - progress.square()
        * (3.0 - 2.0 * progress)
    )

    force = torch.where(
        next_withdrawn,
        torch.zeros_like(height),
        force_cap * fade,
    )

    return force, next_withdrawn