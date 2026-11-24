"""按官方参数分组构建 Adam 与 Muon。

当前采用普通 eager 更新，暂不接入 CUDA Graph。
"""

import math

import torch
from torch import nn

def build_optimizers(
    module,
    lr,
    betas=(0.9, 0.999),
):
    """Sequential 中间 Linear 的权重用 Muon。

    首尾 Linear、所有偏置、LayerNorm 参数等用 Adam。
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("优化器学习率必须为有限正数")
    if (
        len(betas) != 2
        or any(
            not math.isfinite(value)
            or not 0 <= value < 1
            for value in betas
        )
    ):
        raise ValueError("Adam betas 必须包含两个位于 [0, 1) 的有限数")

    parameters = list(
        module.parameters()
    )

    if not parameters:
        raise ValueError("网络没有可优化参数")

    hidden = []         # 存放准备交给 Muon 的权重。
    seen = set()        # 记录已收集参数的身份编号，避免重复加入。

    # 遍历网络本身及其所有子模块，只处理 nn.Sequential
    for block in module.modules():
        if not isinstance(
            block,
            nn.Sequential,
        ):
            continue

        # 筛出这个 Sequential 直接包含的 Linear 层
        linears = [
            layer
            for layer in block
            if isinstance(layer, nn.Linear)
        ]

        # 排除该 Sequential 的首尾 Linear。
        for layer in linears[1:-1]:
            parameter_id = id(layer.weight)
            if parameter_id not in seen:
                hidden.append(layer.weight)
                seen.add(parameter_id)

    # 剩下的参数全部交给 Adam
    rest = [
        parameter
        for parameter in parameters
        if id(parameter) not in seen
    ]

    optimizers = [
        torch.optim.Adam(
            rest,
            lr=lr,
            betas=betas,
        )
    ]

    if hidden:
        if not hasattr(
            torch.optim,
            "Muon",
        ):
            raise RuntimeError("当前 PyTorch 不支持 Muon，不能使用这一官方优化器配置")

        optimizers.append(
            torch.optim.Muon(
                hidden,
                lr=lr,
                adjust_lr_fn="match_rms_adamw",
            )
        )

    return optimizers