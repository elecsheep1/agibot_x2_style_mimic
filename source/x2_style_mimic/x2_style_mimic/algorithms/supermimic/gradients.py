"""冻结参数，保留输入梯度"""

from contextlib import contextmanager


@contextmanager
def frozen_parameters(*modules):
    """冻结模块参数，同时保留输出对输入的梯度。"""
    parameters = []     # 保存所有需要冻结的参数。
    seen = set()        # 记录已经处理过的参数 ID

    for module in modules:
        for parameter in module.parameters():   # 遍历所有模块参数
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)

    original_flags = [              # 保存原始梯度开关
        parameter.requires_grad
        for parameter in parameters
    ]

    try:    # 开始执行临时冻结逻辑
        for parameter in parameters:
            parameter.requires_grad_(False)         # 将所有参数设置为不需要梯度：
        yield                                       # 这表示把控制权交给 with 代码块。
    finally:
        for parameter, enabled in zip(parameters, original_flags):
            parameter.requires_grad_(enabled)       # 恢复
