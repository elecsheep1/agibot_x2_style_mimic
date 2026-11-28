"""冻结参数但保留输入梯度；异常时也恢复原始梯度标志。"""
from contextlib import contextmanager


@contextmanager
def frozen_parameters(*modules):
    params = list(dict.fromkeys(p for module in modules for p in module.parameters()))
    flags = [p.requires_grad for p in params]
    try:
        for p in params: p.requires_grad_(False)
        yield
    finally:
        for p,flag in zip(params,flags): p.requires_grad_(flag)
