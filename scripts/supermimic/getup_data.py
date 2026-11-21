"""训练入口使用的数据采样与检查点校验，不依赖 scripts/checks。"""
import json
import math
from pathlib import Path

import torch


def shared_vector(value, count, name, device):
    value = torch.as_tensor(value, device=device, dtype=torch.float32)
    if value.ndim == 0:
        return value.expand(count).clone()
    if value.ndim == 2:
        if not torch.allclose(value, value[:1].expand_as(value), atol=1e-7, rtol=0):
            raise RuntimeError(f"{name} 在不同环境间不一致，当前 TargetLimiter 不支持")
        value = value[0]
    if value.shape != (count,):
        raise RuntimeError(f"{name} 形状不正确：{value.shape}")
    return value.clone()


def read_world_checkpoint(path, expected_metadata, limiter_values, context_dim, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {'model', 'metadata', 'hidden_dims', 'limiter', 'context_dim'}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError('需要监督训练生成的 world_model.pt 检查点')
    metadata = checkpoint['metadata']
    for name, expected in expected_metadata.items():
        actual = metadata.get(name)
        if name in ('step_dt', 'physics_dt', 'decimation'):
            compatible = actual is not None and math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-9)
        else:
            compatible = actual == expected
        if not compatible:
            raise ValueError(f'世界模型检查点不兼容：{name}')
    if checkpoint['context_dim'] != context_dim:
        raise ValueError('世界模型检查点的 context_dim 不兼容')
    dimensions = checkpoint['hidden_dims']
    if not dimensions or any(not isinstance(v, int) or v < 1 for v in dimensions):
        raise ValueError('世界模型隐藏层维度不合法')
    for name, current in limiter_values.items():
        saved = torch.as_tensor(checkpoint['limiter'][name], device=current.device, dtype=current.dtype)
        torch.testing.assert_close(saved, current, atol=1e-6, rtol=1e-6, msg=f'目标限幅参数不匹配：{name}')
        # 保存的标定参数也必须与模型实际 buffer 一致。
        torch.testing.assert_close(checkpoint['model'][f'limiter.{name}'], current,
                                   atol=1e-6, rtol=1e-6, msg=f'模型状态中的目标限幅参数不匹配：{name}')
    if not bool(checkpoint['model'].get('stats_ready', False)):
        raise ValueError('检查点缺少已拟合的输入统计量')
    for name, value in checkpoint['model'].items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f'世界模型检查点张量包含 NaN/Inf：{name}')
    return checkpoint


def valid_windows(data, horizon):
    """返回 [开始步, 环境编号]，允许最后一条转移终止，但不跨回合。"""
    steps, num_envs = data["terminated"].shape
    if steps < horizon:
        raise ValueError("数据长度小于要求的窗口长度")
    count = steps - horizon + 1
    mask = torch.ones(count, num_envs, dtype=torch.bool)
    episode = data["episode_ids"][:count]
    first_step = data["step_ids"][:count]
    for offset in range(horizon):
        mask &= data["episode_ids"][offset:offset + count] == episode
        mask &= data["step_ids"][offset:offset + count] == first_step + offset
        if offset < horizon - 1:
            mask &= ~(data["terminated"][offset:offset + count] | data["truncated"][offset:offset + count])
    return mask.nonzero()

def sample(data, candidates, batch_size, horizon, device):
    chosen = candidates[torch.randint(len(candidates), (batch_size,))]
    start, env = chosen.unbind(-1)
    time = start[:, None] + torch.arange(horizon)
    envs = env[:, None].expand_as(time)
    states = torch.cat((data["state"][start, env, None], data["next_state"][time, envs]), dim=1)
    return {"states": states.to(device), "actions": data["action"][time, envs].to(device),
            "context": data["contexts"][env].to(device)}


class OfflineWorldReplay:
    def __init__(self, data_path, report_path, expected_metadata, limiter_values, device):
        self.device = device
        self.data = torch.load(data_path, map_location='cpu', weights_only=False)
        report = json.loads(Path(report_path).read_text())
        if Path(report['data']).resolve() != Path(data_path).resolve():
            raise ValueError('离线回合划分报告指向其他数据集')
        metadata = self.data['metadata']
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(f'离线数据集不兼容：{name}')
        for name, current in limiter_values.items():
            torch.testing.assert_close(self.data['limiter'][name].cpu(), current.detach().cpu(), atol=1e-6, rtol=1e-6)
        self.train_keys = set(map(tuple, report['train_episodes']))
        self.holdout_keys = set(map(tuple, report['validation_episodes']))
        actual = {(env, int(episode)) for env in range(self.data['episode_ids'].shape[1])
                  for episode in self.data['episode_ids'][:, env].unique()}
        if not self.train_keys or not self.holdout_keys or self.train_keys & self.holdout_keys or self.train_keys | self.holdout_keys != actual:
            raise ValueError('训练和验证回合必须互斥，且完整覆盖数据集')
        for name in ('state', 'next_state', 'action', 'contexts'):
            if not bool(torch.isfinite(self.data[name]).all()):
                raise ValueError(f'离线数据 {name} 包含 NaN/Inf')
        self.candidates = {}

    def sample_windows(self, batch_size, horizon):
        if batch_size < 1:
            raise ValueError('batch_size 必须为正整数')
        if horizon not in self.candidates:
            candidates = valid_windows(self.data, horizon)
            keep = torch.tensor([(int(env), int(self.data['episode_ids'][start, env])) in self.train_keys
                                 for start, env in candidates], dtype=torch.bool)
            self.candidates[horizon] = candidates[keep]
            if not len(self.candidates[horizon]):
                raise ValueError(f'找不到长度为 {horizon} 的训练回合窗口')
        return sample(self.data, self.candidates[horizon], batch_size, horizon, self.device)
