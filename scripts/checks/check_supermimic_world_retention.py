"""在固定的离线留出窗口上比较预训练模型与在线更新模型。"""
import argparse
import json
from pathlib import Path

import torch

from fit_supermimic_smoke import load_modules, valid_windows, sample
from audit_supermimic_training import prediction_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--updated-checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        parser.error('请使用新的输出文件，且 batch-size 必须为正整数')
    torch.set_num_threads(4)
    torch.manual_seed(4321)
    state, dynamics = load_modules()
    # 复用生产状态解析和旋转函数，仅使用临时包名避免启动仿真。
    import sys
    sys.modules['x2_style_mimic.algorithms.supermimic.state'] = state
    import importlib
    sys.modules['x2_style_mimic.utils.rotation'] = importlib.import_module(state.__package__.split('.')[0] + '.utils.rotation')
    base = torch.load(args.base_model, map_location='cpu', weights_only=False)
    updated = torch.load(args.updated_checkpoint, map_location='cpu', weights_only=False)
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    split = json.loads(args.split.read_text())
    if Path(split['data']).resolve() != args.data.resolve():
        raise ValueError('回合划分报告与数据文件不对应')
    if base['metadata'] != data['metadata']:
        raise ValueError('预训练世界模型与离线数据元信息不一致')
    for name in ('state_spec', 'context_spec', 'joint_names', 'control_joint_names'):
        if updated['metadata'][name] != data['metadata'][name]:
            raise ValueError(f'在线模型元信息不一致：{name}')
    if tuple(updated['metadata']['world_hidden_dims']) != tuple(base['hidden_dims']):
        raise ValueError('两个世界模型的网络结构不一致')
    train, heldout = set(map(tuple, split['train_episodes'])), set(map(tuple, split['validation_episodes']))
    if not heldout or train & heldout:
        raise ValueError('离线训练与验收回合划分无效')
    candidates = valid_windows(data, 32)
    keep = torch.tensor([(int(env), int(data['episode_ids'][start, env])) in heldout
                         for start, env in candidates], dtype=torch.bool)
    candidates = candidates[keep]
    if not len(candidates):
        raise ValueError('没有可用的留出窗口')
    batch = sample(data, candidates, args.batch_size, 32, 'cpu')
    spec = state.StateSpec(**base['metadata']['state_spec'])
    errors = {}
    with torch.no_grad():
        for name, weights in [('pretrained', base['model']), ('mixed_online', updated['modules']['world_model'])]:
            limiter = dynamics.TargetLimiter(**base['limiter'])
            model = dynamics.WorldModel(spec, base['context_dim'], limiter, base['hidden_dims'])
            model.load_state_dict(weights, strict=True)
            predicted = dynamics.rollout_logged_actions(model, batch['states'][:, 0], batch['actions'], batch['context'])
            if not bool(torch.isfinite(predicted).all()):
                raise RuntimeError('留出轨迹预测包含 NaN/Inf')
            errors[name] = {str(h): prediction_errors(predicted[:, h], batch['states'][:, h], spec) for h in (1, 8, 32)}
    report = dict(evaluated=True, seed=4321, sampled_windows=args.batch_size,
                  heldout_episodes=len(heldout), valid_windows_32=len(candidates),
                  fixed_same_windows=True, errors=errors,
                  note='固定离线留出轨迹的误差对照；不设置预测质量通过阈值。')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
