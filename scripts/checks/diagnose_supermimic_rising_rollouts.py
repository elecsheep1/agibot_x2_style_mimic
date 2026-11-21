"""在真实上升窗口上区分局部预测误差与自回归累积误差。"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch

from fit_supermimic_smoke import load_modules, valid_windows
from supermimic_phase_sampling import candidate_labels, gather_windows
from audit_supermimic_training import prediction_errors


def select_per_episode(data, candidates, maximum=6):
    if maximum < 1:
        raise ValueError('每回合窗口数必须为正')
    groups = defaultdict(list)
    for row, (start, env) in enumerate(candidates.tolist()):
        groups[(env, int(data['episode_ids'][start, env]))].append(row)
    selected = []
    for key in sorted(groups):
        rows = groups[key]
        indexes = torch.linspace(0, len(rows) - 1, min(maximum, len(rows))).long().tolist()
        selected.extend(rows[i] for i in indexes)
    if not selected:
        raise ValueError('没有可用的上升窗口')
    return candidates[torch.tensor(selected)], len(groups)


def anchored_rollout(model, states, actions, context, interval=None):
    """interval=1 每步使用真实输入；None 为完全自回归。"""
    if interval is not None and interval < 1:
        raise ValueError('重新锚定间隔必须为正')
    current = states[:, 0]
    predicted = [current]
    for step in range(actions.shape[1]):
        if interval is not None and step % interval == 0:
            current = states[:, step]
        current = model(current, actions[:, step], context)
        predicted.append(current)
    return torch.stack(predicted, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--models', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows-per-episode', type=int, default=6)
    args = parser.parse_args()
    if args.output.exists() or args.windows_per_episode < 1:
        parser.error('请指定新的输出文件，每回合窗口数必须为正')
    torch.set_num_threads(4)
    state_module, dynamics = load_modules()
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    split = json.loads(args.split.read_text())
    if Path(split['data']).resolve() != args.data.resolve():
        raise ValueError('数据文件与回合划分不一致')
    train_keys, test_keys = set(map(tuple, split['train_episodes'])), set(map(tuple, split['validation_episodes']))
    if not train_keys or not test_keys or train_keys & test_keys:
        raise ValueError('训练与验收回合必须非空且互斥')
    all_windows = valid_windows(data, 32)
    rising = all_windows[candidate_labels(data, all_windows) == 1]
    keys = [(int(env), int(data['episode_ids'][start, env])) for start, env in rising]
    batches, selected = {}, {}
    support = {}
    for name, allowed in [('train', train_keys), ('validation', test_keys)]:
        pool = rising[torch.tensor([key in allowed for key in keys])]
        windows, episodes = select_per_episode(data, pool, args.windows_per_episode)
        batches[name] = gather_windows(data, windows, 32, 'cpu')
        selected[name] = windows
        support[name] = dict(available_windows=len(pool), sampled_windows=len(windows), episodes=episodes)
    spec = state_module.StateSpec(**data['metadata']['state_spec'])
    joint_names = data['metadata']['joint_names']
    report = dict(support=support, selected_windows={k: v.tolist() for k, v in selected.items()}, models={},
                  note='上升仅表示窗口根高度净增至少5cm且变化范围至少10cm。逐步真实输入与周期性锚定是误差诊断，不是实际部署方式。窗口可能重叠，未据此判定起身成功或接触阶段原因。')
    with torch.no_grad():
        for model_path in args.models:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if 'model' not in checkpoint:
                raise ValueError('请提供单独的世界模型 checkpoint')
            for key in ('state_spec', 'context_spec', 'joint_names', 'control_joint_names'):
                if checkpoint['metadata'][key] != data['metadata'][key]:
                    raise ValueError(f'世界模型元信息不兼容：{key}')
            for key, value in data['limiter'].items():
                torch.testing.assert_close(checkpoint['limiter'][key], value, atol=1e-6, rtol=1e-6)
            model = dynamics.WorldModel(spec, data['contexts'].shape[-1],
                        dynamics.TargetLimiter(**checkpoint['limiter']), checkpoint['hidden_dims'])
            model.load_state_dict(checkpoint['model'], strict=True)
            initial = {name: value.clone() for name, value in model.state_dict().items()}
            model_report = {}
            for split_name, batch in batches.items():
                truth, actions, context = batch['states'], batch['actions'], batch['context']
                modes = {}
                for mode, interval in [('teacher_forced', 1), ('anchor_every_8', 8), ('free_rollout', None)]:
                    prediction = anchored_rollout(model, truth, actions, context, interval)
                    if not bool(torch.isfinite(prediction).all()):
                        raise RuntimeError('预测出现 NaN/Inf')
                    step_errors = [prediction_errors(prediction[:, step], truth[:, step], spec,
                                      state_module=state_module, rotation_error_fn=dynamics.rotation_error) for step in range(1, 33)]
                    p = state_module.unpack_state(prediction[:, 1:].reshape(-1, spec.dim), spec)
                    t = state_module.unpack_state(truth[:, 1:].reshape(-1, spec.dim), spec)
                    position = (p['joint_position'] - t['joint_position']).square().mean(0).sqrt()
                    velocity = (p['joint_velocity'] - t['joint_velocity']).square().mean(0).sqrt()
                    ordering = velocity.argsort(descending=True)
                    top_joints = [dict(name=joint_names[i], position_rmse_rad=float(position[i]),
                                       velocity_rmse_rad_s=float(velocity[i])) for i in ordering[:8].tolist()]
                    modes[mode] = dict(step_errors=step_errors,
                                      all_steps=prediction_errors(prediction[:, 1:].reshape(-1, spec.dim), truth[:, 1:].reshape(-1, spec.dim), spec,
                                          state_module=state_module, rotation_error_fn=dynamics.rotation_error),
                                      top_velocity_error_joints=top_joints,
                                      previous_target_max_error=float((p['previous_target'] - t['previous_target']).abs().max()))
                features, _ = model.features(truth[:, :-1].reshape(-1, spec.dim), actions.reshape(-1, spec.num_actions),
                                               context[:, None].expand(-1, 32, -1).reshape(-1, context.shape[-1]))
                normalized = ((features - model.input_mean) / model.input_std).abs()
                model_report[split_name] = dict(modes=modes,
                    input_abs_z_p95=float(normalized.flatten().quantile(.95)),
                    input_abs_z_p99=float(normalized.flatten().quantile(.99)),
                    input_abs_z_over_10_fraction=float((normalized > 10).float().mean()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
            report['models'][str(model_path)] = model_report
    report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.data, args.split, *args.models]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[完成] ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
