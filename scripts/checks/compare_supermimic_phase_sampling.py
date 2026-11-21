"""固定数据和多步预算，对照均匀采样与按真实窗口高度变化分层采样。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from fit_supermimic_smoke import load_modules, valid_windows
from compare_supermimic_world_horizons import select_env_windows
from supermimic_offline_world import OfflineWorldReplay
from supermimic_phase_sampling import (
    PHASE_NAMES, candidate_labels, choose_indexes, gather_windows, label_counts,
)
from audit_supermimic_training import prediction_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-run', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--offline-data', type=Path, required=True)
    parser.add_argument('--offline-split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists() or args.cycles < 1 or args.batch_size < 2 or args.batch_size % 2:
        parser.error('输出目录必须不存在，循环数为正，批量为正偶数')
    torch.set_num_threads(4)
    state_module, dynamics = load_modules()
    source = json.loads((args.comparison_run / 'report.json').read_text())
    for name, digest in source['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'原实验输入发生改变：{name}')
    base = torch.load(args.base_model, map_location=args.device, weights_only=False)
    online = torch.load(args.comparison_run / 'online_transitions.pt', map_location='cpu', weights_only=False)
    offline = OfflineWorldReplay(args.offline_data, args.offline_split, online['metadata'], base['limiter'], 'cpu')
    if base['context_dim'] != online['contexts'].shape[-1]:
        raise ValueError('世界模型和新数据的物理上下文维度不一致')
    for name in ('state_spec', 'context_spec', 'joint_names', 'control_joint_names', 'step_dt', 'physics_dt', 'decimation'):
        if base['metadata'][name] != online['metadata'][name]:
            raise ValueError(f'世界模型与新数据不兼容：{name}')
    if set(source['train_envs']) & set(source['holdout_envs']):
        raise ValueError('在线训练与验收环境重叠')
    new_train = select_env_windows(online, source['train_envs'])
    new_test = select_env_windows(online, source['holdout_envs'])
    all_old = valid_windows(offline.data, 32)
    keys = [(int(e), int(offline.data['episode_ids'][s, e])) for s, e in all_old]
    old_train = all_old[torch.tensor([key in offline.train_keys for key in keys])]
    old_test = all_old[torch.tensor([key in offline.holdout_keys for key in keys])]
    pools = {'offline_train': (offline.data, old_train), 'online_train': (online, new_train),
             'offline_validation': (offline.data, old_test), 'online_validation': (online, new_test)}
    labels = {name: candidate_labels(data, candidates) for name, (data, candidates) in pools.items()}
    support = {}
    for name, (data, candidates) in pools.items():
        support[name] = dict(counts=label_counts(labels[name]), episodes={})
        for index, phase in enumerate(PHASE_NAMES):
            selected = candidates[labels[name] == index]
            support[name]['episodes'][phase] = len({(int(e), int(data['episode_ids'][s, e])) for s, e in selected})
    torch.manual_seed(7101)
    tests = {}
    for source_name in ('offline', 'online'):
        pool_name = source_name + '_validation'
        data, candidates = pools[pool_name]
        tests[source_name + '/natural'] = gather_windows(data, candidates[torch.randint(len(candidates), (128,))], 32, args.device)
        for index, phase in enumerate(PHASE_NAMES):
            group = candidates[labels[pool_name] == index]
            if len(group):
                selected = group[torch.randint(len(group), (64,))]
                tests[source_name + '/' + phase] = gather_windows(data, selected, 32, args.device)
    spec = state_module.StateSpec(**base['metadata']['state_spec'])

    def make_world():
        model = dynamics.WorldModel(spec, base['context_dim'], dynamics.TargetLimiter(**base['limiter']), base['hidden_dims']).to(args.device)
        model.load_state_dict(base['model'], strict=True)
        return model

    @torch.no_grad()
    def evaluate(model):
        errors = {}
        for name, batch in tests.items():
            predicted = dynamics.rollout_logged_actions(model, batch['states'][:, 0], batch['actions'], batch['context'])
            if not bool(torch.isfinite(predicted).all()):
                raise RuntimeError('分层验收预测出现 NaN/Inf')
            errors[name] = {str(h): prediction_errors(predicted[:, h], batch['states'][:, h], spec,
                           state_module=state_module, rotation_error_fn=dynamics.rotation_error) for h in (1, 8, 32)}
        macro = {}
        for source_name in ('offline', 'online'):
            phase_errors = [errors[source_name + '/' + phase] for phase in PHASE_NAMES if source_name + '/' + phase in errors]
            macro[source_name] = {str(h): {metric: sum(e[str(h)][metric] for e in phase_errors) / len(phase_errors)
                                 for metric in phase_errors[0][str(h)]} for h in (1, 8, 32)}
        return dict(errors=errors, equal_phase_mean_rmse=macro)

    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(source_run=str(args.comparison_run.resolve()), source_support=support,
                  phase_names=PHASE_NAMES, thresholds=dict(low_height=.3, high_height=.5, height_span=.1, endpoint_change=.05),
                  cycles=args.cycles, batch_size=args.batch_size, offline_fraction=.5,
                  schedule=[8, 16, 32], baseline=evaluate(make_world()), arms={},
                  note='标签仅描述窗口高度及变化，不代表成功起身、静止或接触状态。分层组在每个来源的非空类别内均匀采样，缺失类别不伪造；验收窗口固定。宏平均为各非空类别RMSE的算术平均。')
    for name, balance_fraction in [('uniform', 0.0), ('phase_balanced', 1.0), ('uniform_plus_25pct_balanced', 0.25)]:
        torch.manual_seed(7102)
        model = make_world()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        counts = {'offline': torch.zeros(len(PHASE_NAMES), dtype=torch.long),
                  'online': torch.zeros(len(PHASE_NAMES), dtype=torch.long)}
        target_counts = torch.zeros(3, dtype=torch.long)
        started = time.monotonic()
        for cycle in range(args.cycles):
            parts = []
            for source_name in ('offline', 'online'):
                pool_name = source_name + '_train'
                data, candidates = pools[pool_name]
                source_count = args.batch_size // 2
                balanced_count = int(source_count * balance_fraction)
                uniform_count = source_count - balanced_count
                selections = []
                if uniform_count:
                    selections.append(choose_indexes(labels[pool_name], uniform_count, False))
                if balanced_count:
                    selections.append(choose_indexes(labels[pool_name], balanced_count, True))
                indexes = torch.cat(selections)
                counts[source_name] += torch.bincount(labels[pool_name][indexes], minlength=len(PHASE_NAMES))
                parts.append(gather_windows(data, candidates[indexes], 32, args.device))
            full = {key: torch.cat([part[key] for part in parts], 0) for key in ('states', 'actions', 'context')}
            for horizon in (8, 16, 32):
                target = full['states'][:, :horizon + 1]
                height = target[:, 1:, 2].detach().cpu()
                target_counts += torch.tensor([int((height < .3).sum()), int(((height >= .3) & (height < .5)).sum()), int((height >= .5).sum())])
                optimizer.zero_grad(set_to_none=True)
                predicted = dynamics.rollout_logged_actions(model, target[:, 0], full['actions'][:, :horizon], full['context'])
                loss, _ = dynamics.dynamics_loss(predicted, target, spec)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('世界模型训练损失出现 NaN/Inf')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            if (cycle + 1) % 4 == 0:
                print(f'[{name}] 循环 {cycle + 1}/{args.cycles}，损失 {float(loss.detach()):.6f}', flush=True)
        report['arms'][name] = dict(phase_balance_fraction=balance_fraction, updates=args.cycles * 3, predicted_transitions=args.cycles * args.batch_size * 56,
            training_seconds=time.monotonic() - started,
            sampled_windows={source_name: {phase: int(counts[source_name][i]) for i, phase in enumerate(PHASE_NAMES)} for source_name in counts},
            target_height_fractions=(target_counts.float() / target_counts.sum()).tolist(), **evaluate(model))
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), metadata=online['metadata'],
                        hidden_dims=tuple(base['hidden_dims']), limiter=base['limiter'], context_dim=base['context_dim'],
                        phase_balance_fraction=balance_fraction, source_run=str(args.comparison_run.resolve())), args.output / f'{name}.pt')
    report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (args.base_model, args.offline_data, args.offline_split, args.comparison_run / 'online_transitions.pt')}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[完成] ' + str(args.output / 'report.json'), flush=True)


if __name__ == '__main__':
    main()
