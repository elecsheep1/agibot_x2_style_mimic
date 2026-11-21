"""只增加旧数据快速角速度变化窗口配额的受控对照。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from fit_supermimic_smoke import load_modules, valid_windows
from supermimic_phase_sampling import gather_windows
from compare_supermimic_world_horizons import select_env_windows
from diagnose_supermimic_rising_rollouts import anchored_rollout
from audit_supermimic_training import prediction_errors


def angular_change_scores(data, candidates, horizon=32):
    start, env = candidates.unbind(-1)
    time = start[:, None] + torch.arange(horizon)
    before = data['state'][time, env[:, None], 10:13]
    after = data['next_state'][time, env[:, None], 10:13]
    # RMS 而不是最大值，减少单帧尖峰完全决定窗口标签的情况。
    return (after - before).square().sum(-1).mean(-1).sqrt()


def fast_indexes(scores, fraction=.1):
    if scores.ndim != 1 or not len(scores) or not bool(torch.isfinite(scores).all()) or not 0 < fraction <= 1:
        raise ValueError('窗口分数必须非空且有限，采样比例必须在(0,1]内')
    count = max(1, math.ceil(len(scores) * fraction))
    return scores.argsort(descending=True)[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--offline-data', type=Path, required=True)
    parser.add_argument('--offline-split', type=Path, required=True)
    parser.add_argument('--online-run', type=Path, required=True)
    parser.add_argument('--continuation-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=9413)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists() or args.updates < 1 or args.batch_size < 8 or args.batch_size % 8:
        parser.error('输出目录必须不存在，更新次数为正，批量须为8的正整数倍')
    torch.set_num_threads(4)
    state_module, dynamics = load_modules()
    base = torch.load(args.base_model, map_location=args.device, weights_only=False)
    old = torch.load(args.offline_data, map_location='cpu', weights_only=False)
    new = torch.load(args.online_run / 'online_transitions.pt', map_location='cpu', weights_only=False)
    split = json.loads(args.offline_split.read_text())
    source = json.loads((args.online_run / 'report.json').read_text())
    previous = json.loads((args.continuation_run / 'report.json').read_text())
    if base['metadata'] != old['metadata'] or Path(split['data']).resolve() != args.offline_data.resolve():
        raise ValueError('预训练模型与旧数据划分不一致')
    for key, value in new['metadata'].items():
        if old['metadata'].get(key) != value:
            raise ValueError(f'新旧数据元信息不兼容：{key}')
    train_keys, test_keys = set(map(tuple, split['train_episodes'])), set(map(tuple, split['validation_episodes']))
    if not train_keys or not test_keys or train_keys & test_keys:
        raise ValueError('旧数据训练与验收回合划分无效')
    if set(source['train_envs']) & set(source['holdout_envs']):
        raise ValueError('新数据训练与验收环境重叠')
    all_old = valid_windows(old, 32)
    keys = [(int(e), int(old['episode_ids'][s, e])) for s, e in all_old]
    old_train = all_old[torch.tensor([key in train_keys for key in keys])]
    old_test = all_old[torch.tensor([key in test_keys for key in keys])]
    new_train = select_env_windows(new, source['train_envs'])
    new_test = select_env_windows(new, source['holdout_envs'])
    score = angular_change_scores(old, old_train)
    fast_ids = fast_indexes(score)
    threshold = float(score[fast_ids].min())
    fast_pool = old_train[fast_ids]
    old_test_fast = old_test[angular_change_scores(old, old_test) >= threshold]
    if not len(old_test_fast):
        raise ValueError('训练集阈值下没有快速变化验收窗口')
    rising_rows = torch.tensor(previous['selected_rising_windows']['validation'])
    if any((int(e), int(old['episode_ids'][s, e])) not in test_keys for s, e in rising_rows):
        raise ValueError('上升验收窗口不属于留出回合')
    generator = torch.Generator().manual_seed(9412)
    tests = {}
    for name, data, candidates in [('old_natural', old, old_test), ('old_fast', old, old_test_fast), ('new_natural', new, new_test)]:
        chosen = candidates[torch.randint(len(candidates), (128,), generator=generator)]
        tests[name] = gather_windows(data, chosen, 32, args.device)
    tests['old_rising'] = gather_windows(old, rising_rows, 32, args.device)
    worst = gather_windows(old, torch.tensor([[1080, 4]]), 32, args.device)
    spec = state_module.StateSpec(**base['metadata']['state_spec'])
    half, fast_count = args.batch_size // 2, args.batch_size // 8
    # 两组共享在线窗口，以及旧数据中的前三个四分之一；仅替换旧批次最后四分之一。
    generator = torch.Generator().manual_seed(args.seed)
    fast_generator = torch.Generator().manual_seed(args.seed + 10000)
    draws = [(torch.randint(len(old_train), (half,), generator=generator),
              torch.randint(len(new_train), (half,), generator=generator),
              torch.randint(len(fast_pool), (fast_count,), generator=fast_generator)) for _ in range(args.updates)]

    def make_world():
        model = dynamics.WorldModel(spec, base['context_dim'], dynamics.TargetLimiter(**base['limiter']), base['hidden_dims']).to(args.device)
        model.load_state_dict(base['model'], strict=True)
        return model

    @torch.no_grad()
    def evaluate(model):
        result = {}
        for name, batch in tests.items():
            truth, actions, context = batch['states'], batch['actions'], batch['context']
            predicted = dynamics.rollout_logged_actions(model, truth[:, 0], actions, context)
            if not bool(torch.isfinite(predicted).all()):
                raise RuntimeError('验收预测出现 NaN/Inf')
            result[name] = {str(h): prediction_errors(predicted[:, h], truth[:, h], spec,
                  state_module=state_module, rotation_error_fn=dynamics.rotation_error) for h in (1, 8, 32)}
            if name == 'old_rising':
                teacher = anchored_rollout(model, truth, actions, context, 1)
                actual_delta = truth[:, 1:, 10:13] - truth[:, :-1, 10:13]
                predicted_delta = teacher[:, 1:, 10:13] - truth[:, :-1, 10:13]
                informative = actual_delta.norm(dim=-1) >= .2
                result['angular_direction'] = dict(informative_steps=int(informative.sum()),
                    opposite_fraction=float(((actual_delta * predicted_delta).sum(-1)[informative] < 0).float().mean()),
                    delta_rmse_rad_s=float((actual_delta - predicted_delta).square().mean().sqrt()))
        pred = dynamics.rollout_logged_actions(model, worst['states'][:, 0], worst['actions'], worst['context'])
        result['monitored_worst_window_rotation_deg'] = float(dynamics.rotation_error(pred[:, -1, 3:7], worst['states'][:, -1, 3:7])[0]) * 180 / math.pi
        return result

    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(seed=args.seed, updates=args.updates, batch_size=args.batch_size,
        training_fast_threshold_rms_delta_rad_s=threshold, old_train_windows=len(old_train), fast_train_windows=len(fast_pool),
        old_fast_validation_windows=len(old_test_fast), schedule=[8, 16, 32], optimizer_reinitialized=True,
        fixed_validation_windows=True, shared_online_training_draws=True, baseline=evaluate(make_world()), arms={},
        note='阈值只取旧训练窗口的角速度变化RMS排名前10%。实验组固定12.5%整批快速配额、37.5%旧数据均匀、50%新数据均匀；均匀部分仍可能抽到快速窗口。反复使用的留出数据仅作为验证，不是最终盲测。')
    checkpoints = {min(100, args.updates), args.updates}
    for name, use_fast in [('uniform', False), ('fast_angular_quota', True)]:
        model = make_world()
        normalization = {k: model.state_dict()[k].clone() for k in ('input_mean', 'input_std', 'stats_ready')}
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        arm = dict(measurements={}, quota_fast_fraction=fast_count / args.batch_size if use_fast else 0., trained_transitions=0)
        actual_fast, offline_count = 0, 0
        for update, (old_ids, new_ids, fast_draw) in enumerate(draws):
            selected = old_train[old_ids]
            if use_fast:
                selected = torch.cat((selected[:-fast_count], fast_pool[fast_draw]), 0)
            actual_fast += int((angular_change_scores(old, selected) >= threshold).sum())
            offline_count += len(selected)
            horizon = (8, 16, 32)[update % 3]
            a = gather_windows(old, selected, horizon, args.device)
            b = gather_windows(new, new_train[new_ids], horizon, args.device)
            batch = {k: torch.cat((a[k], b[k]), 0) for k in ('states', 'actions', 'context')}
            optimizer.zero_grad(set_to_none=True)
            prediction = dynamics.rollout_logged_actions(model, batch['states'][:, 0], batch['actions'], batch['context'])
            loss, _ = dynamics.dynamics_loss(prediction, batch['states'], spec)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('训练损失出现 NaN/Inf')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            arm['trained_transitions'] += args.batch_size * horizon
            if (update + 1) % 100 == 0:
                print(f'[{name}] {update + 1}/{args.updates}，损失 {float(loss.detach()):.6f}', flush=True)
            if update + 1 in checkpoints:
                arm['measurements'][str(update + 1)] = evaluate(model)
        for k, value in normalization.items():
            torch.testing.assert_close(model.state_dict()[k], value, atol=0, rtol=0)
        arm.update(actual_fast_fraction_within_offline=actual_fast / offline_count, optimizer_updates=args.updates)
        report['arms'][name] = arm
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), metadata=old['metadata'],
            hidden_dims=tuple(base['hidden_dims']), limiter=base['limiter'], context_dim=base['context_dim'],
            training_horizons=[8, 16, 32], fast_angular_quota=use_fast), args.output / f'{name}.pt')
        (args.output / 'report.partial.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (args.base_model, args.offline_data, args.offline_split, args.online_run / 'online_transitions.pt')}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[完成] ' + str(args.output / 'report.json'), flush=True)


if __name__ == '__main__':
    main()
