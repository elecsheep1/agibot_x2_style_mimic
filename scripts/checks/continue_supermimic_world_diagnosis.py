"""从已有世界模型权重继续训练，固定数据划分和诊断窗口。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from fit_supermimic_smoke import load_modules, valid_windows, sample, sample_training_batch
from supermimic_phase_sampling import candidate_labels, gather_windows
from diagnose_supermimic_rising_rollouts import select_per_episode, anchored_rollout
from audit_supermimic_training import prediction_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=8142)
    args = parser.parse_args()
    if args.output.exists() or args.updates < 1 or args.batch_size < 2:
        parser.error('请指定新的输出目录，更新次数为正，批量至少为2')
    torch.set_num_threads(4)
    state_module, dynamics = load_modules()
    base = torch.load(args.base_model, map_location=args.device, weights_only=False)
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    split = json.loads(args.split.read_text())
    if Path(split['data']).resolve() != args.data.resolve() or base['metadata'] != data['metadata']:
        raise ValueError('世界模型、数据与回合划分不对应')
    if base.get('consistency_weight', 0) != 0:
        raise ValueError('本对照只支持原始真实状态监督损失')
    train_keys, heldout_keys = set(map(tuple, split['train_episodes'])), set(map(tuple, split['validation_episodes']))
    actual = {(env, int(ep)) for env in range(data['episode_ids'].shape[1]) for ep in data['episode_ids'][:, env].unique()}
    if not train_keys or not heldout_keys or train_keys & heldout_keys or train_keys | heldout_keys != actual:
        raise ValueError('训练／验收回合划分无效')
    horizons = tuple(base['training_horizons'])
    early_fraction = base['early_fraction']
    pools = {}
    for horizon in set(horizons) | {32}:
        candidates = valid_windows(data, horizon)
        mask = torch.tensor([(int(e), int(data['episode_ids'][s, e])) in train_keys for s, e in candidates])
        pools[horizon] = dict(train=candidates[mask], validation=candidates[~mask])
        if not len(pools[horizon]['train']) or not len(pools[horizon]['validation']):
            raise ValueError('训练或验收没有足够的连续窗口')
    rising = {}
    support = {}
    selected = {}
    for split_name in ('train', 'validation'):
        candidates = pools[32][split_name]
        candidates = candidates[candidate_labels(data, candidates) == 1]
        chosen, episodes = select_per_episode(data, candidates, maximum=6)
        rising[split_name] = gather_windows(data, chosen, 32, args.device)
        selected[split_name] = chosen.tolist()
        support[split_name] = dict(episodes=episodes, windows=len(chosen))
    torch.manual_seed(8141)
    natural = sample(data, pools[32]['validation'], 128, 32, args.device)
    spec = state_module.StateSpec(**base['metadata']['state_spec'])
    torch.manual_seed(args.seed)
    model = dynamics.WorldModel(spec, base['context_dim'], dynamics.TargetLimiter(**base['limiter']), base['hidden_dims']).to(args.device)
    model.load_state_dict(base['model'], strict=True)
    if not bool(model.stats_ready):
        raise ValueError('预训练模型没有归一化统计量')
    normalization = {k: model.state_dict()[k].clone() for k in ('input_mean', 'input_std', 'stats_ready')}
    for name, value in data['limiter'].items():
        torch.testing.assert_close(model.limiter.state_dict()[name].cpu(), value, atol=0, rtol=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    def errors(prediction, truth):
        return prediction_errors(prediction, truth, spec, state_module=state_module, rotation_error_fn=dynamics.rotation_error)

    @torch.no_grad()
    def evaluate():
        result = {}
        for split_name, batch in rising.items():
            truth, actions, context = batch['states'], batch['actions'], batch['context']
            free = anchored_rollout(model, truth, actions, context)
            teacher = anchored_rollout(model, truth, actions, context, interval=1)
            if not bool(torch.isfinite(free).all()) or not bool(torch.isfinite(teacher).all()):
                raise RuntimeError('诊断预测出现 NaN/Inf')
            result['rising_' + split_name] = dict(free={str(h): errors(free[:, h], truth[:, h]) for h in (1, 8, 16, 32)},
                teacher_forced_all_steps=errors(teacher[:, 1:].reshape(-1, spec.dim), truth[:, 1:].reshape(-1, spec.dim)))
        prediction = dynamics.rollout_logged_actions(model, natural['states'][:, 0], natural['actions'], natural['context'])
        result['natural_validation'] = {str(h): errors(prediction[:, h], natural['states'][:, h]) for h in (1, 8, 16, 32)}
        for key, value in normalization.items():
            torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
        return result

    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(base_model=str(args.base_model.resolve()), base_reported_updates=split.get('updates'),
        seed=args.seed, batch_size=args.batch_size, horizons=horizons, early_fraction=early_fraction,
        support=support, selected_rising_windows=selected, measurements={'0': evaluate()},
        optimizer_state_available='optimizer' in base, optimizer_reinitialized=True,
        input_statistics_frozen=True, actor_updated=False,
        note='已有checkpoint未保存Adam状态，因此这是从权重出发、重新初始化优化器的继续训练对照，不是逐位一致的断点恢复。验收窗口固定，原模型不覆盖。')
    print('[基线] ' + json.dumps(report['measurements']['0']['rising_validation']['free']['32']), flush=True)
    started = time.monotonic()
    checkpoints = {min(100, args.updates), min(500, args.updates), args.updates}
    trained = 0
    for update in range(args.updates):
        horizon = horizons[update % len(horizons)]
        batch = sample_training_batch(data, pools[horizon]['train'], args.batch_size, horizon, args.device, early_fraction)
        optimizer.zero_grad(set_to_none=True)
        prediction = dynamics.rollout_logged_actions(model, batch['states'][:, 0], batch['actions'], batch['context'])
        loss, _ = dynamics.dynamics_loss(prediction, batch['states'], spec)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('训练损失出现 NaN/Inf')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        trained += args.batch_size * horizon
        if (update + 1) % 50 == 0:
            print(f'[继续训练] {update + 1}/{args.updates}，窗口 {horizon}，损失 {float(loss.detach()):.6f}', flush=True)
        if update + 1 in checkpoints:
            report['measurements'][str(update + 1)] = evaluate()
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), metadata=data['metadata'],
                            hidden_dims=tuple(base['hidden_dims']), limiter=base['limiter'], context_dim=base['context_dim'],
                            training_horizons=horizons, early_fraction=early_fraction, consistency_weight=0.,
                            base_checkpoint=str(args.base_model.resolve()), additional_updates=update + 1,
                            optimizer_reinitialized=True, torch_rng=torch.get_rng_state()), args.output / f'world_model_{update + 1}.pt')
            print('[验收] ' + str(update + 1) + ' ' + json.dumps(report['measurements'][str(update + 1)]['rising_validation']['free']['32']), flush=True)
            (args.output / 'report.partial.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    report.update(additional_updates=args.updates, trained_transition_steps=trained, elapsed_seconds=time.monotonic() - started,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.base_model, args.data, args.split)})
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[完成] ' + str(args.output / 'report.json'), flush=True)


if __name__ == '__main__':
    main()
