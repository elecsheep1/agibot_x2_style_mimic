"""复核长度对照的回合隔离、训练预算和实际窗口高度分布。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from fit_supermimic_smoke import valid_windows, sample
from supermimic_offline_world import OfflineWorldReplay
from compare_supermimic_world_horizons import training_schedules, select_env_windows


def height_summary(states):
    heights = states[..., 2]
    return dict(count=heights.numel(), low_fraction=float((heights < .3).float().mean()),
                middle_fraction=float(((heights >= .3) & (heights < .5)).float().mean()),
                high_fraction=float((heights >= .5).float().mean()))


def window_summary(states):
    height = states[..., 2]
    return dict(windows=states.shape[0], initial_height=height_summary(states[:, :1]),
                target_height=height_summary(states[:, 1:]),
                height_range_over_10cm_fraction=float(((height.amax(1) - height.amin(1)) > .1).float().mean()),
                low_to_high_fraction=float(((height[:, 0] < .3) & (height[:, -1] >= .5)).float().mean()),
                high_to_low_fraction=float(((height[:, 0] >= .5) & (height[:, -1] < .3)).float().mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-run', type=Path, required=True)
    parser.add_argument('--offline-data', type=Path, required=True)
    parser.add_argument('--offline-split', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = json.loads((args.comparison_run / 'report.json').read_text())
    output = args.comparison_run / 'data_audit.json'
    if output.exists():
        parser.error('审计输出已存在，请避免覆盖')
    for name, digest in report['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'对照实验输入文件已发生改变：{name}')
    online = torch.load(args.comparison_run / 'online_transitions.pt', map_location='cpu', weights_only=False)
    offline_data = torch.load(args.offline_data, map_location='cpu', weights_only=False)
    offline = OfflineWorldReplay(args.offline_data, args.offline_split,
                                 online['metadata'], offline_data['limiter'], 'cpu')
    train_envs, test_envs = set(report['train_envs']), set(report['holdout_envs'])
    if train_envs & test_envs or train_envs | test_envs != set(range(online['state'].shape[1])):
        raise ValueError('在线训练／验收环境没有形成互斥的完整划分')
    train = select_env_windows(online, sorted(train_envs))
    test = select_env_windows(online, sorted(test_envs))
    all_offline = valid_windows(offline.data, 32)
    keep = torch.tensor([(int(e), int(offline.data['episode_ids'][s, e])) in offline.holdout_keys
                         for s, e in all_offline], dtype=torch.bool)
    old_test = all_offline[keep]
    torch.manual_seed(6002)
    online_validation = sample(online, test, 128, 32, 'cpu')
    offline_validation = sample(offline.data, old_test, 128, 32, 'cpu')
    torch.manual_seed(6003)
    batches = []
    for _ in range(report['cycles']):
        old = offline.sample_windows(report['batch_size'] // 2, 32)
        new = sample(online, train, report['batch_size'] // 2, 32, 'cpu')
        batches.append(torch.cat((old['states'], new['states']), 0))
    results = {}
    for name, schedule in training_schedules().items():
        arm = report['arms'][name]
        expected_updates = report['cycles'] * len(schedule)
        expected_transitions = report['cycles'] * report['batch_size'] * sum(schedule)
        if arm['updates'] != expected_updates or arm['trained_transitions'] != expected_transitions:
            raise ValueError(f'训练预算统计不一致：{name}')
        targets = torch.cat([states[:, 1:h + 1].reshape(-1, states.shape[-1])
                             for states in batches for h in schedule], 0)
        if targets.shape[0] != expected_transitions:
            raise ValueError('重建的实际目标状态数量与预算不一致')
        results[name] = dict(updates=expected_updates, predicted_transitions=expected_transitions,
                             actual_target_heights=height_summary(targets))
    full = torch.cat(batches, 0)
    half = report['batch_size'] // 2
    old = torch.cat([b[:half] for b in batches], 0)
    new = torch.cat([b[half:] for b in batches], 0)
    audit = dict(structural_checks_passed=True, input_hashes_match=True,
                 online_train_holdout_disjoint=True, offline_train_holdout_disjoint=not bool(offline.train_keys & offline.holdout_keys),
                 offline_train_episodes=len(offline.train_keys), offline_holdout_episodes=len(offline.holdout_keys),
                 original_offline=height_summary(offline.data['state']), original_online=height_summary(online['state']),
                 sampled_train_old=window_summary(old), sampled_train_new=window_summary(new),
                 sampled_train_mixed=window_summary(full),
                 validation_old=window_summary(offline_validation['states']), validation_new=window_summary(online_validation['states']),
                 arms=results,
                 limitations=['高度分段只是数据覆盖诊断，不能代替运动阶段、真实接触或起身成功判定。',
                              '实验只覆盖一个短训策略的加噪数据，每组在线验收仅4个环境，窗口可能重叠。',
                              '匹配预测转移数时，固定8步组对同一前缀重复更新7次；混合组更新3次，更新次数和覆盖时刻不同。',
                              '旧数据为反复使用的验证划分，不能当作未参与过模型选择的最终测试集。'])
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
