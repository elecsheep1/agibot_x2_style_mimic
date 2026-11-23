"""离线筛查自缠候选片段；仅供回放核对，不生成正式失败标签或终止回合。"""
import argparse
import json
from pathlib import Path

import torch


def find_candidates(data, *, dt, window_seconds, contact_seconds,
                    height_progress, upright_progress, standing_height, standing_up):
    if dt <= 0 or window_seconds <= 0 or contact_seconds <= 0:
        raise ValueError('时间参数必须为正')
    if min(height_progress, upright_progress) < 0:
        raise ValueError('进展阈值不能为负')
    states = data['history'][:, -1]
    keys = data['episode_keys']
    steps = data['task_memory']['episode_steps']
    # m_t 对齐 history 最后一帧；不使用会重置的下一回合状态。
    active = data['task_memory']['self_contact_strong_active'].bool()
    if not torch.isfinite(states).all():
        raise ValueError('状态含非有限数')
    q = states[:, 3:7]
    norm = q.norm(dim=-1)
    if (norm < 1e-6).any():
        raise ValueError('根四元数无效')
    q = q / norm[:, None]
    up = 1 - 2 * (q[:, 0].square() + q[:, 1].square())
    height = states[:, 2]
    window = max(1, round(window_seconds / dt))
    contact = max(1, round(contact_seconds / dt))
    if contact > window:
        raise ValueError('接触观察长度不能超过进展观察长度')
    records = []
    for key in torch.unique(keys, dim=0):
        ids = torch.where((keys == key).all(-1))[0]
        ids = ids[steps[ids].argsort()]
        run = torch.zeros(active.shape[1], dtype=torch.long)
        start = None
        for j, idx in enumerate(ids.tolist()):
            if j and int(steps[idx] - steps[ids[j - 1]]) != 1:
                run.zero_()
                start = j
            if start is None:
                start = j
            run = torch.where(active[idx], run + 1, 0)
            if j - start < window:
                continue
            recent = ids[j - window:j + 1]
            # 窗口内曾有明显向上进展，先放过，避免误罚正在起身。
            dh = float(height[recent].max() - height[recent[0]])
            du = float(up[recent].max() - up[recent[0]])
            standing = bool((height[recent] >= standing_height).logical_and(
                up[recent] >= standing_up).any())
            pairs = torch.where(run >= contact)[0]
            if len(pairs) and not standing and dh < height_progress and du < upright_progress:
                records.append(dict(episode=key.tolist(), step=int(steps[idx]),
                    time_seconds=float(steps[idx]) * dt, pair_indices=pairs.tolist(),
                    height_m=float(height[idx]), root_up=float(up[idx]),
                    height_progress_m=dh, upright_progress=du))
    # 连续候选帧合并，保留起止及涉及配对，方便回放而非刷屏。
    segments = []
    for row in records:
        if (segments and segments[-1]['episode'] == row['episode']
                and segments[-1]['end_step'] + 1 == row['step']):
            seg = segments[-1]
            seg['end_step'] = row['step']
            seg['end_seconds'] = row['time_seconds']
            seg['pair_indices'] = sorted(set(seg['pair_indices']) | set(row['pair_indices']))
        else:
            segments.append(dict(episode=row['episode'], start_step=row['step'],
                end_step=row['step'], start_seconds=row['time_seconds'],
                end_seconds=row['time_seconds'], pair_indices=row['pair_indices'],
                first_observation=row))
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dt', type=float, required=True, help='采样控制步长，秒')
    parser.add_argument('--window-seconds', type=float, default=2.)
    parser.add_argument('--contact-seconds', type=float, default=.5)
    parser.add_argument('--height-progress', type=float, default=.03)
    parser.add_argument('--upright-progress', type=float, default=.1)
    parser.add_argument('--standing-height', type=float, default=.55)
    parser.add_argument('--standing-up', type=float, default=.85)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('输出文件已存在')
    torch.set_num_threads(4)
    saved = torch.load(args.dataset, weights_only=False, map_location='cpu')
    data = saved['dataset']
    segments = find_candidates(data, **{k: v for k, v in vars(args).items()
                                       if k not in ('dataset', 'output')})
    names = data.get('self_contact_body_names')
    if names:
        pairs = torch.triu_indices(len(names), len(names), offset=1).T.tolist()
        for seg in segments:
            seg['pairs'] = [[names[pairs[i][0]], names[pairs[i][1]]]
                            for i in seg['pair_indices']]
    report = dict(settings={k: str(v) if isinstance(v, Path) else v
                            for k, v in vars(args).items()}, candidate_segments=segments,
        candidate_episode_count=len({tuple(s['episode']) for s in segments}),
        note='候选筛查，未经视频人工确认；不能视为自缠真值。root z 假定数据已减环境原点。强接触沿用传感器滞回状态，不代表持续力严格超过阈值。')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(f'[候选审查完成] 片段={len(segments)}，回合={report["candidate_episode_count"]}，报告={args.output}')


if __name__ == '__main__':
    main()
