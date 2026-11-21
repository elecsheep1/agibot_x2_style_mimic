"""检查真实输入下的角速度增量误差及训练集中相似状态的覆盖。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from fit_supermimic_smoke import load_modules, valid_windows
from supermimic_phase_sampling import gather_windows
from diagnose_supermimic_rising_rollouts import anchored_rollout


def nearest_rows(queries, bank, train_keys, exclude_keys, k=16, chunk_size=4096):
    distances = torch.empty(len(queries), 0)
    indexes = torch.empty(len(queries), 0, dtype=torch.long)
    for offset in range(0, len(bank), chunk_size):
        values = bank[offset:offset + chunk_size]
        current = torch.cdist(queries, values) / math.sqrt(bank.shape[1])
        keys = train_keys[offset:offset + len(values)]
        excluded = (exclude_keys[:, None, :] == keys[None, :, :]).all(-1)
        current[excluded] = float('inf')
        ids = torch.arange(offset, offset + len(values))[None].expand(len(queries), -1)
        joined_d = torch.cat((distances, current), 1)
        joined_i = torch.cat((indexes, ids), 1)
        distances, selection = joined_d.topk(min(k, joined_d.shape[1]), largest=False, dim=1)
        indexes = joined_i.gather(1, selection)
    if distances.shape[1] != k or not bool(torch.isfinite(distances).all()):
        raise ValueError('排除自身回合后，训练样本数量不足')
    return distances, indexes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--continuation-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('输出文件已存在，请指定新的文件')
    torch.set_num_threads(4)
    state_module, dynamics = load_modules()
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    split = json.loads(args.split.read_text())
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    run = json.loads((args.continuation_run / 'report.json').read_text())
    if checkpoint['metadata'] != data['metadata'] or Path(split['data']).resolve() != args.data.resolve():
        raise ValueError('模型、数据与回合划分不对应')
    train_set, heldout_set = set(map(tuple, split['train_episodes'])), set(map(tuple, split['validation_episodes']))
    if not train_set or not heldout_set or train_set & heldout_set:
        raise ValueError('训练与验收回合划分无效')
    candidates = valid_windows(data, 1)
    mask = torch.tensor([(int(e), int(data['episode_ids'][s, e])) in train_set for s, e in candidates])
    train = candidates[mask]
    keys = torch.stack((train[:, 1], data['episode_ids'][train[:, 0], train[:, 1]]), -1)
    spec = state_module.StateSpec(**data['metadata']['state_spec'])
    model = dynamics.WorldModel(spec, checkpoint['context_dim'], dynamics.TargetLimiter(**checkpoint['limiter']), checkpoint['hidden_dims'])
    model.load_state_dict(checkpoint['model'], strict=True)
    dynamic_width = model.input_mean.numel() - checkpoint['context_dim']
    with torch.no_grad():
        normalized = []
        for chunk in train.split(2048):
            s, e = chunk.unbind(-1)
            features, _ = model.features(data['state'][s, e], data['action'][s, e], data['contexts'][e])
            normalized.append((features - model.input_mean) / model.input_std)
        bank = torch.cat(normalized)
        worst_start, worst_env, worst_episode = 1080, 4, 1
        if (worst_env, worst_episode) not in heldout_set:
            raise ValueError('异常回合不属于留出集')
        batch = gather_windows(data, torch.tensor([[worst_start, worst_env]]), 32, 'cpu')
        truth, action, context = batch['states'], batch['actions'], batch['context']
        free = anchored_rollout(model, truth, action, context)
        teacher = anchored_rollout(model, truth, action, context, 1)
        checkpoints = [0, 1, 2, 4, 8, 16, 24, 31]
        real_features, _ = model.features(truth[0, checkpoints], action[0, checkpoints], context.expand(len(checkpoints), -1))
        free_features, _ = model.features(free[0, checkpoints], action[0, checkpoints], context.expand(len(checkpoints), -1))
        queries = torch.cat(((real_features - model.input_mean) / model.input_std,
                             (free_features - model.input_mean) / model.input_std), 0)
        exclude = torch.tensor([[worst_env, worst_episode]] * len(queries))
        distances, neighbors = nearest_rows(queries[:, :dynamic_width], bank[:, :dynamic_width], keys, exclude)
        full_distances, full_neighbors = nearest_rows(queries, bank, keys, exclude)
        traces = []
        for number, step in enumerate(checkpoints):
            actual_delta = truth[0, step + 1, 10:13] - truth[0, step, 10:13]
            predicted_delta = teacher[0, step + 1, 10:13] - truth[0, step, 10:13]
            rows = train[neighbors[number]]
            neighbor_delta = data['next_state'][rows[:, 0], rows[:, 1], 10:13] - data['state'][rows[:, 0], rows[:, 1], 10:13]
            context_rows = train[full_neighbors[number]]
            nstate = data['state'][context_rows[:, 0], context_rows[:, 1]]
            naction = data['action'][context_rows[:, 0], context_rows[:, 1]]
            ncontext = data['contexts'][context_rows[:, 1]]
            nfollowing = model(nstate, naction, ncontext)
            nactual_delta = data['next_state'][context_rows[:, 0], context_rows[:, 1], 10:13] - nstate[:, 10:13]
            npred_delta = nfollowing[:, 10:13] - nstate[:, 10:13]
            traces.append(dict(step=step, prediction_time_s=step * data['metadata']['step_dt'],
                actual_angular_delta=actual_delta.tolist(), predicted_angular_delta_true_input=predicted_delta.tolist(),
                delta_dot_product=float((actual_delta * predicted_delta).sum()),
                nearest_dynamic_rms=float(distances[number, 0]), nearest_full_rms=float(full_distances[number, 0]),
                free_input_nearest_dynamic_rms=float(distances[number + len(checkpoints), 0]),
                neighbor_episodes=len(set(map(tuple, keys[neighbors[number]].tolist()))),
                neighbor_birth_motion_ids=data['motion_ids'][rows[:, 0], rows[:, 1]].tolist(),
                neighbor_angular_delta_mean=neighbor_delta.mean(0).tolist(),
                neighbor_angular_delta_std=neighbor_delta.std(0, unbiased=False).tolist(),
                neighbor_rows=rows.tolist(),
                context_aware_neighbors=dict(episodes=len(set(map(tuple, keys[full_neighbors[number]].tolist()))),
                    env_ids=context_rows[:, 1].tolist(),
                    rows=context_rows.tolist(),
                    actual_angular_delta_mean=nactual_delta.mean(0).tolist(),
                    predicted_angular_delta_mean=npred_delta.mean(0).tolist(),
                    delta_rmse_rad_s=float((nactual_delta - npred_delta).square().mean().sqrt()))))
        # 同类训练状态也排除其自身整个回合，作为距离量级的参照。
        s, e = train.unbind(-1)
        q = data['state'][s, e, 3:7]
        up = 1 - 2 * q[:, :2].square().sum(-1)
        calibration_mask = (data['motion_ids'][s, e] == 6) & (data['state'][s, e, 2] >= .3) & (data['state'][s, e, 2] < .5) & (up < .5)
        calibration_pool = calibration_mask.nonzero().flatten()
        if not len(calibration_pool):
            raise ValueError('没有可用的同类训练状态作距离参照')
        generator = torch.Generator().manual_seed(9123)
        calibration_ids = calibration_pool[torch.randperm(len(calibration_pool), generator=generator)[:32]]
        calibration_distance, _ = nearest_rows(bank[calibration_ids, :dynamic_width], bank[:, :dynamic_width], keys, keys[calibration_ids])
        chosen = torch.tensor(run['selected_rising_windows']['validation'])
        heldout_batch = gather_windows(data, chosen, 32, 'cpu')
        h, a, c = heldout_batch['states'], heldout_batch['actions'], heldout_batch['context']
        outputs = anchored_rollout(model, h, a, c, 1)
        actual_delta = h[:, 1:, 10:13] - h[:, :-1, 10:13]
        predicted_delta = outputs[:, 1:, 10:13] - h[:, :-1, 10:13]
        informative = actual_delta.norm(dim=-1) >= .2
        opposite = (actual_delta * predicted_delta).sum(-1) < 0
        worst_true_delta = truth[:, 1:, 10:13] - truth[:, :-1, 10:13]
        worst_pred_delta = teacher[:, 1:, 10:13] - truth[:, :-1, 10:13]
        worst_informative = worst_true_delta.norm(dim=-1) >= .2
        report = dict(training_states=len(train), dynamic_feature_dimensions=dynamic_width,
            calibration_candidates=len(calibration_pool), calibration_queries=len(calibration_ids),
            calibration_cross_episode_nearest_dynamic_rms_median=float(calibration_distance[:, 0].median()),
            calibration_cross_episode_nearest_dynamic_rms_p90=float(calibration_distance[:, 0].quantile(.9)),
            anomalous_window_true_input_trace=traces,
            heldout_angular_delta=dict(windows=len(chosen), informative_steps=int(informative.sum()),
                threshold_rad_s=.2, opposite_direction_fraction=float(opposite[informative].float().mean()),
                delta_rmse_rad_s=float((actual_delta - predicted_delta).square().mean().sqrt()),
                median_predicted_to_actual_delta_norm_ratio=float((predicted_delta.norm(dim=-1)[informative] / actual_delta.norm(dim=-1)[informative]).median()),
                predicted_delta_below_005_fraction=float((predicted_delta.norm(dim=-1)[informative] < .05).float().mean())),
            worst_window=dict(informative_steps=int(worst_informative.sum()),
                opposite_direction_fraction=float(((worst_true_delta * worst_pred_delta).sum(-1)[worst_informative] < 0).float().mean())),
            note='最近邻距离采用模型输入统计归一化后的RMS，动态距离排除物理context；另报含context距离。距离大不等于已证明分布外，邻域目标分散也不能单独证明接触或隐藏状态原因。方向判定只用于角速度变化范数至少0.2rad/s的步骤。')
    report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.data, args.split, args.model)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[完成] ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
