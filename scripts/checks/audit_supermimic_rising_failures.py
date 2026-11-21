"""分解留出上升窗口的动作／回合误差，并检查位姿与速度一致性。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from fit_supermimic_smoke import load_modules, consistency_residuals
from diagnose_supermimic_rising_rollouts import anchored_rollout
from supermimic_phase_sampling import gather_windows


def distribution(value):
    value = value.flatten()
    return dict(mean=float(value.mean()), median=float(value.median()), p90=float(value.quantile(.9)),
                p99=float(value.quantile(.99)), maximum=float(value.max()))


def local_up(quaternion):
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)
    return torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x),
                        1 - 2 * (x.square() + y.square())), -1)


def gravity_angle(predicted, truth):
    a, b = local_up(predicted), local_up(truth)
    return torch.atan2(torch.cross(a, b, dim=-1).norm(dim=-1), (a * b).sum(-1)) * 180 / math.pi


def residual_report(before, after, spec, dt, state_module, dynamics):
    # 每一对状态单独构成一条转移，避免把 teacher-forced 的不相邻预测错误连起来。
    pairs = torch.stack((before.reshape(-1, spec.dim), after.reshape(-1, spec.dim)), 1)
    residuals = consistency_residuals(pairs, spec, dt, state_module, dynamics)
    return {name: dict(rmse=float(value.square().mean().sqrt()),
                       absolute_p95=float(value.abs().flatten().quantile(.95))) for name, value in residuals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--continuation-run', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('输出文件已存在，请指定新的文件')
    torch.set_num_threads(4)
    state_module, dynamics = load_modules()
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    run = json.loads((args.continuation_run / 'report.json').read_text())
    for name, digest in run['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'继续训练实验输入已发生变化：{name}')
    chosen = torch.tensor(run['selected_rising_windows']['validation'])
    batch = gather_windows(data, chosen, 32, 'cpu')
    spec = state_module.StateSpec(**data['metadata']['state_spec'])
    truth, actions, context = batch['states'], batch['actions'], batch['context']
    start, env = chosen.unbind(-1)
    episode = data['episode_ids'][start, env]
    motion = data['motion_ids'][start, env]
    dt = data['metadata']['step_dt']
    models = {'initial': args.base_model,
              'continued': args.continuation_run / f"world_model_{run['additional_updates']}.pt"}
    report = dict(windows=len(chosen), episodes=len(set(zip(env.tolist(), episode.tolist()))), models={},
        note='动作ID表示出生参考，不是逐帧跟踪目标。总旋转角与局部重力方向误差分别统计。差分速度和瞬时速度本来就可能不同，真实轨迹的残差作为对照；没有接触力记录，不能据此证明某次接触导致误差。')
    real_residual = residual_report(truth[:, :-1], truth[:, 1:], spec, dt, state_module, dynamics)
    with torch.no_grad():
        for name, path in models.items():
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint['metadata'] != data['metadata']:
                raise ValueError('模型元信息与采集数据不一致')
            model = dynamics.WorldModel(spec, checkpoint['context_dim'], dynamics.TargetLimiter(**checkpoint['limiter']), checkpoint['hidden_dims'])
            model.load_state_dict(checkpoint['model'], strict=True)
            free = anchored_rollout(model, truth, actions, context)
            teacher = anchored_rollout(model, truth, actions, context, 1)
            angles = dynamics.rotation_error(free[:, -1, 3:7], truth[:, -1, 3:7]) * 180 / math.pi
            gravity = gravity_angle(free[:, -1, 3:7], truth[:, -1, 3:7])
            squared = angles.square()
            total = squared.sum()
            ordering = squared.argsort(descending=True)

            def group(mask):
                p, t = state_module.unpack_state(free[mask, -1], spec), state_module.unpack_state(truth[mask, -1], spec)
                return dict(windows=int(mask.sum()), root_rotation_rmse_deg=float(angles[mask].square().mean().sqrt()),
                    gravity_direction_rmse_deg=float(gravity[mask].square().mean().sqrt()),
                    angular_squared_error_share=float(squared[mask].sum() / total),
                    root_position_rmse_m=float((p['root_pose'][:, :3] - t['root_pose'][:, :3]).square().mean().sqrt()),
                    root_height_rmse_m=float((p['root_pose'][:, 2] - t['root_pose'][:, 2]).square().mean().sqrt()),
                    root_angular_velocity_rmse_rad_s=float((p['root_velocity'][:, 3:] - t['root_velocity'][:, 3:]).square().mean().sqrt()))

            by_motion = {str(int(mid)): group(motion == mid) for mid in motion.unique()}
            by_episode = {}
            for eid, ep in sorted(set(zip(env.tolist(), episode.tolist()))):
                mask = (env == eid) & (episode == ep)
                entry = group(mask)
                entry.update(env_id=eid, episode_id=ep, birth_motion_ids=motion[mask].unique().tolist())
                by_episode[f'{eid}:{ep}'] = entry
            worst = []
            for i in ordering[:8].tolist():
                p, t = state_module.unpack_state(free[i:i + 1, -1], spec), state_module.unpack_state(truth[i:i + 1, -1], spec)
                joint_error = (p['joint_velocity'][0] - t['joint_velocity'][0]).abs()
                top = joint_error.argsort(descending=True)[:5].tolist()
                worst.append(dict(env_id=int(env[i]), episode_id=int(episode[i]), birth_motion_id=int(motion[i]),
                    absolute_start_step=int(start[i]), episode_elapsed_s=float(data['episode_steps'][start[i], env[i]]) * dt,
                    birth_reference_frame=int(data['reference_frames'][start[i], env[i]]),
                    root_rotation_error_deg=float(angles[i]), gravity_direction_error_deg=float(gravity[i]),
                    height_start_m=float(truth[i, 0, 2]), height_end_m=float(truth[i, -1, 2]),
                    predicted_height_end_m=float(free[i, -1, 2]),
                    worst_joint_velocity_errors=[dict(name=data['metadata']['joint_names'][j], absolute_error_rad_s=float(joint_error[j])) for j in top]))
            episode_shares = sorted([v['angular_squared_error_share'] for v in by_episode.values()], reverse=True)
            report['models'][name] = dict(orientation_error_deg=distribution(angles), gravity_error_deg=distribution(gravity),
                aggregate=group(torch.ones(len(chosen), dtype=torch.bool)),
                top_3_episode_squared_error_share=sum(episode_shares[:3]),
                by_birth_motion=by_motion, by_episode=by_episode, worst_windows=worst,
                consistency=dict(real=real_residual,
                    teacher_forced=residual_report(truth[:, :-1], teacher[:, 1:], spec, dt, state_module, dynamics),
                    free_rollout=residual_report(free[:, :-1], free[:, 1:], spec, dt, state_module, dynamics)))
    report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.data, *models.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[完成] ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
