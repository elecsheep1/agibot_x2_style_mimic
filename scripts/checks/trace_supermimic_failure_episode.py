"""展开异常回合与最差预测窗口，核对数据连续性和姿态漂移时序。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from fit_supermimic_smoke import load_modules
from supermimic_phase_sampling import gather_windows
from diagnose_supermimic_rising_rollouts import anchored_rollout
from audit_supermimic_rising_failures import gravity_angle, residual_report


def integrate_angular_velocity(states, dt, dynamics):
    quaternion = dynamics.unit_quat(states[:, 0, 3:7])
    values = [quaternion]
    for step in range(states.shape[1] - 1):
        angular = .5 * (states[:, step, 10:13] + states[:, step + 1, 10:13])
        quaternion = dynamics.unit_quat(dynamics.quat_mul(dynamics.rotation_increment(angular * dt), quaternion))
        values.append(quaternion)
    return torch.stack(values, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--continued-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-id', type=int, default=4)
    parser.add_argument('--episode-id', type=int, default=1)
    parser.add_argument('--start-step', type=int, default=1080)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('输出文件已存在，请指定新的文件')
    torch.set_num_threads(4)
    state_module, dynamics = load_modules()
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    spec = state_module.StateSpec(**data['metadata']['state_spec'])
    dt = data['metadata']['step_dt']
    env = args.env_id
    slots = (data['episode_ids'][:, env] == args.episode_id).nonzero().flatten()
    if len(slots) < 2 or not bool((slots[1:] == slots[:-1] + 1).all()):
        raise ValueError('目标回合不存在或绝对时间索引不连续')
    if args.start_step not in slots or args.start_step + 31 not in slots:
        raise ValueError('预测窗口不在目标回合内部')
    episode = data['state'][slots, env]
    next_episode = data['next_state'][slots, env]
    batch = gather_windows(data, torch.tensor([[args.start_step, env]]), 32, 'cpu')
    truth, action, context = batch['states'], batch['actions'], batch['context']
    fields = state_module.unpack_state(episode, spec)
    ids = [data['metadata']['joint_names'].index(n) for n in data['metadata']['control_joint_names']]
    q = fields['joint_position'][:, ids]
    violation = (data['limiter']['lower'] - q).clamp_min(0) + (q - data['limiter']['upper']).clamp_min(0)
    true_integrated = integrate_angular_velocity(truth, dt, dynamics)
    episode_trace = []
    for local in range(0, min(len(slots), 151), 5):
        s = episode[local]
        up = float(1 - 2 * s[3:5].square().sum())
        episode_trace.append(dict(time_s=round(local * dt, 4), height_m=float(s[2]),
                                  root_up=up, tilt_deg=math.degrees(math.acos(max(-1., min(1., up)))),
                                  linear_speed_m_s=float(s[7:10].norm()), angular_speed_rad_s=float(s[10:13].norm())))
    report = dict(env_id=env, episode_id=args.episode_id, absolute_start_step=args.start_step,
        start_elapsed_s=float(data['episode_steps'][args.start_step, env]) * dt,
        birth_motion_id=int(data['motion_ids'][slots[0], env]), birth_reference_frame=int(data['reference_frames'][slots[0], env]),
        birth_type=int(data['birth_types'][slots[0], env]), episode_length=len(slots),
        data_checks=dict(finite=bool(torch.isfinite(episode).all() and torch.isfinite(next_episode).all()),
            continuity_max_abs=float((next_episode[:-1] - episode[1:]).abs().max()),
            step_counter_contiguous=bool((data['step_ids'][slots[1:], env] == data['step_ids'][slots[:-1], env] + 1).all()),
            quaternion_norm_max_error=float((fields['root_pose'][:, 3:].norm(dim=-1) - 1).abs().max()),
            controlled_joint_limit_max_violation_rad=float(violation.max()),
            terminated_count=int(data['terminated'][slots, env].sum()), truncated_count=int(data['truncated'][slots, env].sum())),
        real_episode_first_3_seconds=episode_trace,
        actual_velocity_integrated_rotation_error32_deg=float(dynamics.rotation_error(true_integrated[:, -1], truth[:, -1, 3:7])[0]) * 180 / math.pi,
        models={}, note='积分对照仅用于诊断，不修改世界模型更新公式；未记录接触力，不能归因于具体碰撞。')
    with torch.no_grad():
        for name, path in [('initial', args.base_model), ('continued', args.continued_model)]:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint['metadata'] != data['metadata']:
                raise ValueError('模型与数据元信息不一致')
            model = dynamics.WorldModel(spec, checkpoint['context_dim'], dynamics.TargetLimiter(**checkpoint['limiter']), checkpoint['hidden_dims'])
            model.load_state_dict(checkpoint['model'], strict=True)
            free = anchored_rollout(model, truth, action, context)
            teacher = anchored_rollout(model, truth, action, context, 1)
            integrated = integrate_angular_velocity(free, dt, dynamics)
            error = dynamics.rotation_error(free[0, :, 3:7], truth[0, :, 3:7]) * 180 / math.pi
            trace = []
            for step in range(33):
                t, p = truth[0, step], free[0, step]
                true_up, pred_up = float(1 - 2 * t[3:5].square().sum()), float(1 - 2 * p[3:5].square().sum())
                trace.append(dict(step=step, prediction_time_s=step * dt, episode_time_s=report['start_elapsed_s'] + step * dt,
                    true_height_m=float(t[2]), predicted_height_m=float(p[2]), true_root_up=true_up, predicted_root_up=pred_up,
                    rotation_error_deg=float(error[step]), gravity_error_deg=float(gravity_angle(p[None, 3:7], t[None, 3:7])[0]),
                    teacher_forced_rotation_error_deg=float(dynamics.rotation_error(teacher[:, step, 3:7], truth[:, step, 3:7])[0]) * 180 / math.pi,
                    true_world_angular_velocity=t[10:13].tolist(), predicted_world_angular_velocity=p[10:13].tolist()))
            first_exceeded = {}
            for threshold in (5, 10, 20, 45, 90):
                indexes = (error > threshold).nonzero().flatten()
                first_exceeded[str(threshold)] = int(indexes[0]) if len(indexes) else None
            report['models'][name] = dict(trace=trace, first_error_exceedance_steps=first_exceeded,
                predicted_velocity_integrated_rotation_error32_deg=float(dynamics.rotation_error(integrated[:, -1], truth[:, -1, 3:7])[0]) * 180 / math.pi,
                raw_vs_velocity_integrated_rotation_difference32_deg=float(dynamics.rotation_error(integrated[:, -1], free[:, -1, 3:7])[0]) * 180 / math.pi,
                consistency=dict(real=residual_report(truth[:, :-1], truth[:, 1:], spec, dt, state_module, dynamics),
                                 predicted=residual_report(free[:, :-1], free[:, 1:], spec, dt, state_module, dynamics)))
    report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.data, args.base_model, args.continued_model)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[完成] ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
