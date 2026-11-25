"""固定出生状态的真实物理评估，不向训练 replay 写入数据。"""
from contextlib import contextmanager
import math
import random

import numpy as np
import torch

from getup_runtime import tensor
from head_diagnostics import head_snapshot, summarize_head_trace, physical_head_geometry, reset_evaluation_sensors


@contextmanager
def preserved_rng(device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def build_cohort(runtime, seed):
    env, dataset = runtime.env, runtime.reset_reference
    robot = env.scene['robot']
    generator = torch.Generator().manual_seed(seed)
    count = len(dataset.clips)
    motion_ids = torch.arange(env.num_envs, device=env.device) % count
    kinds = (torch.arange(env.num_envs, device=env.device) // count) % 2
    frame_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    ranges = env.cfg.events.reset_from_reference.params.get('transition_frame_ranges')
    states = {name: [] for name in ('root_pose', 'root_velocity', 'joint_pos', 'joint_vel')}
    for i, mid in enumerate(motion_ids.tolist()):
        clip = dataset.clips[mid]
        if int(kinds[i]) == 1:
            low, high = ranges[mid] if ranges is not None else (0, len(clip['root_pose']) - 1)
            high = min(int(high), len(clip['root_pose']) - 1)
            low = max(0, int(low))
            if low > high:
                raise ValueError('固定评估的参考帧区间无效')
            frame_ids[i] = int(torch.randint(low, high + 1, (1,), generator=generator))
        frame = int(frame_ids[i])
        for name in states:
            states[name].append(clip[name][frame].clone())
    states = {name: torch.stack(values) for name, values in states.items()}
    initial = frame_ids == 0
    states['root_velocity'][initial] = 0
    states['joint_vel'][initial] = 0
    states['root_pose'][:, 2] += env.cfg.events.reset_from_reference.params.get('height_offset', .01)
    limits = tensor(robot.data.joint_pos_limits)
    unclamped = states['joint_pos']
    states['joint_pos'] = unclamped.clamp(min=limits[:, :, 0], max=limits[:, :, 1])
    states['joint_vel'] = states['joint_vel'].clamp(min=-tensor(robot.data.joint_vel_limits), max=tensor(robot.data.joint_vel_limits))
    return dict(states=states, motion_ids=motion_ids, frame_ids=frame_ids, birth_types=torch.where(kinds == 0, 0, 2),
                joint_clamp_counts=((states['joint_pos'] - unclamped).abs() > 1e-6).sum(-1),
                motion_paths=[str(p) for p in dataset.paths], seed=seed)


def geometric_stable(fields, torso_up, knee_ids):
    q = fields['root_pose'][:, 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    root_up = 1 - 2 * q[:, :2].square().sum(-1)
    return ((fields['root_pose'][:, 2] >= .5) & (root_up >= math.cos(.35))
            & (torso_up >= math.cos(.55))
            & (fields['root_velocity'][:, :3].norm(dim=-1) <= .2)
            & (fields['root_velocity'][:, 3:].norm(dim=-1) <= .5)
            & (fields['joint_position'][:, knee_ids].abs().amax(-1) <= 1.5))


def actual_reward_terms(env, actual_reward):
    """读取本步加权奖励；当前 IsaacLab 缓冲保存的是未乘 dt 的奖励率。"""
    manager = env.reward_manager
    names = tuple(manager.active_terms)
    values = manager._step_reward.detach().clone() * env.step_dt
    if values.shape != (env.num_envs, len(names)):
        raise RuntimeError('RewardManager 分项缓冲形状不兼容')
    torch.testing.assert_close(values.sum(-1), actual_reward, atol=2e-5, rtol=2e-5)
    return {name: values[:, i] for i, name in enumerate(names)}


class RewardTermTotals:
    """累计首回合分项回报，不纳入自动重置后回合。"""

    def __init__(self, count, device):
        self.count, self.device = count, device
        self.real, self.proxy = {}, {}

    def update(self, active, real, proxy):
        for destination, terms in ((self.real, real), (self.proxy, proxy)):
            for name, value in terms.items():
                if value.shape != (self.count,) or not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f'评估奖励分项形状或数值无效：{name}')
                if name not in destination:
                    destination[name] = torch.zeros(self.count, device=self.device)
                destination[name] += torch.where(active, value, 0.)

    def metrics(self, mask):
        result = {f'real_terms/{name}': float(value[mask].mean())
                  for name, value in self.real.items()}
        result.update({f'proxy_terms/{name}': float(value[mask].mean())
                       for name, value in self.proxy.items()})
        aliases = {'standing': 'standing', 'head_contact': 'head_contact',
                   'action_rate': 'action_rate', 'joint_pos_limits': 'joint_limits',
                   'joint_acc': 'joint_acc', 'out_of_bounds_failure': 'out_of_bounds',
                   'getup_progress_reward': 'progress'}
        for real, proxy in aliases.items():
            if real in self.real and proxy in self.proxy:
                result[f'term_gaps/{real}'] = float((self.real[real][mask] - self.proxy[proxy][mask]).mean())
        return result


class TaskEvaluationTotals:
    """只累计每个环境首回合，使用环境重置前的任务快照。"""

    def __init__(self, count, device):
        self.actual_return = torch.zeros(count, device=device)
        self.recomputed_return = torch.zeros(count, device=device)
        self.success = torch.zeros(count, dtype=torch.bool, device=device)
        self.final_hold = torch.zeros(count, device=device)
        self.max_hold = torch.zeros(count, device=device)
        self.first_success = torch.full((count,), -1., device=device)

    def update(self, active, actual_reward, recomputed_reward, progress):
        self.actual_return += torch.where(active, actual_reward, 0.)
        self.recomputed_return += torch.where(active, recomputed_reward, 0.)
        for target, key in ((self.success, 'succeeded'), (self.final_hold, 'hold_seconds'),
                            (self.max_hold, 'max_hold_seconds'), (self.first_success, 'first_success_time')):
            target.copy_(torch.where(active, progress[key], target))


@torch.no_grad()
def evaluate(runtime, cohort, seconds, seed):
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.getup import _write_reference_state
    from x2_style_mimic.algorithms.supermimic.state import unpack_state
    env, trainer = runtime.env, runtime.trainer
    n, device, dt = env.num_envs, env.device, env.step_dt
    actor, style = trainer.actor, trainer.style
    actor_mode, style_mode = actor.training, style.training
    replay_steps_before = runtime.replay.total_steps
    saved_modules = {name: {key: value.detach().clone() for key, value in module.state_dict().items()}
                     for name, module in runtime.modules.items()}
    knee_ids = [env.scene['robot'].joint_names.index(name) for name in ('left_knee_joint', 'right_knee_joint')]
    try:
        with preserved_rng(device):
            torch.manual_seed(seed)
            env.reset()
            ids = torch.arange(n, device=device)
            states = cohort['states']
            pose = states['root_pose'].clone()
            pose[:, :3] += env.scene.env_origins
            _write_reference_state(env, ids, env.scene['robot'], pose, states['root_velocity'], states['joint_pos'], states['joint_vel'])
            env.action_manager.reset(ids)
            # 固定 cohort 会在 env.reset 后再次设置姿态，旧传感器历史不能沿用。
            reset_evaluation_sensors(env, ids)
            env._getup_birth_types[ids] = cohort['birth_types']
            env._getup_reference_motion_ids[ids] = cohort['motion_ids']
            env._getup_reference_frame_ids[ids] = cohort['frame_ids']
            current = env.read_world_state()
            expected = cohort['states']
            torch.testing.assert_close(current[:, :3], expected['root_pose'][:, :3], atol=2e-5, rtol=0)
            torch.testing.assert_close(current[:, 3:7], expected['root_pose'][:, 3:7], atol=2e-5, rtol=0)
            torch.testing.assert_close(unpack_state(current, env.world_state_spec)['joint_position'], expected['joint_pos'], atol=2e-5, rtol=0)
            history = current[:, None].expand(-1, 5, -1).clone()
            amp_history = trainer.amp_features(current)[:, None].expand(-1, 5, -1).clone()
            actor.eval()
            style.eval()
            active = torch.ones(n, dtype=torch.bool, device=device)
            task_totals = TaskEvaluationTotals(n, device)
            term_totals = RewardTermTotals(n, device)
            required_hold = float(env.cfg.terminations.progress.params['hold_time'])
            steps = torch.zeros(n, device=device)
            quality_sum, task_return, total_return, style_sum, style_count = [torch.zeros(n, device=device) for _ in range(5)]
            hold_counts = torch.zeros(n, dtype=torch.long, device=device)
            hold, max_hold, final_hold = [torch.zeros(n, device=device) for _ in range(3)]
            succeeded = torch.zeros(n, dtype=torch.bool, device=device)
            first = torch.full((n,), -1., device=device)
            last_state = current.clone()
            timeout_count = failure_count = 0
            traces = []
            head_trace = []
            for step in range(math.ceil(seconds / dt)):
                mask = active.clone()
                action = actor(trainer.proprioception(history), torch.empty(n, 0, device=device))
                action[~active] = 0
                _, actual_reward, terminated, truncated, extras = env.step(action)
                real_terms = actual_reward_terms(env, actual_reward)
                following = extras['world_model']['next_state']
                head_sample = head_snapshot(env, trainer.task, following, mask & ~(terminated | truncated))
                head_sample.update(physical_head_geometry(env, trainer.task, following))
                head_trace.append(dict(time_s=(step + 1) * dt, **head_sample))
                if not bool(torch.isfinite(following).all()):
                    raise RuntimeError('真实评估状态包含 NaN/Inf')
                history = torch.cat((history[:, 1:], following[:, None]), 1)
                amp_history = torch.cat((amp_history[:, 1:], trainer.amp_features(following)[:, None]), 1)
                # 初始重复帧不计入真实 AMP 窗口评分。
                eligible = step + 2 >= amp_history.shape[1]
                style_reward = style.evaluate(amp_history) if eligible else torch.zeros(n, device=device)
                goals = torch.empty(n, 0, device=device)
                raw_task, proxy_terms = trainer.task.reward(current, action, following, goals, torch.zeros_like(style_reward))
                term_totals.update(mask, real_terms, proxy_terms)
                raw_total, _ = trainer.task.reward(current, action, following, goals, style_reward)
                task_totals.update(mask, actual_reward, raw_task, extras['getup_progress'])
                quality = trainer.task.standing_quality(following)
                fields = unpack_state(following, env.world_state_spec)
                body_pose, _ = trainer.task.kinematics(fields['root_pose'], fields['root_velocity'], fields['joint_position'], fields['joint_velocity'])
                torso_up = trainer.task.upright_component(body_pose[:, trainer.task.torso_index, 3:7])
                stable = geometric_stable(fields, torso_up, knee_ids) & ~terminated
                hold_counts = torch.where(mask, torch.where(stable, hold_counts + 1, torch.zeros_like(hold_counts)), hold_counts)
                hold = hold_counts * dt
                max_hold = torch.maximum(max_hold, hold)
                final_hold = torch.where(mask, hold, final_hold)
                newly = mask & ~succeeded & (hold >= 3.)
                first[newly] = (step + 1) * dt
                succeeded |= newly
                steps += mask
                quality_sum += quality * mask
                task_return += raw_task * mask
                total_return += raw_total * mask
                if eligible:
                    style_sum += style_reward * mask
                    style_count += mask
                last_state[mask] = following[mask]
                timeout_count += int((truncated & mask).sum())
                failure_count += int((terminated & mask).sum())
                if step % max(1, round(.1 / dt)) == 0:
                    traces.append(dict(time_s=(step + 1) * dt, height=fields['root_pose'][:, 2].tolist(),
                                       standing_quality=quality.tolist(), active=mask.tolist()))
                active &= ~(terminated | truncated)
                current = following
                if not bool(active.any()):
                    break
            per_env = []
            for i in range(n):
                per_env.append(dict(env_id=i, birth_type=int(cohort['birth_types'][i]), motion_id=int(cohort['motion_ids'][i]),
                    real_reward_terms={name: float(value[i]) for name, value in term_totals.real.items()},
                    proxy_reward_terms={name: float(value[i]) for name, value in term_totals.proxy.items()},
                    frame_id=int(cohort['frame_ids'][i]), geometric_success=bool(succeeded[i]),
                    geometric_stable_at_end=bool(final_hold[i] >= 3.), max_hold_seconds=float(max_hold[i]),
                    final_hold_seconds=float(final_hold[i]), first_success_seconds=float(first[i]),
                    task_return=float(task_totals.actual_return[i]),
                    recomputed_task_return=float(task_totals.recomputed_return[i]),
                    recomputed_total_return=float(total_return[i]),
                    task_success=bool(task_totals.success[i]),
                    task_stable_at_end=bool(task_totals.final_hold[i] >= required_hold),
                    task_max_hold_seconds=float(task_totals.max_hold[i]),
                    task_final_hold_seconds=float(task_totals.final_hold[i]),
                    task_first_success_seconds=float(task_totals.first_success[i]),
                    standing_quality_mean=float(quality_sum[i] / steps[i].clamp_min(1)),
                    height_at_end=float(last_state[i, 2]), evaluated_steps=int(steps[i])))
            def stats(which):
                mask = torch.tensor(which, dtype=torch.bool, device=device)
                return dict(standing_quality_mean=float((quality_sum[mask] / steps[mask].clamp_min(1)).mean()),
                    real_task_return=float(task_totals.actual_return[mask].mean()),
                    recomputed_task_return=float(task_totals.recomputed_return[mask].mean()),
                    recomputed_total_return=float(total_return[mask].mean()),
                    task_return_gap=float((task_totals.actual_return[mask] - task_totals.recomputed_return[mask]).mean()),
                    real_task_reward_mean_per_step=float(task_totals.actual_return[mask].sum() / steps[mask].sum().clamp_min(1)),
                    task_success_rate=float(task_totals.success[mask].float().mean()),
                    task_stable_at_end_rate=float((task_totals.final_hold[mask] >= required_hold).float().mean()),
                    task_max_hold_seconds=float(task_totals.max_hold[mask].mean()),
                    task_final_hold_seconds=float(task_totals.final_hold[mask].mean()),
                    style_reward_mean=float(style_sum[mask].sum() / style_count[mask].sum().clamp_min(1)),
                    geometric_success_rate=float(succeeded[mask].float().mean()),
                    geometric_stable_at_end_rate=float((final_hold[mask] >= 3.).float().mean()),
                    max_hold_seconds=float(max_hold[mask].mean()), final_hold_seconds=float(final_hold[mask].mean()))
            metrics = stats([True] * n)
            metrics.update(term_totals.metrics(torch.ones(n, dtype=torch.bool, device=device)))
            for kind, name in [(0, 'reference_initial'), (2, 'reference_rsi')]:
                mask = (cohort['birth_types'] == kind).tolist()
                if any(mask):
                    metrics.update({name + '/' + key: value for key, value in stats(mask).items()})
                    metrics.update({name + '/' + key: value for key, value in term_totals.metrics(torch.tensor(mask, device=device)).items()})
            metrics.update(timeouts=timeout_count, failures=failure_count)
            metrics.update({f'head_diagnostic/{key}': value for key, value in summarize_head_trace(head_trace).items()})
            report = dict(metrics=metrics, per_env=per_env, trace=traces, duration_limit_s=seconds,
                          head_contact_trace=head_trace,
                          head_diagnostic_note='每控制步采样；valid 排除自动重置及已结束回合。current 与 history 标签分别按实际力阈值生成；proxy 不表示牛顿力。',
                          schema_version=3, contact_conditions_checked=True,
                          note='task_* 使用原起身环境的完整成功判据，含双脚接触；geometric_* 仅为几何辅助诊断。real_task_return 为 RewardManager 返回值，不含 AMP；recomputed_* 为想象公式在真实状态上的重算值，其中 total 包含未门控风格奖励。评估只统计各环境首回合；达到评估时长不代表完整回合结束。')
    finally:
        actor.train(actor_mode)
        style.train(style_mode)
        # 真实评估操作过仿真，必须清空并重建训练历史，标记手动回合边界。
        runtime.runner.reset()
    for name, module in runtime.modules.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, saved_modules[name][key], atol=0, rtol=0)
    report['evaluation_isolation_verified'] = True
    if runtime.replay.total_steps != replay_steps_before:
        raise RuntimeError('评估数据意外写入训练 replay')
    return report
