"""Completed-episode diagnostics for non-terminating multi-motion get-up tasks."""
import json
import math

import torch
from pathlib import Path


CONDITION_LABELS = {
    "height": "高度", "pelvis_tilt": "骨盆倾角", "torso_tilt": "躯干倾角",
    "linear_speed": "线速度", "angular_speed": "角速度",
    "knees": "膝角", "feet_contact": "双脚接触",
}


def standing_conditions(env, params):
    """Replay-only measurement at the termination stage, before auto-reset."""
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.getup import ground_contact_force

    robot = env.scene[params["asset_cfg"].name]
    data = robot.data
    upright = -data.projected_gravity_b.torch[:, 2] > math.cos(params["maximum_tilt"])
    torso = params.get("torso_cfg")
    torso_ok = torch.ones_like(upright)
    if torso is not None:
        q = data.body_link_pose_w.torch[:, torso.body_ids, 3:7]
        torso_ok = (1 - 2 * (q[..., 0].square() + q[..., 1].square()) >
                    math.cos(params.get("maximum_torso_tilt", .55))).all(dim=-1)
    return {
        "height": data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2] > params["minimum_height"],
        "pelvis_tilt": upright,
        "torso_tilt": torso_ok,
        "linear_speed": torch.linalg.vector_norm(data.root_lin_vel_w.torch, dim=-1) < params["maximum_linear_speed"],
        "angular_speed": torch.linalg.vector_norm(data.root_ang_vel_w.torch, dim=-1) < params["maximum_angular_speed"],
        "knees": data.joint_pos.torch[:, params["asset_cfg"].joint_ids].abs().amax(dim=-1) < params["maximum_knee_angle"],
        "feet_contact": torch.stack([
            ground_contact_force(env, cfg) > params["force_threshold"]
            for cfg in params["ground_sensor_cfgs"]
        ], dim=-1).all(dim=-1),
    }


def pelvis_pose(env, params):
    """Heading-independent Euler roll/pitch and total tilt, in degrees."""
    robot = env.scene[params["asset_cfg"].name]
    gravity = robot.data.projected_gravity_b.torch
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    pitch = torch.asin(gravity[:, 0].clamp(-1, 1))
    tilt = torch.acos((-gravity[:, 2]).clamp(-1, 1))
    height = robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return torch.stack((torch.rad2deg(roll), torch.rad2deg(pitch), torch.rad2deg(tilt), height), dim=-1)


def standing_reward_comparison(env, params):
    """Evaluate the existing reward twice, changing only its root_up input."""
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.getup import (
        ground_contact_force, standing_quality, standing_quality_reward,
    )
    robot = env.scene["robot"]
    q = robot.data.body_link_pose_w.torch[:, params["torso_cfg"].body_ids, 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    velocity = robot.data.root_link_vel_w.torch
    inputs = dict(
        height=robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2],
        root_up=-robot.data.projected_gravity_b.torch[:, 2],
        torso_up=(1 - 2 * (q[..., 0].square() + q[..., 1].square())).amin(dim=-1),
        linear_velocity=velocity[:, :3], angular_velocity=velocity[:, 3:6],
        feet_support=torch.stack([
            ground_contact_force(env, cfg) > 5.0 for cfg in params["ground_sensor_cfgs"]
        ], dim=-1).float().mean(dim=-1),
        wrist_load=torch.stack([
            ground_contact_force(env, cfg) for cfg in params["wrist_sensor_cfgs"]
        ], dim=-1).sum(dim=-1),
    )
    actual = standing_quality(**inputs)
    # Verify the replay-side input extraction against the production reward.
    expected = standing_quality_reward(env, **params)
    if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-5):
        raise RuntimeError("Replay reward extraction disagrees with standing_quality_reward")
    inputs["root_up"] = torch.ones_like(inputs["root_up"])
    corrected = standing_quality(**inputs)
    return torch.stack((actual, corrected, corrected - actual), dim=-1)


def tail_pose_summary(trace):
    if not trace:
        return None
    end = trace[-1]["time_s"]
    tail = [row for row in trace if row["time_s"] >= end - 5.0 - 1e-6]
    result = dict(start_s=tail[0]["time_s"], end_s=end, samples=len(tail))
    for name in ("roll_deg", "pitch_deg", "tilt_deg", "standing_raw", "root_up1_raw", "root_up1_gain_raw"):
        values = torch.tensor([row[name] for row in tail], dtype=torch.float64)
        result[name] = dict(mean=values.mean().item(), min=values.min().item(),
                            max=values.max().item(), median=values.median().item())
    return result


def empty_stability():
    return dict(steps=0, height_qualified_steps=0, all_pass_steps=0,
                passes={k: 0 for k in CONDITION_LABELS},
                passes_when_high={k: 0 for k in CONDITION_LABELS},
                interruptions=0, interruption_causes={k: 0 for k in CONDITION_LABELS})


def stability_summary(records):
    total = empty_stability()
    for r in records:
        for key in ("steps", "height_qualified_steps", "all_pass_steps", "interruptions"):
            total[key] += r[key]
        for key in ("passes", "passes_when_high", "interruption_causes"):
            for name in CONDITION_LABELS:
                total[key][name] += r[key][name]
    total["pass_percent"] = {k: 100 * v / total["steps"] if total["steps"] else None
                             for k, v in total["passes"].items()}
    total["pass_percent_when_high"] = {
        k: 100 * v / total["height_qualified_steps"] if total["height_qualified_steps"] else None
        for k, v in total["passes_when_high"].items()}
    total["all_pass_percent"] = 100 * total["all_pass_steps"] / total["steps"] if total["steps"] else None
    return total


class MultiMotionDiagnostics:
    def __init__(self, env, episodes_per_env, output_path, require_zero_assist=True):
        self.env = env
        self.require_zero_assist = require_zero_assist
        self.quota = episodes_per_env
        self.output_path = Path(output_path).resolve()
        self.counts = [0] * env.num_envs
        self.rows = []
        self.steps = 0
        self.reported_steps = -1
        self.latest = None
        self.motion_paths = list(env._getup_reference_motion_paths)
        progress_cfg = env.termination_manager.get_term_cfg('progress')
        self.required_hold = float(progress_cfg.params['hold_time'])
        self.condition_thresholds = {
            k: progress_cfg.params[k] for k in (
                "minimum_height", "maximum_tilt", "maximum_linear_speed",
                "maximum_angular_speed", "maximum_knee_angle", "force_threshold")
        }
        self.condition_thresholds["maximum_torso_tilt"] = progress_cfg.params.get("maximum_torso_tilt", .55)
        self._stability = [empty_stability() for _ in self.counts]
        self._previous_hold = [0.0 for _ in self.counts]
        self._step_conditions = None
        self._pose_traces = [[] for _ in self.counts]
        self._step_pose = None
        self._step_rewards = None
        reward_cfg = env.reward_manager.get_term_cfg("standing")
        self.standing_reward_weight = float(reward_cfg.weight)
        self._pose_stride = max(1, round(.1 / env.step_dt))
        self._episode_steps = [0 for _ in self.counts]
        # Only this play process is instrumented. Training source and criteria are unchanged.
        original_compute = env.termination_manager.compute

        def compute_with_diagnostics(*args, **kwargs):
            result = original_compute(*args, **kwargs)
            self._step_conditions = {
                key: mask.clone() for key, mask in standing_conditions(env, progress_cfg.params).items()
            }
            self._step_pose = pelvis_pose(env, progress_cfg.params).clone()
            self._step_rewards = standing_reward_comparison(env, reward_cfg.params).clone()
            return result

        env.termination_manager.compute = compute_with_diagnostics

    @property
    def complete(self):
        return bool(self.quota and min(self.counts) >= self.quota)

    def update(self, dones, extras):
        progress = extras.get('getup_progress')
        if progress is None:
            raise RuntimeError('Missing extras[getup_progress]; use X2-Prone-MultiMotion-v0')
        for key in ("assist_episode_force_n", "assist_applied_force_n"):
            if key not in progress:
                raise RuntimeError("Missing assist force snapshot; cannot certify zero-assist evaluation")
            if self.require_zero_assist and bool((progress[key].abs() > 1e-6).any()):
                raise RuntimeError("Nonzero force detected in zero-assist evaluation")
        if self._step_conditions is None:
            raise RuntimeError("Missing pre-reset standing measurements")
        masks = torch.stack(list(self._step_conditions.values()), dim=-1)
        if not torch.equal(masks.all(dim=-1), progress["hold_seconds"] > 0):
            raise RuntimeError("Replay standing conditions disagree with actual hold timer")
        measurements = masks.cpu().tolist()
        holds = progress["hold_seconds"].cpu().tolist()
        poses = self._step_pose.cpu().tolist()
        reward_comparisons = self._step_rewards.cpu().tolist()
        done_list = dones.cpu().tolist()
        for env_id, values in enumerate(measurements):
            if self.quota and self.counts[env_id] >= self.quota:
                continue
            self._episode_steps[env_id] += 1
            step = self._episode_steps[env_id]
            if step == 1 or step % self._pose_stride == 0 or done_list[env_id]:
                roll, pitch, tilt, height = poses[env_id]
                actual, corrected, gain = reward_comparisons[env_id]
                self._pose_traces[env_id].append(dict(
                    time_s=round(step * self.env.step_dt, 6),
                    roll_deg=roll, pitch_deg=pitch, tilt_deg=tilt, height_m=height,
                    standing_raw=actual, root_up1_raw=corrected, root_up1_gain_raw=gain))
            r = self._stability[env_id]
            r["steps"] += 1
            high = values[0]
            r["height_qualified_steps"] += int(high)
            r["all_pass_steps"] += int(all(values))
            interrupted = self._previous_hold[env_id] > 0 and not all(values)
            r["interruptions"] += int(interrupted)
            for name, passed in zip(CONDITION_LABELS, values):
                r["passes"][name] += int(passed)
                r["passes_when_high"][name] += int(high and passed)
                r["interruption_causes"][name] += int(interrupted and not passed)
            self._previous_hold[env_id] = holds[env_id]
        self.steps += 1
        # These snapshots belong to the just-finished step, before any automatic reset.
        self.latest = {key: value[0].item() for key, value in progress.items()}
        terminated = self.env.reset_terminated
        truncated = self.env.reset_time_outs
        for env_id in dones.nonzero(as_tuple=False).flatten().tolist():
            if self.quota and self.counts[env_id] >= self.quota:
                continue
            self.counts[env_id] += 1
            self.rows.append({
                'env_id': env_id,
                'episode_number': self.counts[env_id],
                'motion_id': int(progress['motion_ids'][env_id]),
                'succeeded': bool(progress['succeeded'][env_id]),
                'first_success_time': float(progress['first_success_time'][env_id]),
                'max_hold_seconds': float(progress['max_hold_seconds'][env_id]),
                'final_hold_seconds': float(progress['hold_seconds'][env_id]),
                'terminated': bool(terminated[env_id]),
                'timed_out': bool(truncated[env_id]),
                'standing_diagnostics': self._stability[env_id],
                'pelvis_trace': self._pose_traces[env_id],
                'pelvis_last_5s': tail_pose_summary(self._pose_traces[env_id]),
            })
            self._stability[env_id] = empty_stability()
            self._pose_traces[env_id] = []
            self._episode_steps[env_id] = 0
            self._previous_hold[env_id] = 0.0

    def summarize(self):
        def stats(rows):
            count = len(rows)
            successes = sum(row['succeeded'] for row in rows)
            first = [row['first_success_time'] for row in rows if row['succeeded']]
            # A separate measure: still satisfies the full hold requirement at episode end.
            stable_at_end = sum(row['final_hold_seconds'] + 1e-6 >= self.required_hold for row in rows)
            return dict(
                standing_diagnostics=stability_summary([r['standing_diagnostics'] for r in rows]),
                completed=count, successes=successes,
                success_percent=100 * successes / count if count else None,
                stable_at_end=stable_at_end,
                stable_at_end_percent=100 * stable_at_end / count if count else None,
                mean_first_success_seconds=sum(first) / len(first) if first else None,
                mean_max_hold_seconds=sum(row['max_hold_seconds'] for row in rows) / count if count else None,
                mean_final_hold_seconds=sum(row['final_hold_seconds'] for row in rows) / count if count else None,
                timeouts=sum(row['timed_out'] for row in rows),
                terminations=sum(row['terminated'] for row in rows),
            )
        per_motion = []
        for motion_id, path in enumerate(self.motion_paths):
            per_motion.append(dict(motion_id=motion_id, file=Path(path).name,
                                   **stats([row for row in self.rows if row['motion_id'] == motion_id])))
        params = self.env.cfg.events.reset_from_reference.params
        return dict(
            complete=self.complete, episodes_per_env=self.quota, zero_assist_verified=self.require_zero_assist and self.steps > 0,
            target_episodes=self.quota * self.env.num_envs if self.quota else None,
            completed_per_env=self.counts.copy(),
            simulation_seconds=self.steps * self.env.step_dt,
            start_phase=params.get('start_phase'), end_phase=params.get('end_phase'),
            condition_thresholds=self.condition_thresholds,
            standing_reward_weight=self.standing_reward_weight,
            standing_reward_step_scale=self.standing_reward_weight * self.env.step_dt,
            required_hold_seconds=self.required_hold,
            overall=stats(self.rows), per_motion=per_motion,
            latest_env0=self.latest,
        )

    def report(self):
        summary = self.summarize()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(dict(summary=summary, episodes=self.rows),
                                              ensure_ascii=False, indent=2), encoding='utf-8')
        def pct(value):
            return '暂无' if value is None else f'{value:.2f}%'
        overall = summary['overall']
        final = '最终值' if self.complete else '暂计值'
        line = '─' * 76
        print(f'\n{line}\n[多动作起身诊断] {"均衡验收已完成" if self.complete else "验收进行中"}'
              f' | 仿真 {summary["simulation_seconds"]:.2f} s\n{line}', flush=True)
        print(f'出生范围：{summary["start_phase"]:.4f}～{summary["end_phase"]:.4f}'
              f' | 连续稳定要求：{self.required_hold:.2f} s', flush=True)
        print('成功不会重置；只在完整回合结束后统计。' + ('环境起身辅助力已关闭，并逐步核对为零。' if self.require_zero_assist else '本次带辅助力，仅为诊断对照，不是零辅助验收。'), flush=True)
        target = summary['target_episodes']
        print(f'采样进度：{len(self.rows)}/{target if target is not None else "不限"}'
              f' | 每环境前 {self.quota if self.quota else "不限"} 回合'
              f' | 覆盖 {sum(n > 0 for n in self.counts)}/{len(self.counts)} 环境', flush=True)
        print(f'曾成功比例：{pct(overall["success_percent"])}（{final}）'
              f' | 成功 {overall["successes"]}/{overall["completed"]}', flush=True)
        print(f'回合末仍连续稳定达标：{pct(overall["stable_at_end_percent"])}', flush=True)
        if overall['completed']:
            print(f'平均最长连续稳定：{overall["mean_max_hold_seconds"]:.2f} s'
                  f' | 平均末态连续稳定：{overall["mean_final_hold_seconds"]:.2f} s', flush=True)
        if overall['mean_first_success_seconds'] is not None:
            print(f'首次成功平均时间：{overall["mean_first_success_seconds"]:.2f} s（仅成功回合）', flush=True)
        print(f'结束原因：超时 {overall["timeouts"]} | 其他终止 {overall["terminations"]}', flush=True)
        print('\n按出生动作统计（随机加权采样，不保证每条动作回合数相同）：', flush=True)
        for row in summary['per_motion']:
            print(f'  {row["motion_id"]}: {row["file"]}\n'
                  f'     回合 {row["completed"]} | 曾成功 {pct(row["success_percent"])}'
                  f' | 末态稳定 {pct(row["stable_at_end_percent"])}', flush=True)
        if overall['completed']:
            detail = overall['standing_diagnostics']
            print("\n[站稳条件诊断] 仅统计已完成的验收回合：", flush=True)
            print(f"全部条件同时达标：{pct(detail['all_pass_percent'])}"
                  f" | 连续计时中断：{detail['interruptions']} 次", flush=True)
            for name, label in CONDITION_LABELS.items():
                print(f"  {label}：全回合 {pct(detail['pass_percent'][name])}"
                      f" | 高度达标时 {pct(detail['pass_percent_when_high'][name])}"
                      f" | 中断涉及 {detail['interruption_causes'][name]} 次", flush=True)
            print("高度达标时比例仅筛选高度超过成功阈值的帧，不代表已经站稳。"
                  "同次中断可涉及多项；回合结束本身不记为中断。", flush=True)
        if self.rows:
            print("\n[骨盆末段倾角] 各回合最后5秒；角度单位为度，正俯仰为前倾：", flush=True)
            for row in self.rows:
                tail = row['pelvis_last_5s']
                print(f"  环境 {row['env_id']} | 动作 {row['motion_id']} | 回合 {row['episode_number']}"
                      f" | 滚转均值 {tail['roll_deg']['mean']:+.1f}"
                      f" | 俯仰均值 {tail['pitch_deg']['mean']:+.1f}"
                      f" | 总倾角均值/最大 {tail['tilt_deg']['mean']:.1f}/{tail['tilt_deg']['max']:.1f}", flush=True)
            print("[骨盆回正奖励对照] 最后5秒的原始站立奖励；只修改root_up输入，其他输入固定：", flush=True)
            for row in self.rows:
                tail = row['pelvis_last_5s']
                actual = tail['standing_raw']['mean']
                corrected = tail['root_up1_raw']['mean']
                gain = tail['root_up1_gain_raw']['mean']
                print(f"  环境 {row['env_id']} | 动作 {row['motion_id']}"
                      f" | 当前 {actual:.6f} → root_up=1 {corrected:.6f}"
                      f" | 增益 {gain:.6f} | 加权每步增益 {gain * self.standing_reward_weight * self.env.step_dt:.6f}", flush=True)
            print("这是公式输入对照，不代表机器人实际回正后的动力学收益。", flush=True)
            print("完整0.1秒采样见JSON的episodes/pelvis_trace；总倾角用于大倾斜姿态判断。", flush=True)
        print('“曾成功”之后仍可能倒下，请结合末态稳定比例和画面判断。', flush=True)
        print(f'完整报告：{self.output_path}\n{line}', flush=True)
        self.reported_steps = self.steps
