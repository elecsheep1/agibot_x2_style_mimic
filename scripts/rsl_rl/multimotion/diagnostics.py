"""Completed-episode diagnostics for non-terminating multi-motion get-up tasks."""
import json
from pathlib import Path


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
            })

    def summarize(self):
        def stats(rows):
            count = len(rows)
            successes = sum(row['succeeded'] for row in rows)
            first = [row['first_success_time'] for row in rows if row['succeeded']]
            # A separate measure: still satisfies the full hold requirement at episode end.
            stable_at_end = sum(row['final_hold_seconds'] + 1e-6 >= self.required_hold for row in rows)
            return dict(
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
        print('“曾成功”之后仍可能倒下，请结合末态稳定比例和画面判断。', flush=True)
        print(f'完整报告：{self.output_path}\n{line}', flush=True)
        self.reported_steps = self.steps
