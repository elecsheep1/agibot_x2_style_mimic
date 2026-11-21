"""起身实验调度：组装现有算法，执行预热、联合训练、评估和保存。"""
import hashlib
from pathlib import Path

import torch

from getup_checkpointing import plot_metrics, restore_rng, save_checkpoint
from getup_config import PROJECT
from getup_evaluation import build_cohort, evaluate
from getup_logging import GetupLogger
from getup_warmup import warmup_world_model


def file_hashes(paths):
    return {
        str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in sorted(paths, key=str)
    }


class GetupTrainingRun:
    """实验层 runner；网络更新仍由 source 中已有 runner/trainer 完成。"""

    def __init__(self, runtime, cfg, checkpoint, source_before):
        self.runtime = runtime
        self.cfg = cfg
        self.checkpoint = checkpoint
        self.start = int(checkpoint['iteration']) if checkpoint else 0
        if self.start >= cfg.iterations:
            raise ValueError('目标总轮数必须大于保存的轮数')
        self.source_before = source_before
        self.cohort = build_cohort(runtime, cfg.eval_seed)
        self.metadata = self._metadata()
        self.logger = None
        if checkpoint is not None:
            self._check_resume()
            restore_rng(checkpoint)

    def _metadata(self):
        cfg, runtime = self.cfg, self.runtime
        inputs = set(runtime.all_reference_paths) | {
            cfg.motion_manifest, cfg.urdf, Path(runtime.metadata['reset_manifest']),
        }
        if runtime.offline is not None:
            inputs |= {cfg.offline_world_data, cfg.offline_world_split}
        return dict(
            **runtime.metadata,
            input_sha256=file_hashes(inputs),
            source_code_sha256=self.source_before,
            evaluation_seed=cfg.eval_seed,
            evaluation_seconds=cfg.eval_seconds,
            training_schedule={
                key: getattr(cfg, key) for key in (
                    'collection_steps', 'batch_size', 'world_updates', 'discriminator_updates',
                )
            },
        )

    def _check_resume(self):
        saved = self.checkpoint
        for key in (
            'input_sha256', 'source_code_sha256', 'action_noise', 'world_horizons',
            'offline_world_fraction', 'evaluation_seed', 'evaluation_seconds', 'training_schedule',
        ):
            if saved['metadata'][key] != self.metadata[key]:
                raise ValueError(f'恢复训练的配置或输入已改变：{key}')
        for key in ('motion_ids', 'frame_ids', 'birth_types'):
            torch.testing.assert_close(
                self.cohort[key], saved['cohort'][key].to(self.runtime.env.device), atol=0, rtol=0,
            )
        for key, value in self.cohort['states'].items():
            torch.testing.assert_close(
                value, saved['cohort']['states'][key].to(self.runtime.env.device), atol=0, rtol=0,
            )

    def _save(self, name, iteration):
        save_checkpoint(
            self.cfg.output / name, self.runtime, iteration, self.cohort, self.metadata,
        )

    def _evaluate(self, iteration):
        print(f'[真实评估 {iteration}] 开始，每环境最多 {self.cfg.eval_seconds:g} 秒仿真时间', flush=True)
        report = evaluate(self.runtime, self.cohort, self.cfg.eval_seconds, self.cfg.eval_seed)
        self.logger.write_json(f'evaluation_{iteration:06d}.json', report)
        self.logger.record('evaluation', iteration, report['metrics'])
        self.logger.flush()
        metrics = report['metrics']
        print(
            f"[真实评估 {iteration}] 站立质量={metrics['standing_quality_mean']:.4f}，"
            f"任务成功率={metrics['task_success_rate']:.1%}，"
            f"任务最长稳定={metrics['task_max_hold_seconds']:.2f}s，"
            f"真实任务回报={metrics['real_task_return']:.4f}，"
            f"公式重算回报={metrics['recomputed_task_return']:.4f}",
            flush=True,
        )

    def _train_iteration(self, iteration):
        cfg = self.cfg
        horizon = cfg.world_horizons[(iteration - 1) % len(cfg.world_horizons)]
        metrics = self.runtime.runner.train_iteration(
            self.runtime.reference,
            collection_steps=cfg.collection_steps,
            batch_size=cfg.batch_size,
            world_horizon=horizon,
            world_updates=cfg.world_updates,
            discriminator_updates=cfg.discriminator_updates,
            offline_world_replay=self.runtime.offline,
            offline_world_fraction=cfg.offline_world_fraction,
        )
        metrics['world_horizon'] = horizon
        metrics['imagination_horizon'] = cfg.imagination_horizon
        metrics['imagined_task_reward'] = metrics['imagined_reward'] - metrics['reward/style']
        self.logger.record('train', iteration, metrics)
        if iteration % cfg.log_interval == 0 or iteration == self.start + 1:
            print(
                f'[联合训练 {iteration}/{cfg.iterations}] '
                f"模型损失={metrics['world_loss']:.6f}，"
                f"想象任务奖励={metrics['imagined_task_reward']:.6f}，"
                f"动作RMS={metrics['collection/action_rms']:.3f}",
                flush=True,
            )
            self.logger.flush()

    def _finish(self):
        after = file_hashes((PROJECT / 'source').rglob('*.py'))
        if after != self.source_before:
            raise RuntimeError('运行期间 source 文件发生了改变，请核对修改来源')
        self.logger.write_json('summary.json', dict(
            completed_iterations=self.cfg.iterations,
            formal_source_unchanged=True,
            total_training_transitions=self.runtime.runner.total_env_steps * self.runtime.env.num_envs,
            replay_steps=self.runtime.replay.total_steps,
            output=str(self.cfg.output),
            evaluation_note='几何站稳不包含脚接触检查，不等同于原起身任务的完整成功率。',
        ))
        self.logger.flush()
        print(f'[完成] {self.cfg.output}', flush=True)

    def run(self):
        cfg, runtime = self.cfg, self.runtime
        print(f'[输出目录] {cfg.output}', flush=True)
        print(
            f"[起身实验] 初始化={runtime.metadata['initialization_mode']}，环境={cfg.num_envs}，"
            f'模型窗口={list(cfg.world_horizons)}，策略想象={cfg.imagination_horizon}，联合训练={cfg.iterations}轮',
            flush=True,
        )
        with GetupLogger(cfg, self.metadata, self.cohort) as self.logger:
            self._save(f'checkpoint_{self.start:06d}.pt', self.start)
            self._evaluate(self.start)
            needs_warmup = not runtime.warmup_complete
            warmup_world_model(runtime, cfg, self.logger.record)
            if needs_warmup:
                self._save('checkpoint_warmup.pt', self.start)
                self._save('latest.pt', self.start)
            for iteration in range(self.start + 1, cfg.iterations + 1):
                self._train_iteration(iteration)
                if iteration % cfg.eval_interval == 0 or iteration == cfg.iterations:
                    self._evaluate(iteration)
                    plot_metrics(cfg.output)
                if iteration % cfg.checkpoint_interval == 0 or iteration == cfg.iterations:
                    self._save(f'checkpoint_{iteration:06d}.pt', iteration)
                    self._save('latest.pt', iteration)
            self._finish()
