"""起身实验调度：组装现有算法，执行预热、联合训练、评估和保存。"""
import ast
import hashlib
from pathlib import Path

import torch

from getup_checkpointing import plot_metrics, restore_rng, save_checkpoint
from getup_config import PROJECT
from getup_evaluation import build_cohort, evaluate
from getup_logging import ConsoleTable, GetupLogger
from getup_warmup import warmup_world_model


def file_hashes(paths):
    return {
        str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in sorted(paths, key=str)
    }


def semantic_file_hashes(paths):
    """忽略注释及格式；保留代码、常量和文档字符串变化。"""
    return {
        str(Path(path).resolve()): hashlib.sha256(
            ast.dump(ast.parse(Path(path).read_text()), include_attributes=False).encode()
        ).hexdigest()
        for path in sorted(paths, key=str)
    }


class GetupTrainingRun:
    """实验层 runner；网络更新仍由 source 中已有 runner/trainer 完成。"""

    def __init__(self, runtime, cfg, checkpoint, source_before, source_semantic_before=None):
        self.runtime = runtime
        self.cfg = cfg
        self.checkpoint = checkpoint
        self.start = int(checkpoint['iteration']) if checkpoint else 0
        if self.start >= cfg.iterations:
            raise ValueError('目标总轮数必须大于保存的轮数')
        self.source_before = source_before
        self.source_semantic_before = source_semantic_before
        self.cohort = build_cohort(runtime, cfg.eval_seed)
        self.metadata = self._metadata()
        self.logger = None
        self.train_table = ConsoleTable(
            'train',
            ('iter', 'world_loss', 'imagined', 'act_rms', 'style', 'progress'),
            (6, 12, 12, 10, 9, 9),
        )
        self.eval_table = ConsoleTable(
            'eval',
            ('iter', 'standing', 'progress', 'success', 'max_hold', 'real_ret', 'recomputed'),
            (6, 10, 10, 9, 10, 12, 12),
        )
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
            source_semantic_sha256=self.source_semantic_before,
            evaluation_seed=cfg.eval_seed,
            evaluation_seconds=cfg.eval_seconds,
            training_schedule={
                key: getattr(cfg, key) for key in (
                    'collection_steps', 'world_batch_size', 'policy_batch_size', 'amp_batch_size',
                    'world_updates', 'discriminator_updates',
                )
            },
        )

    def _check_resume(self):
        saved = self.checkpoint
        for key in (
            'input_sha256', 'action_noise', 'world_horizons',
            'offline_world_fraction', 'evaluation_seed', 'evaluation_seconds', 'training_schedule',
        ):
            if saved['metadata'][key] != self.metadata[key]:
                raise ValueError(f'恢复训练的配置或输入已改变：{key}')
        saved_semantics = saved['metadata'].get('source_semantic_sha256')
        if saved_semantics is not None and self.source_semantic_before is not None:
            if saved_semantics != self.source_semantic_before:
                raise ValueError('恢复训练的 source 代码逻辑已改变')
        elif saved['metadata']['source_code_sha256'] != self.source_before:
            raise ValueError('旧检查点没有语义摘要，无法确认 source 变更是否仅为注释；拒绝直接恢复')
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
        print(f'eval  {iteration}  开始，每环境最多 {self.cfg.eval_seconds:g} 秒', flush=True)
        report = evaluate(self.runtime, self.cohort, self.cfg.eval_seconds, self.cfg.eval_seed)
        self.logger.write_json(f'evaluation_{iteration:06d}.json', report)
        self.logger.record('evaluation', iteration, report['metrics'])
        self.logger.flush()
        metrics = report['metrics']
        progress = metrics.get('proxy_terms/progress')
        self.eval_table.row((
            iteration,
            f"{metrics['standing_quality_mean']:.4f}",
            f'{progress:+.4f}' if progress is not None else '-',
            f"{metrics['task_success_rate']:.1%}",
            f"{metrics['task_max_hold_seconds']:.2f}s",
            f"{metrics['real_task_return']:+.4f}",
            f"{metrics['recomputed_task_return']:+.4f}",
        ))

    def _train_iteration(self, iteration):
        cfg = self.cfg
        horizon = cfg.world_horizons[(iteration - 1) % len(cfg.world_horizons)]
        metrics = self.runtime.runner.train_iteration(
            self.runtime.reference,
            collection_steps=cfg.collection_steps,
            world_batch_size=cfg.world_batch_size,
            policy_batch_size=cfg.policy_batch_size,
            amp_batch_size=cfg.amp_batch_size,
            world_window_frames=horizon,
            world_updates=cfg.world_updates,
            discriminator_updates=cfg.discriminator_updates,
            offline_world_replay=self.runtime.offline,
            offline_world_fraction=cfg.offline_world_fraction,
        )
        metrics['world_window_frames'] = horizon
        metrics['world_transition_steps'] = horizon - 1
        metrics['imagination_horizon'] = cfg.imagination_horizon
        metrics['imagined_task_reward'] = metrics['imagined_reward'] - metrics['reward/style']
        self.logger.record('train', iteration, metrics)
        if iteration % cfg.log_interval == 0 or iteration == self.start + 1:
            progress = metrics.get('reward/progress')
            self.train_table.row((
                iteration,
                f"{metrics['world_loss']:.6f}",
                f"{metrics['imagined_task_reward']:+.6f}",
                f"{metrics['collection/action_rms']:.3f}",
                f"{metrics['reward/style']:+.4f}",
                f'{progress:+.4f}' if progress is not None else '-',
            ))
            self.logger.flush()

    def _finish(self):
        after = file_hashes((PROJECT / 'source').rglob('*.py'))
        byte_identical = after == self.source_before
        semantics_identical = byte_identical
        if not byte_identical:
            changed = sorted(name for name in set(after) | set(self.source_before)
                             if after.get(name) != self.source_before.get(name))
            semantic_error = None
            try:
                semantics_identical = (self.source_semantic_before is not None and
                    semantic_file_hashes((PROJECT / 'source').rglob('*.py')) == self.source_semantic_before)
            except (SyntaxError, OSError) as exc:
                semantic_error = str(exc)
            self.logger.write_json('source_changes.json', dict(
                files=changed, semantics_unchanged=semantics_identical, error=semantic_error))
            self.logger.flush()
            if not semantics_identical:
                raise RuntimeError(f'运行期间 source 代码逻辑发生变化，检查点已保存；详见 source_changes.json：{changed}')
            print('[源码检查] 仅注释或格式变化，代码语义未变；已记录 source_changes.json', flush=True)
        self.logger.write_json('summary.json', dict(
            completed_iterations=self.cfg.iterations,
            formal_source_unchanged=byte_identical,
            formal_source_semantics_unchanged=semantics_identical,
            total_training_transitions=self.runtime.runner.total_env_steps * self.runtime.env.num_envs,
            replay_steps=self.runtime.replay.total_steps,
            output=str(self.cfg.output),
            evaluation_note='几何站稳不包含脚接触检查，不等同于原起身任务的完整成功率。',
        ))
        self.logger.flush()
        print(f'[完成] {self.cfg.output}', flush=True)

    def run(self):
        cfg, runtime = self.cfg, self.runtime
        print('SuperMimic 起身训练', flush=True)
        print(f'  output   {cfg.output}', flush=True)
        print(
            f'  init     {runtime.metadata["initialization_mode"]}'
            f'    envs {cfg.num_envs}'
            f'    window {list(cfg.world_horizons)} frames'
            f' / {[frames - 1 for frames in cfg.world_horizons]} transitions'
            f'    imagine {cfg.imagination_horizon} steps'
            f'    iters {cfg.iterations}',
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
