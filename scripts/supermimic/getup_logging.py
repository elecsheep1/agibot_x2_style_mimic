"""实验文件和 TensorBoard 输出；不负责采样或网络更新。"""
import json
import math
import textwrap
import time
import unicodedata

import torch


class ConsoleTable:
    """固定列宽的终端表；表头只在第一次输出时打印。

    每行都带表名前缀，训练行和评估行交错时也能直接分辨。
    """

    def __init__(self, title, columns, widths):
        if len(columns) != len(widths):
            raise ValueError('表格列名与列宽数量不一致')
        self.title = title
        self.columns = tuple(columns)
        self.widths = tuple(widths)
        self.name_width = max(len(title) + 2, 8)
        self._started = False

    def row(self, values):
        if len(values) != len(self.columns):
            raise ValueError('表格行列数量不一致')
        if not self._started:
            print(self._line(self.columns), flush=True)
            self._started = True
        print(self._line(values), flush=True)

    def _line(self, cells):
        body = '  '.join(
            str(cell).rjust(width)
            for cell, width in zip(cells, self.widths)
        )
        return f'{self.title:<{self.name_width}}{body}'


def display_width(value):
    """兼容中文和全角字符的终端列宽。"""
    return sum(2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
               for char in str(value))


def duration_text(seconds):
    """小时不按 24 取模，长训超过一天仍可正确显示。"""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('显示时间必须为有限非负数')
    total = int(seconds)
    return f'{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}'


class IsaacLabConsole:
    """沿用 RSL-RL 的分块终端布局；不写 TensorBoard，不参与算法更新。"""

    def __init__(self, run_name, width=100, pad=45):
        if width < 60 or not 10 <= pad <= width - 12:
            raise ValueError('终端面板宽度或标签宽度不正确')
        self.run_name = run_name
        self.width, self.pad = width, pad
        self.started_at = time.perf_counter()
        self.training_seconds = 0.
        self.completed_iterations = 0
        self.latest_evaluation = None

    def _row(self, name, value):
        label = f'{name}:'
        prefix = ' ' * max(0, self.pad - display_width(label)) + label + ' '
        # 路径或运行名称很长时使用续行，不挤乱下一项指标。
        lines = textwrap.wrap(str(value), width=max(12, self.width - self.pad - 1),
                              break_long_words=True, break_on_hyphens=False) or ['']
        return '\n'.join([prefix + lines[0]] + [' ' * (self.pad + 1) + line for line in lines[1:]])

    def format_panel(self, title, rows, footer=()):
        padding = max(0, self.width - display_width(title))
        heading = ' ' * (padding // 2) + title
        output = ['#' * self.width, heading, '']
        output.extend(self._row(name, value) for name, value in rows)
        output.append('-' * self.width)
        output.extend(self._row(name, value) for name, value in footer)
        return '\n'.join(output) + '\n'

    def panel(self, title, rows, footer=()):
        print(self.format_panel(title, rows, footer), flush=True)

    def startup(self, cfg, metadata):
        self.panel('SuperMimic + AMP training', [
            ('Run name', self.run_name), ('Output directory', cfg.output),
            ('Initialization', 'resume' if cfg.resume is not None else metadata['initialization_mode']),
            ('Environments', cfg.num_envs), ('Learning iterations', cfg.iterations),
            ('World model batch size', cfg.world_batch_size), ('Policy batch size', cfg.policy_batch_size),
            ('AMP batch size', cfg.amp_batch_size),
            ('World model window', f'{list(cfg.world_horizons)} frames'),
            ('Imagination horizon', f'{cfg.imagination_horizon} steps'),
        ])

    def iteration(self, iteration, total, metrics, seconds, total_steps, action_noise, emit=True):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('迭代耗时必须为有限非负数')
        self.training_seconds += seconds
        self.completed_iterations += 1
        if not emit:
            return
        rows = [('Run name', self.run_name), ('Total steps', f'{int(total_steps):,}'),
                ('Steps per second', f'{metrics.get("collection/transitions", 0) / max(seconds, 1e-9):.0f}')]
        labels = (
            ('world_loss', 'Mean world model loss'),
            ('actor_loss', 'Mean actor loss'),
            ('critic_loss', 'Mean critic loss'),
            ('amp/loss', 'Mean AMP discriminator loss'),
            ('amp/reference_score', 'Mean AMP reference score'),
            ('amp/policy_score', 'Mean AMP policy score'),
            ('imagined_task_reward', 'Mean imagined task reward'),
            ('reward/style', 'Mean weighted style reward'),
            ('imagined_reward', 'Mean imagined total reward'),
            ('terminal_value', 'Mean terminal value'),
            ('collection/action_rms', 'Mean sampled action RMS'),
            ('actor_gradient_norm', 'Actor grad norm before clipping'),
        )
        rows += [(label, f'{metrics[key]:.6f}') for key, label in labels if key in metrics]
        rows.append(('Exploration noise std', f'{action_noise:.3f}'))
        for key, value in metrics.items():
            if key.startswith('reward/') and key != 'reward/style':
                rows.append((f'Imagined_Reward/{key[7:]}', f'{value:+.6f}'))
        if self.latest_evaluation is not None:
            evaluated_at, evaluation = self.latest_evaluation
            rows.append(('Last real evaluation iteration', evaluated_at))
            for key, label, pattern in (
                ('task_success_rate', 'Eval/success_rate', '.2%'),
                ('task_stable_at_end_rate', 'Eval/stable_at_end_rate', '.2%'),
                ('task_max_hold_seconds', 'Eval/max_hold_seconds', '.3f'),
                ('standing_quality_mean', 'Eval/standing_quality', '.6f'),
            ):
                if key in evaluation:
                    rows.append((label, format(evaluation[key], pattern)))
        elapsed = time.perf_counter() - self.started_at
        eta = self.training_seconds / self.completed_iterations * max(0, total - iteration)
        self.panel(f'Learning iteration {iteration}/{total}', rows, [
            ('Iteration time', f'{seconds:.2f}s'), ('Time elapsed', duration_text(elapsed)),
            ('ETA (training)', duration_text(eta)),
        ])

    def evaluation(self, iteration, total, metrics, seconds):
        self.latest_evaluation = (iteration, dict(metrics))
        rows = [('Run name', self.run_name)]
        for key, label, pattern in (
            ('standing_quality_mean', 'Standing quality', '.6f'),
            ('task_success_rate', 'Task success rate', '.2%'),
            ('task_stable_at_end_rate', 'Stable at episode end', '.2%'),
            ('task_max_hold_seconds', 'Longest stable hold (mean)', '.3f'),
            ('task_final_hold_seconds', 'Final stable hold (mean)', '.3f'),
            ('real_task_return', 'Real task return', '+.6f'),
            ('recomputed_task_return', 'Recomputed task return', '+.6f'),
            ('task_return_gap', 'Task return gap', '+.6f'),
        ):
            if key in metrics:
                rows.append((label, format(metrics[key], pattern)))
        for group, label in (('reference_initial', 'Reference initial'), ('reference_rsi', 'Reference RSI')):
            key = group + '/task_success_rate'
            if key in metrics:
                rows.append((f'{label} success rate', f'{metrics[key]:.2%}'))
        self.panel(f'Real evaluation {iteration}/{total}', rows,
                   [('Evaluation time', f'{seconds:.2f}s')])

    def warmup(self, update, total, metrics, seconds):
        rows = [('Run name', self.run_name), ('World model loss', f'{metrics["world_loss"]:.6f}')]
        if 'world_grad_norm' in metrics:
            rows.append(('World grad norm before clipping', f'{metrics["world_grad_norm"]:.6f}'))
        self.panel(f'World model warmup {update}/{total}', rows,
                   [('Time elapsed', duration_text(seconds))])


class GetupLogger:
    def __init__(self, cfg, metadata, cohort):
        from torch.utils.tensorboard import SummaryWriter

        self.output = cfg.output
        self.output.mkdir(parents=True, exist_ok=False)
        self.write_json('arguments.json', cfg.to_dict())
        self.write_json('metadata.json', metadata)
        torch.save(cohort, self.output / 'evaluation_cohort.pt')
        self.writer = SummaryWriter(str(self.output / 'tensorboard'))

    def write_json(self, name, value):
        (self.output / name).write_text(
            json.dumps(value, indent=2, default=str, ensure_ascii=False, allow_nan=False),
            encoding='utf-8',
        )

    def record(self, phase, step, metrics):
        values = {key: float(value) for key, value in metrics.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError(f'{phase} 指标包含 NaN/Inf')
        row = {'kind': phase, 'iteration': step, 'metrics': values}
        with (self.output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        prefix = {'train': 'train', 'evaluation': 'eval', 'warmup': 'warmup'}[phase]
        for key, value in values.items():
            self.writer.add_scalar(f'{prefix}/{key}', value, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
