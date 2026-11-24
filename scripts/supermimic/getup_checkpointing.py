"""训练脚本的完整网络／优化器保存与恢复；不保存物理场景和 replay。"""
import os
from pathlib import Path
import random

import numpy as np
import torch


def collect_optimizers(trainer, discriminator):
    """逐个保存官方分组优化器；兼容迁移前的单 Adam 入口。"""
    result = {'discriminator': discriminator.optimizer}
    for group in ('world', 'actor', 'critic'):
        grouped = getattr(trainer, group + '_optimizers', None)
        if grouped is None:
            result[group] = getattr(trainer, group + '_optimizer')
        else:
            for index, optimizer in enumerate(grouped):
                result[f'{group}/{index}'] = optimizer
    return result


def save_checkpoint(path, runtime, iteration, cohort, metadata):
    payload = dict(kind='supermimic_getup_training', iteration=iteration,
        warmup_complete=getattr(runtime, 'warmup_complete', True),
        total_env_steps=runtime.runner.total_env_steps, metadata=metadata,
        modules={name: module.state_dict() for name, module in runtime.modules.items()},
        optimizers={name: optimizer.state_dict() for name, optimizer in runtime.optimizers.items()},
        physics_context=runtime.context.detach().cpu(), cohort=cohort,
        rng=dict(torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 python=random.getstate(), numpy=np.random.get_state()),
        resume_note='恢复网络和优化器后重新采集 replay，不恢复物理场景，不能保证与不中断训练逐位一致。')
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_rng(checkpoint):
    values = checkpoint['rng']
    torch.set_rng_state(values['torch_cpu'].cpu())
    if values['torch_cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in values['torch_cuda']])
    random.setstate(values['python'])
    np.random.set_state(values['numpy'])


def plot_metrics(output):
    import json
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/supermimic_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    train = [r for r in rows if r['kind'] == 'train']
    evaluation = [r for r in rows if r['kind'] == 'evaluation']
    figure, axes = plt.subplots(3, 1, figsize=(8, 8))
    if train:
        axes[0].plot([r['iteration'] for r in train], [r['metrics']['imagined_task_reward'] for r in train], label='Imagined task reward / step')
    if evaluation:
        x = [r['iteration'] for r in evaluation]
        axes[0].plot(x, [r['metrics']['real_task_reward_mean_per_step'] for r in evaluation], marker='o', label='Real task reward / step')
        axes[1].plot(x, [r['metrics']['standing_quality_mean'] for r in evaluation], marker='o', label='Real standing quality')
        axes[2].plot(x, [r['metrics']['max_hold_seconds'] for r in evaluation], marker='o', label='Real geometric max hold (s)')
        aligned = [r for r in evaluation if 'task_max_hold_seconds' in r['metrics']]
        if aligned:
            axes[2].plot([r['iteration'] for r in aligned],
                         [r['metrics']['task_max_hold_seconds'] for r in aligned],
                         marker='x', label='Full task max hold (s)')
    for axis in axes:
        axis.set_xlabel('Training iteration')
        axis.grid(alpha=.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / 'training_curves.png', dpi=150)
    plt.close(figure)
