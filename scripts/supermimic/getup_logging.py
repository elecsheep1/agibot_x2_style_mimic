"""实验文件和 TensorBoard 输出；不负责采样或网络更新。"""
import json
import math

import torch


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
