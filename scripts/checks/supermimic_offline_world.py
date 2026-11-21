"""World-model-only access to logged expert-policy transitions, preserving episode splits."""
import json
from pathlib import Path

import torch

from fit_supermimic_smoke import valid_windows, sample


class OfflineWorldReplay:
    def __init__(self, data_path, report_path, expected_metadata, limiter_values, device):
        self.device = device
        self.data = torch.load(data_path, map_location='cpu', weights_only=False)
        report = json.loads(Path(report_path).read_text())
        if Path(report['data']).resolve() != Path(data_path).resolve():
            raise ValueError('Offline split report refers to a different dataset')
        metadata = self.data['metadata']
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(f'Offline dataset incompatible: {name}')
        for name, current in limiter_values.items():
            torch.testing.assert_close(self.data['limiter'][name].cpu(), current.detach().cpu(), atol=1e-6, rtol=1e-6)
        self.train_keys = set(map(tuple, report['train_episodes']))
        self.holdout_keys = set(map(tuple, report['validation_episodes']))
        actual = {(env, int(episode)) for env in range(self.data['episode_ids'].shape[1])
                  for episode in self.data['episode_ids'][:, env].unique()}
        if not self.train_keys or not self.holdout_keys or self.train_keys & self.holdout_keys or self.train_keys | self.holdout_keys != actual:
            raise ValueError('Train/holdout keys must disjointly partition all episodes')
        for name in ('state', 'next_state', 'action', 'contexts'):
            if not bool(torch.isfinite(self.data[name]).all()):
                raise ValueError(f'Nonfinite offline {name}')
        self.candidates = {}

    def sample_windows(self, batch_size, horizon):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if horizon not in self.candidates:
            candidates = valid_windows(self.data, horizon)
            keep = torch.tensor([(int(env), int(self.data['episode_ids'][start, env])) in self.train_keys
                                 for start, env in candidates], dtype=torch.bool)
            self.candidates[horizon] = candidates[keep]
            if not len(self.candidates[horizon]):
                raise ValueError(f'No training-only windows for horizon {horizon}')
        return sample(self.data, self.candidates[horizon], batch_size, horizon, self.device)
