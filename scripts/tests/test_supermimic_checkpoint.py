"""Pretrained dynamics initialization must reject incompatible robot semantics."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from test_supermimic_scripts import load_script

loader = load_script('supermimic_checkpoint')


class WorldCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.metadata = dict(joint_names=['left', 'right'], step_dt=.02)
        self.limiter = dict(offset=torch.zeros(2), scale=torch.ones(2))
        self.payload = dict(metadata=self.metadata, hidden_dims=(16, 16), context_dim=3,
                            limiter=self.limiter, model={'stats_ready': torch.tensor(True),
                            **{f'limiter.{k}': v.clone() for k, v in self.limiter.items()}})

    def read(self, payload, metadata=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(payload, path)
            return loader.read_world_checkpoint(path, self.metadata if metadata is None else metadata,
                                                 self.limiter, 3, 'cpu')

    def test_valid_checkpoint(self):
        self.assertEqual(self.read(self.payload)['hidden_dims'], (16, 16))

    def test_joint_order_and_timing_rejected(self):
        for change in (dict(joint_names=['right', 'left']), dict(step_dt=.01)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read(self.payload, {**self.metadata, **change})

    def test_duplicate_limiter_disagreement_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload['model']['limiter.scale'][0] = .5
        with self.assertRaises(AssertionError):
            self.read(payload)

    def test_unfitted_and_nonfinite_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload['model']['stats_ready'] = torch.tensor(False)
        with self.assertRaises(ValueError):
            self.read(payload)
        payload = copy.deepcopy(self.payload)
        payload['model']['weight'] = torch.tensor(float('nan'))
        with self.assertRaises(ValueError):
            self.read(payload)

    def test_full_training_checkpoint_is_not_world_initialization(self):
        with self.assertRaises(ValueError):
            self.read(dict(modules={'world_model': self.payload['model']}))


if __name__ == '__main__':
    unittest.main()
