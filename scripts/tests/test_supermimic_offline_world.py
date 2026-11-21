"""Offline dynamics data must never mix held-out episodes into training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_supermimic_scripts import load_script, fitter

with patch.dict(sys.modules, {'fit_supermimic_smoke': fitter}):
    offline = load_script('supermimic_offline_world')


class OfflineWorldTests(unittest.TestCase):
    def fixture(self, directory):
        path = Path(directory) / 'transitions.pt'
        data = dict(state=torch.tensor([0., 1., 2., 100., 101., 102.]).reshape(6, 1, 1),
                    action=torch.zeros(6, 1, 1), contexts=torch.ones(1, 2),
                    episode_ids=torch.tensor([0, 0, 0, 1, 1, 1]).reshape(6, 1),
                    step_ids=torch.tensor([0, 1, 2, 0, 1, 2]).reshape(6, 1),
                    terminated=torch.zeros(6, 1, dtype=torch.bool),
                    truncated=torch.tensor([False, False, True, False, False, True]).reshape(6, 1),
                    metadata={'step_dt': .02}, limiter={'offset': torch.zeros(1)})
        data['next_state'] = data['state'] + 1
        torch.save(data, path)
        report = dict(data=str(path), train_episodes=[[0, 0]], validation_episodes=[[0, 1]])
        report_path = Path(directory) / 'report.json'
        report_path.write_text(json.dumps(report))
        return path, report_path, report

    def test_only_train_windows_and_terminal_next_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report, _ = self.fixture(directory)
            replay = offline.OfflineWorldReplay(path, report, {'step_dt': .02}, {'offset': torch.zeros(1)}, 'cpu')
            batch = replay.sample_windows(32, 2)
            self.assertEqual(batch['states'].shape, (32, 3, 1))
            self.assertLess(float(batch['states'].max()), 100)
            self.assertTrue(bool((batch['states'][:, 1:] == batch['states'][:, :-1] + 1).all()))

    def test_overlapping_split_and_wrong_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report_path, report = self.fixture(directory)
            with self.assertRaises(ValueError):
                offline.OfflineWorldReplay(path, report_path, {'step_dt': .01}, {'offset': torch.zeros(1)}, 'cpu')
            report['validation_episodes'] = [[0, 0], [0, 1]]
            report_path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                offline.OfflineWorldReplay(path, report_path, {'step_dt': .02}, {'offset': torch.zeros(1)}, 'cpu')


if __name__ == '__main__':
    unittest.main()
