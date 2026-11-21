"""分层标签、配额和真实后继状态的回归检查。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from test_supermimic_scripts import load_script

checks = str(Path(__file__).resolve().parents[1] / 'checks')
with patch.object(sys, 'path', [checks] + sys.path):
    phases = load_script('supermimic_phase_sampling')


class PhaseSamplingTests(unittest.TestCase):
    def test_height_change_labels(self):
        heights = torch.tensor([[.1, .1, .1], [.1, .2, .4], [.4, .4, .4],
                                [.6, .4, .1], [.6, .6, .6]])
        self.assertEqual(phases.labels_from_heights(heights).tolist(), [0, 1, 2, 3, 4])
        with self.assertRaises(ValueError):
            phases.labels_from_heights(torch.tensor([[0., float('nan')]]))

    def test_balanced_quotas_and_missing_groups(self):
        labels = torch.tensor([0] * 100 + [1] + [2] * 3 + [3] * 2 + [4])
        chosen = phases.choose_indexes(labels, 20, True)
        self.assertEqual(torch.bincount(labels[chosen]).tolist(), [4] * 5)
        labels = torch.tensor([0, 0, 1])
        chosen = phases.choose_indexes(labels, 12, True)
        self.assertEqual(torch.bincount(labels[chosen]).tolist(), [6, 6])
        chosen = phases.choose_indexes(labels, 5, True)
        counts = torch.bincount(labels[chosen])
        self.assertEqual(int(counts.sum()), 5)
        self.assertEqual(int(counts.max() - counts.min()), 1)

    def test_terminal_next_state_not_reset_state(self):
        data = dict(state=torch.tensor([0., 1., -50.]).reshape(3, 1, 1),
                    next_state=torch.tensor([1., 9., -49.]).reshape(3, 1, 1),
                    action=torch.tensor([2., 3., 4.]).reshape(3, 1, 1), contexts=torch.ones(1, 2))
        batch = phases.gather_windows(data, torch.tensor([[0, 0]]), 2, 'cpu')
        self.assertEqual(batch['states'].flatten().tolist(), [0., 1., 9.])
        self.assertEqual(batch['actions'].flatten().tolist(), [2., 3.])


if __name__ == '__main__':
    unittest.main()
