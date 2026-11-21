"""快速窗口排名必须使用真实转移的角速度变化，包含正确的末端状态。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from test_supermimic_scripts import load_script

checks = str(Path(__file__).resolve().parents[1] / 'checks')
with patch.object(sys, 'path', [checks] + sys.path):
    experiment = load_script('compare_supermimic_fast_angular_sampling')


class FastAngularSamplingTests(unittest.TestCase):
    def test_rms_score_and_actual_terminal_velocity(self):
        state = torch.zeros(3, 1, 13)
        following = state.clone()
        following[0, 0, 10] = 1.
        following[1, 0, 11] = 3.
        data = dict(state=state, next_state=following)
        score = experiment.angular_change_scores(data, torch.tensor([[0, 0], [1, 0]]), horizon=2)
        torch.testing.assert_close(score, torch.tensor([5**.5, (9 / 2)**.5]))

    def test_exact_training_rank_quota(self):
        scores = torch.arange(23, dtype=torch.float32)
        ids = experiment.fast_indexes(scores, .1)
        self.assertEqual(ids.tolist(), [22, 21, 20])
        with self.assertRaises(ValueError):
            experiment.fast_indexes(torch.tensor([float('nan')]))


if __name__ == '__main__':
    unittest.main()
