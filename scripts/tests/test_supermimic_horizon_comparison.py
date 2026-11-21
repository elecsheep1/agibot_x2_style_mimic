"""检查预测长度对照的预算匹配和在线验收数据隔离。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from test_supermimic_scripts import load_script

checks = str(Path(__file__).resolve().parents[1] / 'checks')
with patch.object(sys, 'path', [checks] + sys.path):
    comparison = load_script('compare_supermimic_world_horizons')


class HorizonComparisonTests(unittest.TestCase):
    def test_both_budget_controls(self):
        schedules = comparison.training_schedules()
        self.assertEqual(sum(schedules['h8_same_transitions']), sum(schedules['h8_16_32']))
        self.assertEqual(len(schedules['h8_same_updates']), len(schedules['h8_16_32']))
        self.assertEqual(set(schedules['h8_same_transitions']), {8})
        self.assertEqual(schedules['h8_16_32'], (8, 16, 32))

    def test_whole_environment_holdout(self):
        data = dict(terminated=torch.zeros(12, 4, dtype=torch.bool),
                    truncated=torch.zeros(12, 4, dtype=torch.bool),
                    episode_ids=torch.zeros(12, 4, dtype=torch.long),
                    step_ids=torch.arange(12)[:, None].expand(-1, 4))
        train = comparison.select_env_windows(data, [0, 2, 3], horizon=4)
        holdout = comparison.select_env_windows(data, [1], horizon=4)
        self.assertEqual(set(train[:, 1].tolist()), {0, 2, 3})
        self.assertEqual(set(holdout[:, 1].tolist()), {1})
        with self.assertRaises(ValueError):
            comparison.select_env_windows(data, [8], horizon=4)


if __name__ == '__main__':
    unittest.main()
