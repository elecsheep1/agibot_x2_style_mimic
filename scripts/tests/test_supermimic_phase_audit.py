"""高度统计的边界和起落方向不能混淆。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from test_supermimic_scripts import load_script

checks = str(Path(__file__).resolve().parents[1] / 'checks')
with patch.object(sys, 'path', [checks] + sys.path):
    audit = load_script('audit_supermimic_horizon_data')


class PhaseAuditTests(unittest.TestCase):
    def test_height_boundaries_partition_all_frames(self):
        states = torch.zeros(4, 3)
        states[:, 2] = torch.tensor([.1, .3, .49, .5])
        result = audit.height_summary(states)
        self.assertEqual(result['low_fraction'], .25)
        self.assertEqual(result['middle_fraction'], .5)
        self.assertEqual(result['high_fraction'], .25)

    def test_rising_and_falling_are_separate(self):
        states = torch.zeros(3, 3, 3)
        states[:, :, 2] = torch.tensor([[.1, .4, .6], [.6, .4, .1], [.1, .1, .1]])
        result = audit.window_summary(states)
        self.assertAlmostEqual(result['low_to_high_fraction'], 1 / 3)
        self.assertAlmostEqual(result['high_to_low_fraction'], 1 / 3)
        self.assertAlmostEqual(result['height_range_over_10cm_fraction'], 2 / 3)


if __name__ == '__main__':
    unittest.main()
