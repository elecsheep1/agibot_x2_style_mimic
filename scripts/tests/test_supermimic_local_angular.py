"""最近邻诊断必须排除查询自身的完整回合。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from test_supermimic_scripts import load_script

checks = str(Path(__file__).resolve().parents[1] / 'checks')
with patch.object(sys, 'path', [checks] + sys.path):
    audit = load_script('check_supermimic_local_angular_fit')


class LocalAngularTests(unittest.TestCase):
    def test_exclude_whole_episode_across_chunks(self):
        bank = torch.tensor([[0., 0.], [.1, .1], [1., 1.], [1.1, 1.1]])
        keys = torch.tensor([[0, 0], [0, 0], [0, 1], [0, 2]])
        distance, indexes = audit.nearest_rows(torch.zeros(1, 2), bank, keys, torch.tensor([[0, 0]]), k=2, chunk_size=1)
        self.assertEqual(indexes.tolist(), [[2, 3]])
        torch.testing.assert_close(distance, torch.tensor([[1., 1.1]]), atol=1e-6, rtol=0)

    def test_insufficient_cross_episode_neighbors_rejected(self):
        with self.assertRaises(ValueError):
            audit.nearest_rows(torch.zeros(1, 2), torch.zeros(2, 2),
                               torch.tensor([[0, 0], [0, 0]]), torch.tensor([[0, 0]]), k=1)


if __name__ == '__main__':
    unittest.main()
