"""重新锚定必须发生在动作执行前；窗口抽取不能隐式重复回合。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from test_supermimic_scripts import load_script

checks = str(Path(__file__).resolve().parents[1] / 'checks')
with patch.object(sys, 'path', [checks] + sys.path):
    diagnosis = load_script('diagnose_supermimic_rising_rollouts')


class RisingDiagnosisTests(unittest.TestCase):
    def test_anchor_timing(self):
        truth = torch.tensor([0., 10., 20., 30., 40.]).reshape(1, 5, 1)
        action = torch.ones(1, 4, 1)
        model = lambda state, action, context: state + action
        for interval, expected in [(None, [0., 1., 2., 3., 4.]),
                                   (1, [0., 1., 11., 21., 31.]),
                                   (2, [0., 1., 2., 21., 22.])]:
            actual = diagnosis.anchored_rollout(model, truth, action, torch.zeros(1, 1), interval)
            self.assertEqual(actual.flatten().tolist(), expected)

    def test_episode_selection_without_replacement(self):
        data = dict(episode_ids=torch.tensor([0] * 5 + [1] * 5).reshape(10, 1))
        candidates = torch.tensor([[i, 0] for i in range(10)])
        selected, episodes = diagnosis.select_per_episode(data, candidates, maximum=2)
        self.assertEqual(episodes, 2)
        self.assertEqual(selected[:, 0].tolist(), [0, 4, 5, 9])
        self.assertEqual(len(selected.unique(dim=0)), len(selected))


if __name__ == '__main__':
    unittest.main()
