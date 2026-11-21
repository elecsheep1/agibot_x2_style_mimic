"""角速度积分对照必须保持世界坐标系左乘约定。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from test_supermimic_scripts import load_script, fitter

checks = str(Path(__file__).resolve().parents[1] / 'checks')
with patch.object(sys, 'path', [checks] + sys.path):
    trace = load_script('trace_supermimic_failure_episode')


class FailureTraceTests(unittest.TestCase):
    def test_world_angular_velocity_left_multiply(self):
        _, dynamics = fitter.load_modules()
        states = torch.zeros(1, 5, 13)
        initial = dynamics.rotation_increment(torch.tensor([[.7, 0., 0.]]))
        states[:, 0, 3:7] = initial
        states[:, :, 12] = 1.
        predicted = trace.integrate_angular_velocity(states, .02, dynamics)
        expected = dynamics.quat_mul(dynamics.rotation_increment(torch.tensor([[0., 0., .08]])), initial)
        torch.testing.assert_close(predicted[:, -1], expected, atol=1e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()
