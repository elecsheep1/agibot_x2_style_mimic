"""总旋转误差和重力方向误差不可混同，速度残差应以真实转移为基准。"""
from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch

import torch

from test_supermimic_scripts import load_script, fitter

checks = str(Path(__file__).resolve().parents[1] / 'checks')
with patch.object(sys, 'path', [checks] + sys.path):
    audit = load_script('audit_supermimic_rising_failures')


class FailureAuditTests(unittest.TestCase):
    def test_gravity_direction_ignores_pure_world_yaw(self):
        _, dynamics = fitter.load_modules()
        q = dynamics.rotation_increment(torch.tensor([[.6, -.2, .3]]))
        yaw = dynamics.rotation_increment(torch.tensor([[0., 0., 1.2]]))
        rotated = dynamics.quat_mul(yaw, q)
        angle = audit.gravity_angle(rotated, q)
        self.assertLess(float(angle), 1e-4)
        tilted = dynamics.rotation_increment(torch.tensor([[math.pi / 2, 0., 0.]]))
        identity = torch.tensor([[0., 0., 0., 1.]])
        self.assertAlmostEqual(float(audit.gravity_angle(tilted, identity)), 90., places=4)

    def test_constant_velocity_residual(self):
        state, dynamics = fitter.load_modules()
        spec = state.StateSpec(2, 1)
        before = torch.zeros(1, 2, spec.dim)
        before[..., 6] = 1
        after = before.clone()
        b = state.unpack_state(before.reshape(-1, spec.dim), spec)
        a = state.unpack_state(after.reshape(-1, spec.dim), spec)
        b['root_velocity'][:, 0] = 1
        a['root_velocity'][:, 0] = 1
        a['root_pose'][:, 0] += .02
        b['joint_velocity'][:] = 2
        a['joint_velocity'][:] = 2
        a['joint_position'][:] += .04
        result = audit.residual_report(before, after, spec, .02, state, dynamics)
        self.assertLess(result['root_linear']['rmse'], 1e-6)
        self.assertLess(result['joint_velocity']['rmse'], 1e-6)
        self.assertLess(result['root_angular']['rmse'], 1e-6)


if __name__ == '__main__':
    unittest.main()
