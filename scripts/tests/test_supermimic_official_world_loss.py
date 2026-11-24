"""官方广义坐标损失的通道权重、旋转误差和监督梯度回归。"""
import inspect
import unittest
import torch
from test_supermimic_world_model import state, wm


@unittest.skipUnless('nominal_error_root_pose_pos' in inspect.signature(wm.dynamics_loss).parameters,
                     '等待 source 对齐官方损失')
class OfficialWorldLossTests(unittest.TestCase):
    def setUp(self):
        self.spec = state.StateSpec(31, 29)
        self.target = torch.zeros(2, 3, self.spec.dim, dtype=torch.float64)
        self.target[:, :, 6] = 1.
        self.width = 3 + 9 + 2 * self.spec.num_joints

    def test_single_position_channel_weight_and_scale_override(self):
        predicted = self.target.clone(); predicted[:, 1:, 0] = .01
        loss, terms = wm.dynamics_loss(predicted, self.target, self.spec)
        torch.testing.assert_close(loss, torch.tensor(1 / self.width, dtype=loss.dtype))
        torch.testing.assert_close(terms['root_pos'], torch.tensor(.01**2/3, dtype=loss.dtype))
        changed, _ = wm.dynamics_loss(predicted, self.target, self.spec, nominal_error_root_pose_pos=.02)
        torch.testing.assert_close(changed, loss/4)

    def test_rotation_matches_exact_chordal_error_and_quaternion_sign(self):
        angle = .4
        predicted = self.target.clone()
        predicted[:, 1:, 3:7] = wm.rotation_increment(torch.tensor([[0., 0., angle]], dtype=torch.float64))
        loss, _ = wm.dynamics_loss(predicted, self.target, self.spec)
        expected = 2 * (1 - torch.cos(torch.tensor(angle, dtype=loss.dtype))) / .02**2 / self.width
        torch.testing.assert_close(loss, expected)
        predicted[:, :, 3:7] *= -1
        flipped, _ = wm.dynamics_loss(predicted, self.target, self.spec)
        torch.testing.assert_close(flipped, loss)

    def test_joint_velocity_channels_have_correct_weight(self):
        predicted = self.target.clone()
        start = 13 + self.spec.num_joints
        predicted[:, 1:, start:start + self.spec.num_joints] = 3.14
        loss, terms = wm.dynamics_loss(predicted, self.target, self.spec)
        torch.testing.assert_close(loss, torch.tensor(31/self.width, dtype=loss.dtype))
        self.assertEqual(set(terms), {'root_pos','root_ori','joint_pos','joint_vel'})

    def test_initial_state_ignored_and_targets_detached(self):
        predicted = self.target.clone(); predicted[:, 0] = 100
        predicted[:, 1:, 0] = .01
        predicted.requires_grad_(); target = self.target.clone().requires_grad_()
        loss, _ = wm.dynamics_loss(predicted, target, self.spec)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertEqual(float(predicted.grad[:, 0].abs().sum()), 0.)

    def test_invalid_scales_rejected(self):
        for value in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                wm.dynamics_loss(self.target, self.target, self.spec, nominal_error_joint_pos=value)


if __name__ == '__main__':
    unittest.main()
