"""广义坐标状态转换、子步积分和梯度传播回归测试。"""
import unittest
import torch
from test_supermimic_world_model import state, wm


@unittest.skipUnless(hasattr(wm.WorldModel, 'DYNAMICS_MODE'), '等待 source 完成广义坐标动力学修改')
class GenCoordTests(unittest.TestCase):
    def setUp(self):
        self.spec = state.StateSpec(31, 29)
        self.rep = wm.GenCoordRepresentation(self.spec, .02, 10)
        self.full = torch.zeros(2, self.spec.dim)
        self.full[:, 6] = 1.
        self.full[:, 2] = .65

    def test_roundtrip_and_history_preserved(self):
        torch.manual_seed(11)
        self.full[:, 7:] = torch.randn_like(self.full[:, 7:])
        fields = state.unpack_state(self.full, self.spec)
        result = self.rep.reconstruct_full_state(self.rep.extract_aux_state(self.full),
            self.rep.represent(self.full), fields['previous_action'], fields['previous_target'])
        torch.testing.assert_close(result, self.full)
        self.assertEqual(self.rep.rep_state_dim, 72)
        self.assertEqual(self.rep.prediction_dim, 37)

    def test_constant_velocity_moves_even_with_zero_prediction(self):
        self.full[:, 7] = 1.
        fields = state.unpack_state(self.full, self.spec)
        fields['joint_velocity'][:] = 2.
        aux, rep = self.rep.apply_prediction(self.rep.extract_aux_state(self.full),
            self.rep.represent(self.full), torch.zeros(2, 37))
        out = state.unpack_state(self.rep.reconstruct_full_state(aux, rep,
            fields['previous_action'], fields['previous_target']), self.spec)
        torch.testing.assert_close(out['root_pose'][:, 0], torch.full((2,), .02))
        torch.testing.assert_close(out['joint_position'], torch.full((2, 31), .04))
        torch.testing.assert_close(out['root_velocity'], fields['root_velocity'])

    def test_substep_acceleration_integration_and_action_gradient(self):
        delta = torch.ones(2, 37, requires_grad=True)
        aux, rep = self.rep.apply_prediction(self.rep.extract_aux_state(self.full),
            self.rep.represent(self.full), delta)
        # 对线速度增量，位移是 dt * (N+1)/(2N)，并非 dt。
        torch.testing.assert_close(aux[:, :3] - self.full[:, :3], torch.full((2, 3), .011))
        parts = self.rep.decompose(rep)
        torch.testing.assert_close(parts[-2], torch.full((2, 31), .011))
        torch.testing.assert_close(parts[-1], torch.ones(2, 31))
        aux[:, 2].sum().backward()
        torch.testing.assert_close(delta.grad[:, 2], torch.full((2,), .011))

    def test_local_linear_increment_rotates_to_world(self):
        self.full[:, 3:7] = wm.rotation_increment(torch.tensor([[0., 0., torch.pi/2]])).expand(2, -1)
        delta = torch.zeros(2, 37); delta[:, 0] = 1.
        aux, rep = self.rep.apply_prediction(self.rep.extract_aux_state(self.full),
            self.rep.represent(self.full), delta)
        f = state.unpack_state(self.full, self.spec)
        out = state.unpack_state(self.rep.reconstruct_full_state(aux, rep,
            f['previous_action'], f['previous_target']), self.spec)
        torch.testing.assert_close(out['root_velocity'][:, :3],
            torch.tensor([[0., 1., 0.]]).expand(2, -1), atol=1e-6, rtol=0)
        torch.testing.assert_close(aux[:, 1], torch.full((2,), .011))

    def test_angular_integration_and_new_frame_velocity(self):
        self.full[:, 7] = 1.; self.full[:, 12] = 1.
        aux, rep = self.rep.apply_prediction(self.rep.extract_aux_state(self.full),
            self.rep.represent(self.full), torch.zeros(2, 37))
        expected = wm.rotation_increment(torch.tensor([[0., 0., .02]])).expand(2, -1)
        torch.testing.assert_close(aux[:, 3:], expected)
        f = state.unpack_state(self.full, self.spec)
        out = state.unpack_state(self.rep.reconstruct_full_state(aux, rep,
            f['previous_action'], f['previous_target']), self.spec)
        torch.testing.assert_close(out['root_velocity'], f['root_velocity'])

    def test_invalid_timing_rejected(self):
        for dt, decimation in [(0., 10), (float('nan'), 10), (.02, 0), (.02, True)]:
            with self.assertRaises(ValueError):
                wm.GenCoordRepresentation(self.spec, dt, decimation)


if __name__ == '__main__':
    unittest.main()
