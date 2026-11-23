"""动作边界必须反映标定后的关节范围，而不是固定裁成 [-1,1]。"""
import importlib.util
from pathlib import Path
import unittest
import torch

path = Path(__file__).resolve().parents[1] / 'supermimic/getup_data.py'
spec = importlib.util.spec_from_file_location('_action_bounds_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ActionBoundsTests(unittest.TestCase):
    def calibration(self):
        return dict(offset=torch.tensor([.5, -.25]), scale=torch.tensor([.25, -.5]),
                    lower=torch.tensor([-1., -2.]), upper=torch.tensor([2., 1.]))

    def test_positive_and_negative_scales_cover_joint_limits(self):
        cfg = self.calibration()
        lower, upper = module.action_bounds_from_calibration(cfg)
        torch.testing.assert_close(lower, torch.tensor([-6., -2.5]))
        torch.testing.assert_close(upper, torch.tensor([6., 3.5]))
        ends = torch.stack((lower, upper)) * cfg['scale'] + cfg['offset']
        torch.testing.assert_close(ends.amin(0), cfg['lower'])
        torch.testing.assert_close(ends.amax(0), cfg['upper'])

    def test_invalid_calibration_rejected(self):
        for change in (dict(scale=torch.zeros(2)), dict(offset=torch.ones(3)),
                       dict(scale=torch.tensor([float('nan'), 1.])),
                       dict(offset=torch.tensor([2., -.25]))):
            with self.assertRaises(ValueError):
                module.action_bounds_from_calibration(dict(self.calibration(), **change))

    def test_actor_output_bounded_zero_preserving_and_differentiable(self):
        from test_supermimic_world_model import load
        networks = load('actor_critic')
        lower, upper = module.action_bounds_from_calibration(self.calibration())
        actor = networks.GoalConditionedActor(4, 0, 2, action_lower=lower, action_upper=upper)
        observation, goals = torch.zeros(3, 4), torch.empty(3, 0)
        for bias in (0., 1000., -1000.):
            with torch.no_grad():
                actor.network[-1].weight.zero_()
                actor.network[-1].bias.fill_(bias)
            output = actor(observation, goals)
            self.assertTrue(bool(((output >= lower) & (output <= upper)).all()))
            if bias == 0.:
                torch.testing.assert_close(output, torch.zeros_like(output))
        with torch.no_grad():
            actor.network[-1].bias.copy_(torch.tensor([.2, -.2]))
        actor(observation, goals).sum().backward()
        self.assertTrue(bool(torch.isfinite(actor.network[-1].bias.grad).all()))
        self.assertTrue(bool((actor.network[-1].bias.grad > 0).all()))


if __name__ == '__main__':
    unittest.main()
