"""CPU dynamics regressions; no simulator or project task imports required."""
import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType
import unittest

import torch
from torch import nn

SOURCE = Path(__file__).resolve().parents[2] / 'source/x2_style_mimic/x2_style_mimic/algorithms/supermimic'
ROOT_PACKAGE = "_supermimic_dynamics_tests"
PACKAGE = f"{ROOT_PACKAGE}.algorithms.supermimic"

package_paths = {
    ROOT_PACKAGE: SOURCE.parent.parent,
    f"{ROOT_PACKAGE}.algorithms": SOURCE.parent,
    PACKAGE: SOURCE,
}

for name, directory in package_paths.items():
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package


def load(name):
    spec = importlib.util.spec_from_file_location(f'{PACKAGE}.{name}', SOURCE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


state = load('state')
wm = load('world_model')


class WorldModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.spec = state.StateSpec(31, 29)
        self.initial = torch.zeros(3, self.spec.dim)
        self.initial[:, 6] = 1.
        self.actions = torch.randn(3, 8, 29) * .1
        self.context = torch.randn(3, 4)
        self.batch = dict(states=self.initial[:, None].repeat(1, 9, 1),
                          actions=self.actions, context=self.context)
        self.model = self.make_model()

    def rollout(self, actions):
        return wm.rollout_logged_actions(
            self.model, self.initial, actions, self.context,
        )

    def make_model(self):
        limiter = wm.TargetLimiter(torch.zeros(29), .25, -torch.ones(29),
                                   torch.ones(29), torch.full((29,), .1))
        timing = dict(dt=.02, decimation=10) if hasattr(wm.WorldModel, 'DYNAMICS_MODE') else {}
        return wm.WorldModel(self.spec, 4, limiter, hidden_dims=(32, 32), **timing)

    def test_normalization_required_and_finite(self):
        with self.assertRaises(RuntimeError):
            self.model(self.initial, self.actions[:, 0], self.context)
        self.model.fit_input_stats(self.batch)
        self.assertFalse(self.model.input_mean.requires_grad)
        self.assertTrue(bool(torch.isfinite(self.model.input_std).all()))
        self.assertTrue(bool((self.model.input_std >= 1e-3).all()))



    def test_limiter_matches_hard_limit_then_rate_limit(self):
        limiter = self.model.limiter
        previous = torch.tensor([.95, -3., 3.]).unsqueeze(1).expand(3, 29)
        actions = torch.tensor([-20., 20., -20.]).unsqueeze(1).expand(3, 29)
        expected = torch.tensor([.85, -.9, .9]).unsqueeze(1).expand(3, 29)
        torch.testing.assert_close(limiter(previous, actions), expected)

    def test_analytic_history_and_identity_prediction(self):
        self.model.fit_input_stats(self.batch)
        with torch.no_grad():
            for parameter in self.model.network.parameters():
                parameter.zero_()
        out = self.rollout(self.actions)
        self.assertEqual(out.shape, (3, 9, self.spec.dim))
        physical_dim = 13 + 2*self.spec.num_joints
        torch.testing.assert_close(out[:, :, :physical_dim],
                                   self.initial[:, None, :physical_dim].expand(-1, 9, -1))
        previous = torch.zeros(3, 29)
        for t in range(8):
            expected = self.model.limiter(previous, self.actions[:, t])
            parts = state.unpack_state(out[:, t+1], self.spec)
            torch.testing.assert_close(parts['previous_action'], self.actions[:, t])
            torch.testing.assert_close(parts['previous_target'], expected)
            previous = expected

    def test_final_physical_state_gradient_reaches_first_action(self):
        self.model.fit_input_stats(self.batch)
        actions = self.actions.clone().requires_grad_()
        out = self.rollout(actions)
        # Exclude analytic action/target history from the differentiated objective.
        objective = out[:, -1, :13+2*self.spec.num_joints].square().sum()
        derivative = torch.autograd.grad(objective, actions)[0]
        self.assertTrue(bool(torch.isfinite(derivative).all()))
        self.assertGreater(float(derivative[:, 0].abs().sum()), 0.)

    def test_loss_backpropagates_to_model_not_targets(self):
        self.model.fit_input_stats(self.batch)
        out = self.rollout(self.actions)
        target = self.batch['states'].clone().requires_grad_()
        loss, terms = wm.dynamics_loss(out, target, self.spec)
        expected = 4 if 'nominal_error_root_pose_pos' in inspect.signature(wm.dynamics_loss).parameters else 6
        self.assertEqual(len(terms), expected)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        grads = [p.grad for p in self.model.network.parameters()]
        self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in grads))
        self.assertTrue(any(bool((g != 0).any()) for g in grads))
        self.assertIsNone(target.grad)

    def test_rotation_sign_invariance_and_unit_norm(self):
        self.model.fit_input_stats(self.batch)
        flipped = self.initial.clone()
        flipped[:, 3:7] *= -1
        features, _ = self.model.features(self.initial, self.actions[:, 0], self.context)
        other, _ = self.model.features(flipped, self.actions[:, 0], self.context)
        torch.testing.assert_close(features, other)
        torch.testing.assert_close(wm.rotation_error(self.initial[:, 3:7], flipped[:, 3:7]),
                                   torch.zeros(3))
        out = self.rollout(self.actions)
        torch.testing.assert_close(out[:, :, 3:7].norm(dim=-1), torch.ones(3, 9))

    def test_rotation_identity_and_known_angle(self):
        vector = torch.zeros(2, 3, requires_grad=True)
        q = wm.rotation_increment(vector)
        torch.testing.assert_close(q, torch.tensor([[0., 0., 0., 1.]]).expand(2, -1))
        q.sum().backward()
        self.assertTrue(bool(torch.isfinite(vector.grad).all()))
        turned = wm.rotation_increment(torch.tensor([[0., 0., torch.pi/2]]))
        torch.testing.assert_close(wm.rotation_error(turned, q[:1].detach()),
                                   torch.tensor([torch.pi/2]))

    def test_zero_loss_has_finite_gradient(self):
        predicted = self.batch['states'].clone().requires_grad_()
        loss, _ = wm.dynamics_loss(predicted, predicted.detach(), self.spec)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))

    def test_checkpoint_restores_network_stats_and_limiter(self):
        self.model.fit_input_stats(self.batch)
        restored = self.make_model()
        restored.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(restored(self.initial, self.actions[:, 0], self.context),
                                   self.model(self.initial, self.actions[:, 0], self.context))
        self.assertTrue(bool(restored.stats_ready))

    def test_frozen_model_retains_action_gradient(self):
        from test_supermimic_objectives import gradients
        self.model.fit_input_stats(self.batch)
        actions = self.actions.clone().requires_grad_()
        with gradients.frozen_parameters(self.model):
            out = self.rollout(actions)
            out[:, -1, :75].square().sum().backward()
        self.assertIsNotNone(actions.grad)
        self.assertGreater(float(actions.grad[:, 0].abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))


if __name__ == '__main__':
    unittest.main()
