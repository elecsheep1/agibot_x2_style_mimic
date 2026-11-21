"""Regression checks for policy collection and offline sequence sampling."""

import importlib.util
from pathlib import Path
import unittest

import torch


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "checks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = load_script("collect_supermimic_smoke")
fitter = load_script("fit_supermimic_smoke")


class PolicyCollectionTests(unittest.TestCase):
    def test_policy_reads_current_observation_and_zero_noise(self):
        observation = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        action = collector.policy_action(lambda value: value[:, :2], {"policy": observation}, 3, 2, 0.0)
        torch.testing.assert_close(action, observation[:, :2], atol=0, rtol=0)

    def test_noise_is_added_to_policy_output(self):
        observation = {"policy": torch.zeros(3, 4)}
        torch.manual_seed(42)
        expected = 0.05 * torch.randn(3, 2) + 0.3
        torch.manual_seed(42)
        action = collector.policy_action(lambda value: value[:, :2] + 0.3, observation, 3, 2, 0.05)
        torch.testing.assert_close(action, expected, atol=0, rtol=0)

    def test_invalid_policy_output(self):
        observation = {"policy": torch.zeros(3, 4)}
        with self.assertRaises(RuntimeError):
            collector.policy_action(lambda value: value, observation, 3, 2, 0.0)
        with self.assertRaises(RuntimeError):
            collector.policy_action(lambda value: value[:, :2] + float("nan"), observation, 3, 2, 0.0)


class OfflineWindowsTests(unittest.TestCase):
    def test_early_sampling_respects_fraction(self):
        data = dict(
            state=torch.arange(10, dtype=torch.float32).reshape(10, 1, 1),
            next_state=torch.arange(1, 11, dtype=torch.float32).reshape(10, 1, 1),
            action=torch.zeros(10, 1, 1), contexts=torch.ones(1, 1),
            episode_steps=torch.arange(10).reshape(10, 1), metadata={"step_dt": 1.0},
        )
        candidates = torch.stack((torch.arange(9), torch.zeros(9, dtype=torch.long)), dim=-1)
        batch = fitter.sample_training_batch(data, candidates, 10, 2, "cpu", 0.5)
        self.assertEqual(int((batch["states"][:, 0, 0] < 3).sum()), 5)
        with self.assertRaises(ValueError):
            fitter.sample_training_batch(data, candidates[:2], 10, 2, "cpu", 0.5)

    def test_windows_keep_terminal_state_and_do_not_cross_reset(self):
        state = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1)
        next_state = state + 1
        next_state[2] = 99  # Terminal snapshot differs from next reset state.
        data = dict(
            state=state, next_state=next_state, action=torch.zeros(6, 1, 1),
            contexts=torch.ones(1, 2), terminated=torch.zeros(6, 1, dtype=torch.bool),
            truncated=torch.tensor([[False], [False], [True], [False], [False], [False]]),
            episode_ids=torch.tensor([[0], [0], [0], [1], [1], [1]]),
            step_ids=torch.tensor([[0], [1], [2], [0], [1], [2]]),
        )
        candidates = fitter.valid_windows(data, 2)
        self.assertEqual(candidates.tolist(), [[0, 0], [1, 0], [3, 0], [4, 0]])
        batch = fitter.sample(data, torch.tensor([[1, 0]]), 2, 2, "cpu")
        torch.testing.assert_close(batch["states"][:, :, 0], torch.tensor([[1., 2., 99.], [1., 2., 99.]]))


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.state, self.dynamics = fitter.load_modules()
        self.spec = self.state.StateSpec(2, 2)

    def test_constant_rates_world_rotation_and_quaternion_sign(self):
        dt = 0.02
        q0 = self.dynamics.rotation_increment(torch.tensor([[0.4, 0.0, 0.0]]))
        q1 = self.dynamics.quat_mul(self.dynamics.rotation_increment(torch.tensor([[0.0, 0.0, dt]])), q0)
        velocity = torch.tensor([[1., 0., 0., 0., 0., 1.]])
        def pack(position, quaternion, joints):
            return self.state.pack_state(torch.cat((position, quaternion), -1), velocity, joints,
                                         torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2))
        before = pack(torch.zeros(1, 3), q0, torch.zeros(1, 2))
        after = pack(torch.tensor([[dt, 0., 0.]]), -q1, torch.full((1, 2), dt))
        states = torch.stack((before, after), dim=1)
        residuals = fitter.consistency_residuals(states, self.spec, dt, self.state, self.dynamics)
        for residual in residuals.values():
            torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2e-6, rtol=0)

    def test_deadzone_and_finite_identity_gradient(self):
        states = torch.zeros(1, 2, self.spec.dim)
        states[:, :, 6] = 1
        states[:, 1, 0] = 0.002
        states.requires_grad_()
        margins = dict(root_linear=0.2, root_angular=0.2, joint_velocity=0.2)
        loss, _ = fitter.soft_consistency_loss(states, self.spec, 0.02, margins, self.state, self.dynamics)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(states.grad).all()))
        margins = dict(root_linear=0.0, root_angular=0.0, joint_velocity=0.0)
        states.grad = None
        loss, _ = fitter.soft_consistency_loss(states, self.spec, 0.02, margins, self.state, self.dynamics)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertGreater(float(states.grad.abs().sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(states.grad).all()))


if __name__ == "__main__":
    unittest.main()
