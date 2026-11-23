"""Check the differentiable observation and actor/critic integration used by the loop smoke."""

import importlib
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from test_supermimic_scripts import collector, fitter, load_script

with patch.dict(sys.modules, {"collect_supermimic_smoke": collector, "fit_supermimic_smoke": fitter}):
    loop = load_script("supermimic_loop_smoke")


class ZeroBasePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_normalizer = nn.Identity()
        self.linear = nn.Linear(465, 29, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, observation):
        return self.linear(observation)


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.state, self.dynamics = fitter.load_modules()
        self.spec = self.state.StateSpec(31, 29)
        self.history = torch.zeros(2, 5, self.spec.dim)
        self.history[:, :, 2] = 0.65
        self.history[:, :, 6] = 1.0
        self.ids = torch.arange(29)
        self.task_ids = torch.tensor([11, 12, 21, 22])

    def test_observation_layout_and_rotation(self):
        self.history[:, :, 10] = torch.arange(1, 6)
        obs = loop.policy_observation(self.history, self.spec, self.ids, torch.zeros(29), self.state)
        self.assertEqual(obs.shape, (2, 465))
        torch.testing.assert_close(obs[0, :15].reshape(5, 3)[:, 0], torch.arange(1., 6.))
        torch.testing.assert_close(obs[0, 15:30].reshape(5, 3), torch.tensor([[0., 0., -1.]]).expand(5, -1))
        q = self.dynamics.rotation_increment(torch.tensor([[0., 0., torch.pi / 2]]))
        rotated = loop.rotate_inverse(q, torch.tensor([[1., 0., 0.]]))
        torch.testing.assert_close(rotated, torch.tensor([[0., -1., 0.]]), atol=1e-6, rtol=0)

    def test_actor_backward_freezes_model_and_critic(self):
        context = torch.ones(2, 3)
        limiter = self.dynamics.TargetLimiter(torch.zeros(29), 0.25, -torch.ones(29), torch.ones(29), torch.full((29,), 0.1))
        timing = dict(dt=.02, decimation=10) if hasattr(self.dynamics.WorldModel, 'DYNAMICS_MODE') else {}
        model = self.dynamics.WorldModel(self.spec, 3, limiter, hidden_dims=(16, 16), **timing)
        model.fit_input_stats(dict(states=self.history, actions=torch.zeros(2, 4, 29), context=context))
        actor = loop.ResidualActor(ZeroBasePolicy(), self.ids, torch.zeros(29), self.task_ids, self.spec, self.state)
        critic = nn.Linear(model.input_mean.numel() + 13, 1)
        goal = torch.full((2, 4), 0.08)
        pose_goal = self.history[:, -1, :7]
        objectives = importlib.import_module(self.state.__package__ + ".objectives")
        gradients = importlib.import_module(self.state.__package__ + ".gradients")
        with gradients.frozen_parameters(model, critic, actor.base_policy):
            states, rewards, probabilities = loop.imagine(actor, model, self.history, context, goal, pose_goal, 4,
                                                         self.task_ids, self.state, self.dynamics)
            terminal = loop.critic_value(critic, model, states[-1], context, goal, pose_goal, self.task_ids, self.state, self.dynamics)
            loss = objectives.imagined_actor_loss(rewards, probabilities, terminal)
            loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in critic.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in actor.base_policy.parameters()))
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in actor.residual.parameters()), 0)
        self.assertTrue(all(bool(torch.isfinite(parameter.grad).all()) for parameter in actor.residual.parameters()))


if __name__ == "__main__":
    unittest.main()
