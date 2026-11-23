"""Regressions for get-up semantics: lying is valid, empty goals and differentiable reward."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from test_supermimic_world_model import load, state

head = load('heads')
runner_module = load('runner')


class TorsoKinematics(nn.Module):
    body_names = ('torso_link',)

    def forward(self, root_pose, root_velocity, joint_position, joint_velocity):
        return root_pose[:, None], root_velocity[:, None]


class ContactGeometry(nn.Module):
    def forward(self, states):
        height = states[:, 2]
        return torch.stack((height - .65, height - .65, height + .1,
                            height + .1, height + .4, height + .4), dim=-1)


class GetUpTests(unittest.TestCase):
    def setUp(self):
        self.spec = state.StateSpec(3, 2)
        self.cfg = head.GetupTaskCfg()
        self.task = head.GetupTask(self.spec, torch.tensor([[-1., 1.]] * 3), self.cfg,
                                  TorsoKinematics(), torch.tensor([0, 2]), ContactGeometry())
        self.standing = torch.zeros(2, self.spec.dim)
        self.standing[:, 2], self.standing[:, 6] = 0.65, 1.

    def test_lying_is_not_terminal_but_out_of_bounds_is(self):
        lying = self.standing.clone()
        lying[:, 2] = 0.09
        lying[:, 3:7] = torch.tensor([0., 2**-0.5, 0., 2**-0.5])
        self.assertFalse(bool(self.task.terminated(lying).any()))
        self.assertLess(float(self.task.termination_probability(lying).max()), 1e-6)
        lying[0, 0] = 2.1
        self.assertEqual(self.task.terminated(lying).tolist(), [True, False])

    def test_boundary_forward_is_hard_but_gradient_is_smooth(self):
        states = self.standing.clone()
        states[:, 0] = torch.tensor([1.9584, 2.01])
        states.requires_grad_()
        probability = self.task.termination_probability(states)
        torch.testing.assert_close(probability, torch.tensor([0., 1.]), atol=0, rtol=0)
        probability.sum().backward()
        self.assertTrue(bool((states.grad[:, 0] > 0).all()))
        self.assertTrue(bool(torch.isfinite(states.grad).all()))
        _, terms = self.task.reward(states, torch.zeros(2, 2), states,
                                    torch.empty(2, 0), torch.zeros(2))
        torch.testing.assert_close(terms['out_of_bounds'], torch.tensor([0., -5.]), atol=0, rtol=0)

    def test_head_proxy_has_no_positive_clearance_tail(self):
        class Geometry(torch.nn.Module):
            def forward(self, states):
                return states[:, :6]
        original = self.task.contact_geometry
        self.task.contact_geometry = Geometry()
        try:
            states = self.standing.clone()
            states[:, 4:6] = torch.tensor([[.01, .001], [-.0025, -.005]])
            states.requires_grad_()
            cost = self.task.head_contact_cost(states)
            torch.testing.assert_close(cost, torch.tensor([0., 1.5]))
            cost.sum().backward()
            self.assertLess(float(states.grad[1, 4]), 0.)
            self.assertTrue(bool(torch.isfinite(states.grad).all()))
        finally:
            self.task.contact_geometry = original

    def test_standing_quality_and_height_gradient(self):
        low = self.standing.clone()
        low[:, 2] = 0.3
        self.assertTrue(bool((self.task.standing_quality(self.standing) > self.task.standing_quality(low)).all()))
        low.requires_grad_()
        self.task.standing_quality(low).sum().backward()
        self.assertTrue(bool((low.grad[:, 2] > 0).all()))
        self.assertTrue(bool(torch.isfinite(low.grad).all()))

    def test_reward_terms_dt_and_empty_goals(self):
        reward, terms = self.task.reward(self.standing, torch.zeros(2, 2), self.standing,
                                         torch.empty(2, 0), torch.ones(2))
        torch.testing.assert_close(reward, torch.stack(list(terms.values())).sum(0))
        torch.testing.assert_close(terms['style'], torch.full((2,), self.cfg.step_dt))
        torch.testing.assert_close(terms['joint_acc'], torch.zeros(2))
        with self.assertRaises(ValueError):
            self.task.reward(self.standing, torch.zeros(2, 2), self.standing,
                             torch.zeros(2, 3), torch.ones(2))

    def test_controlled_joint_acceleration(self):
        following = self.standing.clone()
        fields = state.unpack_state(following, self.spec)
        fields['joint_velocity'][:, 1] = 10.  # uncontrolled joint
        _, terms = self.task.reward(self.standing, torch.zeros(2, 2), following,
                                    torch.empty(2, 0), torch.zeros(2))
        torch.testing.assert_close(terms['joint_acc'], torch.zeros(2))

    def test_critic_updates_with_empty_goal_dimension(self):
        trainer_module = load('trainer')
        networks = load('actor_critic')
        trainer = trainer_module.SuperMimicTrainer.__new__(trainer_module.SuperMimicTrainer)
        trainer.device = 'cpu'
        trainer.cfg = trainer_module.TrainerCfg(critic_epochs=1, critic_minibatches=1)
        trainer.critic = networks.GoalConditionedCritic(self.spec.dim, 2, 0)
        trainer.critic_optimizer = torch.optim.Adam(trainer.critic.parameters(), lr=1e-3)
        trajectory = SimpleNamespace(states=self.standing[None].expand(4, -1, -1),
                                     goals=torch.empty(4, 2, 0), task_contexts=torch.empty(4, 2, 0), rewards=torch.ones(3, 2),
                                     termination_probabilities=torch.zeros(3, 2))
        metrics = trainer.update_critic(trajectory, torch.zeros(2, 2))
        self.assertGreater(metrics['critic_gradient_norm'], 0)
        self.assertEqual(metrics['critic_updates'], 1)

    def test_mixed_world_batches_keep_sources_aligned(self):
        class Replay:
            state_dim, action_dim = 2, 1
            contexts = torch.zeros(1, 3)

            def __init__(self, marker):
                self.marker = marker

            def sample_windows(self, batch_size, horizon):
                return dict(states=torch.full((batch_size, horizon + 1, 2), self.marker),
                            actions=torch.full((batch_size, horizon, 1), self.marker),
                            context=torch.full((batch_size, 3), self.marker))

        runner = runner_module.SuperMimicRunner.__new__(runner_module.SuperMimicRunner)
        runner.device, runner.replay = 'cpu', Replay(0.)
        for fraction, expected in ((0., 0), (.5, 4), (1., 8)):
            batch = runner.sample_world_batch(8, 3, Replay(1.), fraction)
            marker = batch['states'][:, 0, 0]
            self.assertEqual(int(marker.sum()), expected)
            torch.testing.assert_close(marker, batch['actions'][:, 0, 0])
            torch.testing.assert_close(marker, batch['context'][:, 0])
        batch = runner.sample_world_batch(8, 3)
        self.assertEqual(float(batch['states'].sum()), 0.)
        with self.assertRaises(ValueError):
            runner.sample_world_batch(8, 3, Replay(1.), 1.5)

    def test_runner_empty_goals_needs_no_command_manager(self):
        runner = runner_module.SuperMimicRunner.__new__(runner_module.SuperMimicRunner)
        runner.command_name, runner.device = None, 'cpu'
        runner.env = SimpleNamespace(num_envs=2)
        runner.replay = SimpleNamespace(command_dim=0)
        self.assertEqual(runner._read_commands().shape, (2, 0))
        runner.replay.command_dim = 3
        with self.assertRaises(ValueError):
            runner._read_commands()


if __name__ == '__main__':
    unittest.main()
