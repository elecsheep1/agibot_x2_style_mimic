"""CPU regressions for SuperMimic foundations; no Isaac Sim startup required.

The environment tests execute the production class with a mock base. They verify
capture ordering, not physics, sensor behavior, or real ManagerBased integration.
"""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'source/x2_style_mimic/x2_style_mimic'


def load_module(name):
    path = SOURCE / 'algorithms/supermimic' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'_supermimic_test_{name}', path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations using the owning module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


objectives = load_module('objectives')
gradients = load_module('gradients')
state = load_module('state')


class ObjectiveTests(unittest.TestCase):
    def test_discounted_bootstrap(self):
        loss = objectives.imagined_actor_loss(
            torch.ones(2, 1), torch.zeros(2, 1), torch.tensor([3.]), gamma=.5)
        self.assertEqual(loss.ndim, 0)
        torch.testing.assert_close(loss, torch.tensor(-2.25))

    def test_terminal_step_keeps_reward_and_removes_future(self):
        rewards = torch.tensor([[2.], [100.]], requires_grad=True)
        terminal = torch.tensor([[1.], [0.]])
        bootstrap = torch.tensor([500.], requires_grad=True)
        loss = objectives.imagined_actor_loss(rewards, terminal, bootstrap)
        loss.backward()
        torch.testing.assert_close(loss, torch.tensor(-2.))
        torch.testing.assert_close(rewards.grad, torch.tensor([[-1.], [0.]]))
        torch.testing.assert_close(bootstrap.grad, torch.zeros(1))

    def test_soft_survival(self):
        loss = objectives.imagined_actor_loss(
            torch.tensor([[2.], [4.]]), torch.tensor([[.25], [.5]]),
            torch.tensor([8.]), gamma=.5)
        # 2 + .5*.75*4 + .5**2*.75*.5*8
        torch.testing.assert_close(loss, torch.tensor(-4.25))

    def test_actor_gradcheck(self):
        generator = torch.Generator().manual_seed(7)
        rewards = torch.randn(3, 2, generator=generator, dtype=torch.double,
                              requires_grad=True)
        probabilities = torch.full((3, 2), .2, dtype=torch.double, requires_grad=True)
        bootstrap = torch.randn(2, generator=generator, dtype=torch.double,
                                requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            objectives.imagined_actor_loss, (rewards, probabilities, bootstrap)))

    def test_lambda_endpoints_and_detached_targets(self):
        rewards = torch.ones(2, 1, requires_grad=True)
        probabilities = torch.zeros(2, 1, requires_grad=True)
        values = torch.tensor([[0.], [2.], [3.]], requires_grad=True)
        for lam, expected in ((0., [[2.], [2.5]]), (1., [[2.25], [2.5]])):
            with self.subTest(lam=lam):
                targets = objectives.lambda_returns(
                    rewards, probabilities, values, gamma=.5, lam=lam)
                torch.testing.assert_close(targets, torch.tensor(expected))
                self.assertFalse(targets.requires_grad)

    def test_lambda_terminal(self):
        targets = objectives.lambda_returns(
            torch.tensor([[2.], [100.]]), torch.tensor([[1.], [0.]]),
            torch.tensor([[0.], [0.], [500.]]))
        torch.testing.assert_close(targets[0], torch.tensor([2.]))

    def test_critic_does_not_backpropagate_into_targets(self):
        predictions = torch.tensor([1., 3.], requires_grad=True)
        targets = torch.tensor([2., 1.], requires_grad=True)
        loss = objectives.critic_loss(predictions, targets)
        torch.testing.assert_close(loss, torch.tensor(2.5))
        loss.backward()
        torch.testing.assert_close(predictions.grad, torch.tensor([-1., 2.]))
        self.assertIsNone(targets.grad)

    def test_invalid_objective_inputs(self):
        r, d, v = torch.ones(2, 1), torch.zeros(2, 1), torch.zeros(1)
        for args in ((r, d[:1], v), (r, d, v[:, None]),
                     (r[:0], d[:0], v)):
            with self.subTest(shapes=[x.shape for x in args]):
                with self.assertRaises(ValueError):
                    objectives.imagined_actor_loss(*args)
        with self.assertRaises(ValueError):
            objectives.imagined_actor_loss(r, d, v, gamma=1.)
        with self.assertRaises(ValueError):
            objectives.lambda_returns(r, d, torch.zeros(3, 1), lam=1.1)
        with self.assertRaises(ValueError):
            objectives.critic_loss(torch.zeros(2, 1), torch.zeros(2))


class GradientTests(unittest.TestCase):
    def test_freeze_keeps_input_gradient(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.)
        action = torch.ones(1, 1, requires_grad=True)
        with gradients.frozen_parameters(model, model):
            self.assertFalse(model.weight.requires_grad)
            model(action).sum().backward()
        torch.testing.assert_close(action.grad, torch.tensor([[2.]]))
        self.assertIsNone(model.weight.grad)
        self.assertTrue(model.weight.requires_grad)

    def test_exception_and_nested_context_restore_flags(self):
        model = nn.Linear(1, 1)
        model.bias.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with gradients.frozen_parameters(model):
                with gradients.frozen_parameters(model):
                    pass
                self.assertFalse(model.weight.requires_grad)
                raise RuntimeError('probe')
        self.assertTrue(model.weight.requires_grad)
        self.assertFalse(model.bias.requires_grad)


class StateTests(unittest.TestCase):
    def test_roundtrip_and_no_alias_to_source(self):
        fields = [torch.full((2, width), float(i))
                  for i, width in enumerate((7, 6, 31, 31, 29, 29))]
        spec = state.StateSpec(31, 29)
        packed = state.pack_state(*fields)
        self.assertEqual(packed.shape, (2, 133))
        for actual, expected in zip(state.unpack_state(packed, spec).values(), fields):
            torch.testing.assert_close(actual, expected)
        fields[0].fill_(100.)
        torch.testing.assert_close(packed[:, :7], torch.zeros(2, 7))

    def test_invalid_state_dimensions(self):
        fields = [torch.zeros(2, width) for width in (7, 6, 31, 31, 29, 29)]
        fields[-1] = torch.zeros(2, 28)
        with self.assertRaises(ValueError):
            state.pack_state(*fields)
        with self.assertRaises(ValueError):
            state.unpack_state(torch.zeros(2, 132), state.StateSpec(31, 29))


def environment_class():
    class MockBase:
        def step(self, action):
            self.current = self.current + 10
            if self.fail_step:
                raise RuntimeError('step probe')
            ids = (self.terminated | self.truncated).nonzero().flatten()
            if ids.numel():
                self._reset_idx(ids)
            return ({}, torch.zeros(self.num_envs), self.terminated,
                    self.truncated, {'getup_progress': {'assist_applied_force_n': torch.ones(self.num_envs)},
                    'amp_reward_gate': torch.ones(self.num_envs),
                    'amp_policy_sample_mask': torch.ones(self.num_envs, dtype=torch.bool)})

        def _reset_idx(self, ids):
            self.current[ids] = -100

    path = SOURCE / 'tasks/manager_based/x2_style_mimic/x2_supermimic_env.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'X2SuperMimicEnv')
    namespace = dict(torch=torch, X2MultiMotionGetUpEnv=MockBase,
                     read_task_memory=lambda env: {'hold_steps': env.current[:, 0].clone()},
                     read_contact_targets=lambda env: {'self_contact_cost': env.current[:, 0].clone()},
                     StateSpec=state.StateSpec, pack_state=state.pack_state)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['X2SuperMimicEnv']


class EnvironmentCaptureTests(unittest.TestCase):
    def make_env(self, terminated, truncated):
        env = environment_class()()
        env.num_envs = 3
        env.device = 'cpu'
        env.current = torch.tensor([[1.], [2.], [3.]]).expand(-1, 13).clone()
        env.terminated = torch.tensor(terminated)
        env.truncated = torch.tensor(truncated)
        env.fail_step = False
        env.read_world_state = lambda: env.current.clone()
        env.scene = {'robot': SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(3, 1)))}
        env.action_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(raw_actions=torch.zeros(3, 1)))
        return env

    def test_mixed_termination_and_timeout_capture(self):
        env = self.make_env([True, False, False], [False, True, False])
        result = env.step(torch.zeros(3, 1))
        record = result[-1]['world_model']
        torch.testing.assert_close(record['state'][:, :1], torch.tensor([[1.], [2.], [3.]]))
        torch.testing.assert_close(record['next_state'][:, :1], torch.tensor([[11.], [12.], [13.]]))
        torch.testing.assert_close(env.current[:, :1], torch.tensor([[-100.], [-100.], [13.]]))
        torch.testing.assert_close(record['next_task_memory']['hold_steps'], torch.tensor([11., 12., 13.]))
        torch.testing.assert_close(record['contact_targets']['self_contact_cost'], torch.tensor([11., 12., 13.]))
        self.assertFalse(env._wm_capture_active)
        env.terminated.zero_()
        env.truncated.zero_()
        self.assertTrue(record['terminated'][0])
        self.assertTrue(record['truncated'][1])

    def test_no_resets(self):
        env = self.make_env([False]*3, [False]*3)
        record = env.step(torch.zeros(3, 1))[-1]['world_model']
        torch.testing.assert_close(record['next_state'][:, :1], torch.tensor([[11.], [12.], [13.]]))

    def test_manual_reset_does_not_capture_and_exception_clears_flag(self):
        env = self.make_env([False]*3, [False]*3)
        env._reset_idx(torch.tensor([0]))
        self.assertFalse(hasattr(env, '_wm_terminal_state'))
        env.fail_step = True
        with self.assertRaisesRegex(RuntimeError, 'step probe'):
            env.step(torch.zeros(3, 1))
        self.assertFalse(env._wm_capture_active)

    def test_read_world_state_uses_local_position_and_target(self):
        env = environment_class()()
        pose = torch.tensor([[3., 4., .6, 0., 0., 0., 1.]])
        data = SimpleNamespace(
            root_link_pose_w=SimpleNamespace(torch=pose),
            root_link_vel_w=SimpleNamespace(torch=torch.zeros(1, 6)),
            joint_pos=SimpleNamespace(torch=torch.ones(1, 31)),
            joint_vel=SimpleNamespace(torch=torch.zeros(1, 31)))
        action = SimpleNamespace(raw_actions=torch.full((1, 29), .2),
                                 _previous_target=torch.full((1, 29), .3))
        class Scene(dict):
            env_origins = torch.tensor([[3., 4., 0.]])
        env.scene = Scene(robot=SimpleNamespace(data=data))
        env.action_manager = SimpleNamespace(get_term=lambda name: action)
        result = state.unpack_state(env.read_world_state(), state.StateSpec(31, 29))
        torch.testing.assert_close(result['root_pose'][:, :3], torch.tensor([[0., 0., .6]]))
        torch.testing.assert_close(pose[:, :2], torch.tensor([[3., 4.]]))
        torch.testing.assert_close(result['previous_target'], action._previous_target)


if __name__ == '__main__':
    unittest.main()
