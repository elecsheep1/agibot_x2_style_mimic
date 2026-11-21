"""CPU tests for SuperMimic context and sequential replay data contracts.

Physics reads are mocked: these tests do not validate actual PhysX integration.
"""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_supermimic_objectives import load_module

context = load_module('context')
buffer_module = load_module('buffer')


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.spec = context.ContextSpec(('base', 'torso'), ('joint',), 3)
        self.fields = dict(
            material=torch.arange(18.).reshape(2, 3, 3),
            mass=torch.tensor([[1., 2.], [3., 4.]]),
            inertia=torch.arange(36.).reshape(2, 2, 9),
            stiffness=torch.tensor([[80.], [88.]]),
            damping=torch.tensor([[4.5], [4.9]]))

    def test_layout_and_roundtrip(self):
        packed = context.pack_context(**self.fields, spec=self.spec)
        self.assertEqual(packed.shape, (2, 31))
        self.assertEqual(self.spec.dim, 31)
        for name, value in context.unpack_context(packed, self.spec).items():
            torch.testing.assert_close(value, self.fields[name])
        torch.testing.assert_close(packed[:, :9], self.fields['material'].flatten(1))
        torch.testing.assert_close(packed[:, 9:11], self.fields['mass'])

    def test_reject_old_matrix_shape_and_wrong_dimension(self):
        fields = dict(self.fields)
        fields['inertia'] = fields['inertia'].reshape(2, 2, 3, 3)
        with self.assertRaises(ValueError):
            context.pack_context(**fields, spec=self.spec)
        with self.assertRaises(ValueError):
            context.unpack_context(torch.zeros(2, 30), self.spec)

    def test_packing_does_not_alias_inputs(self):
        packed = context.pack_context(**self.fields, spec=self.spec)
        original = packed.clone()
        self.fields['material'].zero_()
        torch.testing.assert_close(packed, original)

    def read_with_fake_backend(self, nonfinite=False, explicit=False):
        class Implicit:
            pass
        warp = ModuleType('warp')
        warp.to_torch = lambda value: value
        actuators = ModuleType('isaaclab.actuators')
        actuators.ImplicitActuator = Implicit
        material = self.fields['material'].clone()
        if nonfinite:
            material[0, 0, 0] = float('nan')
        data = SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.zeros(2, 1)))
        for field, key in (('body_mass', 'mass'), ('body_inertia', 'inertia'),
                           ('joint_stiffness', 'stiffness'), ('joint_damping', 'damping')):
            setattr(data, field, SimpleNamespace(torch=self.fields[key]))
        robot = SimpleNamespace(
            body_names=['base', 'torso'], joint_names=['joint'], data=data,
            actuators={'pd': object() if explicit else Implicit()},
            root_view=SimpleNamespace(get_material_properties=lambda: material))
        env = SimpleNamespace(scene={'robot': robot}, num_envs=2)
        with patch.dict(sys.modules, {'warp': warp, 'isaaclab.actuators': actuators}):
            return context.read_sim_context(env)

    def test_reader_uses_actual_values_and_returns_detached_copy(self):
        result, spec = self.read_with_fake_backend()
        self.assertEqual(spec, self.spec)
        torch.testing.assert_close(result, context.pack_context(**self.fields, spec=spec))
        self.assertFalse(result.requires_grad)
        before = result.clone()
        self.fields['mass'].zero_()
        torch.testing.assert_close(result, before)

    def test_reader_rejects_nan_and_unsupported_actuator(self):
        with self.assertRaises(RuntimeError):
            self.read_with_fake_backend(nonfinite=True)
        with self.assertRaises(NotImplementedError):
            self.read_with_fake_backend(explicit=True)


class BufferTests(unittest.TestCase):
    def make_buffer(self, capacity=5):
        return buffer_module.SequenceReplayBuffer(
            2, capacity, 1, 1, torch.tensor([[100.], [200.]]))

    def transition(self, t, terminated=(False, False), truncated=(False, False)):
        value = torch.tensor([[float(t)], [float(t + 100)]])
        return dict(state=value, next_state=value+1, action=value+1000,
                    reward=value[:, 0]+2000, terminated=torch.tensor(terminated),
                    truncated=torch.tensor(truncated))

    def assert_alignment(self, batch):
        self.assertEqual(batch['states'].shape[1], batch['actions'].shape[1]+1)
        torch.testing.assert_close(batch['states'][:, 1:], batch['states'][:, :-1]+1)
        torch.testing.assert_close(batch['actions'], batch['states'][:, :-1]+1000)
        torch.testing.assert_close(batch['rewards'], batch['states'][:, :-1, 0]+2000)
        torch.testing.assert_close(batch['context'][:, 0], (batch['context_id'].float()+1)*100)
        self.assertFalse(bool((batch['terminated'][:, :-1] | batch['truncated'][:, :-1]).any()))

    def test_empty_and_insufficient_data(self):
        buf = self.make_buffer()
        with self.assertRaises(RuntimeError):
            buf.sample_initial_states(1)
        buf.add(self.transition(0))
        with self.assertRaises(RuntimeError):
            buf.sample_windows(1, 2)

    def test_ring_wrap_episode_boundaries_and_manual_reset(self):
        buf = self.make_buffer()
        for t in range(9):
            if t == 7:
                buf.mark_reset([1])
            buf.add(self.transition(t, terminated=(t == 5, False), truncated=(False, t == 4)))
        starts, candidates = buf._valid_starts(2)
        actual = {(int(starts[i]), int(e)) for i, e in candidates}
        self.assertEqual(actual, {(4, 0), (5, 1), (6, 0), (7, 0), (7, 1)})
        self.assertEqual(buf.size_steps, 5)
        batch = buf.sample_windows(256, 2)
        self.assert_alignment(batch)
        for i in range(256):
            env = int(batch['context_id'][i])
            t = int(batch['states'][i, 0, 0]) - 100*env
            self.assertIn((t, env), actual)

    def test_final_terminal_state_is_not_next_reset_state(self):
        buf = self.make_buffer()
        buf.add(self.transition(0))
        buf.add(self.transition(1, terminated=(True, True)))
        buf.add(self.transition(50))
        batch = buf.sample_windows(32, 2)
        self.assert_alignment(batch)
        self.assertTrue(bool(batch['terminated'][:, -1].all()))
        self.assertTrue(bool((batch['step_id'] == 0).all()))

    def test_initial_state_sampling_metadata(self):
        buf = self.make_buffer()
        for t in range(4):
            buf.add(self.transition(t))
        initial = buf.sample_initial_states(64)
        self.assertEqual(initial['state'].shape, (64, 1))
        torch.testing.assert_close(
            initial['state'][:, 0], initial['step_id'].float()+100*initial['context_id'])
        self.assertTrue(bool((initial['episode_id'] == 0).all()))

    def test_copy_and_detach(self):
        buf = self.make_buffer()
        item = self.transition(0)
        item['state'].requires_grad_(True)
        buf.add(item)
        item['next_state'].fill_(-100.)
        batch = buf.sample_windows(8, 1)
        self.assert_alignment(batch)
        self.assertFalse(batch['states'].requires_grad)
        batch['states'].fill_(-999.)
        self.assert_alignment(buf.sample_windows(8, 1))

    def test_invalid_input_rejected_without_advancing(self):
        buf = self.make_buffer()
        item = self.transition(0)
        item['action'] = torch.zeros(2, 2)
        with self.assertRaises(ValueError):
            buf.add(item)
        item = self.transition(0)
        item['terminated'] = torch.zeros(2)
        with self.assertRaises(ValueError):
            buf.add(item)
        self.assertEqual(buf.total_steps, 0)

    def test_invalid_sampling_arguments(self):
        buf = self.make_buffer()
        for horizon in (0, 6):
            with self.assertRaises(ValueError):
                buf.sample_windows(1, horizon)
        with self.assertRaises(ValueError):
            buf.sample_windows(0, 1)


if __name__ == '__main__':
    unittest.main()
