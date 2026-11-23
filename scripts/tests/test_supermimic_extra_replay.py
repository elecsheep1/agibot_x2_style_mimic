"""附加转移字段的回放、回合边界和校验回归。"""
import ast
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'source/x2_style_mimic/x2_style_mimic'
spec = importlib.util.spec_from_file_location('_extra_replay_buffer', BASE / 'algorithms/supermimic/buffer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
schema_path = BASE / 'tasks/manager_based/x2_style_mimic/mdp/supermimic/transition_data.py'
tree = ast.parse(schema_path.read_text())
namespace = {'torch': torch}
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'transition_extra_fields')
exec(compile(ast.Module(body=[node], type_ignores=[]), str(schema_path), 'exec'), namespace)


class ExtraReplayTests(unittest.TestCase):
    def make_buffer(self):
        return module.SequenceReplayBuffer(2, 6, 13, 1, torch.zeros(2, 1),
                                          extra_fields=namespace['transition_extra_fields'](13))

    def row(self, buffer, step):
        row = dict(state=torch.full((2, 13), float(step)), next_state=torch.full((2, 13), float(step + 1)),
                   action=torch.zeros(2, 1), reward=torch.zeros(2), terminated=torch.zeros(2, dtype=torch.bool),
                   truncated=torch.full((2,), step % 4 == 3, dtype=torch.bool))
        for path, (shape, dtype) in buffer.extra_fields.items():
            target = row
            for key in path[:-1]:
                target = target.setdefault(key, {})
            target[path[-1]] = torch.full((2, *shape), bool(step % 2) if dtype == torch.bool else step,
                                          dtype=dtype)
        row['task_memory']['hold_steps'].fill_(step)
        row['next_task_memory']['hold_steps'].fill_(step + 1)
        return row

    def test_ring_wrap_nested_fields_and_terminal_boundaries(self):
        replay = self.make_buffer()
        for t in range(10):
            row = self.row(replay, t)
            replay.add(row)
            row['task_memory']['hold_steps'].fill_(-99)
        batch = replay.sample_windows(128, 3)
        expected = batch['states'][:, :-1, 0].long()
        torch.testing.assert_close(batch['task_memory']['hold_steps'], expected)
        torch.testing.assert_close(batch['next_task_memory']['hold_steps'], expected + 1)
        self.assertFalse(batch['truncated'][:, :-1].any())
        self.assertTrue(batch['truncated'][:, -1].any())

    def test_bad_extra_field_never_changes_buffer_or_counters(self):
        replay = self.make_buffer()
        replay.add(self.row(replay, 0))
        before = replay.state.clone()
        for kind in ('missing', 'dtype', 'shape', 'nan'):
            row = self.row(replay, 1)
            if kind == 'missing':
                del row['task_memory']['hold_steps']
            elif kind == 'dtype':
                row['task_memory']['hold_steps'] = torch.zeros(2)
            elif kind == 'shape':
                row['amp_reward_gate'] = torch.zeros(2, 5)
            else:
                row['amp_reward_gate'][0] = float('nan')
            with self.assertRaises(ValueError):
                replay.add(row)
            self.assertEqual(replay.total_steps, 1)
            torch.testing.assert_close(replay.state, before, equal_nan=True)

    def test_imagination_start_memory_matches_last_history_state(self):
        replay = self.make_buffer()
        for t in range(3):
            replay.add(self.row(replay, t))
        batch = replay.sample_policy_histories(8, 3)
        torch.testing.assert_close(batch['task_memory']['hold_steps'], batch['history'][:, -1, 0].long())


if __name__ == '__main__':
    unittest.main()
