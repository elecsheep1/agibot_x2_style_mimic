"""模型补训必须按完整回合隔离训练与验证，固定验收窗口。"""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
import torch

path = Path(__file__).resolve().parents[1] / 'checks/refit_supermimic_world_fixed_policy.py'
spec = importlib.util.spec_from_file_location('_fixed_policy_refit_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FixedPolicySplitTests(unittest.TestCase):
    def test_collection_reset_marks_boundary_before_discarding_steps(self):
        replay = SimpleNamespace(total_steps=256, episode=0, step=256)
        def reset():
            if replay.total_steps:
                replay.episode += 1
                replay.step = 0
        runtime = SimpleNamespace(replay=replay, runner=SimpleNamespace(reset=reset))
        module.reset_collection(runtime)
        self.assertEqual((replay.total_steps, replay.episode, replay.step), (0, 1, 0))

    def data(self):
        steps = torch.arange(100).repeat(2)[:, None].expand(-1, 2).clone()
        episodes = (torch.arange(200) // 100)[:, None].expand(-1, 2).clone()
        done = torch.zeros(200, 2, dtype=torch.bool)
        done[99] = True; done[199] = True
        return dict(state=steps.float()[..., None], next_state=(steps.float() + 1)[..., None],
                    action=torch.zeros(200, 2, 1), contexts=torch.ones(2, 3),
                    terminated=torch.zeros_like(done), truncated=done,
                    episode_ids=episodes, step_ids=steps)

    def test_whole_episode_disjoint_split_and_windows(self):
        data = self.data()
        train, holdout = module.episode_split(data, 43)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(holdout), 2)
        self.assertFalse(set(train) & set(holdout))
        self.assertEqual((train, holdout), module.episode_split(data, 43))
        for keys in (train, holdout):
            pool = module.candidates_for(data, 32, keys)
            for start, env in pool.tolist():
                self.assertIn((env, int(data['episode_ids'][start, env])), keys)
                self.assertFalse(bool(data['truncated'][start:start + 31, env].any()))

    def test_partial_episode_not_used_to_fill_split(self):
        data = self.data(); data['truncated'][-1] = False
        with self.assertRaisesRegex(ValueError, '不足两个完整回合'):
            module.episode_split(data, 43)

    def test_continuity_error_rejected(self):
        data = self.data(); data['step_ids'][50, 0] = 100
        with self.assertRaisesRegex(ValueError, '连续步数'):
            module.episode_split(data, 43)

    def test_fixed_balanced_windows_are_repeatable_and_exclude_holdout(self):
        data = self.data(); train, holdout = module.episode_split(data, 43)
        pool = module.candidates_for(data, 32, train)
        batch = module.fixed_evaluation_batch(data, pool, 16, 44, .1)
        repeated = module.fixed_evaluation_batch(data, pool, 16, 44, .1)
        torch.testing.assert_close(batch['selected_windows'], repeated['selected_windows'])
        self.assertEqual(batch['states'].shape, (16, 33, 1))
        self.assertEqual(int((batch['phase'] == 0).sum()), 8)
        for start, env in batch['selected_windows'].tolist():
            self.assertNotIn((env, int(data['episode_ids'][start, env])), holdout)


if __name__ == '__main__':
    unittest.main()
