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

    def test_fixed_windows_are_unique_within_batch(self):
        data = self.data(); train, _ = module.episode_split(data, 43)
        batch = module.fixed_evaluation_batch(data, module.candidates_for(data, 32, train), 16, 44, .1)
        selected = batch['selected_windows'].tolist()
        self.assertEqual(len({tuple(row) for row in selected}), 16)
        self.assertGreaterEqual(batch['episodes'], 1)
        self.assertLessEqual(batch['episodes'], len(train))

    def test_fixed_windows_reject_batch_larger_than_pool(self):
        data = self.data(); train, _ = module.episode_split(data, 43)
        pool = module.candidates_for(data, 32, train)
        with self.assertRaisesRegex(ValueError, '不足以无重复抽取'):
            module.fixed_evaluation_batch(data, pool, 400, 44, .1)

    def test_complete_episodes_skips_partial_rounds(self):
        data = self.data(); data['truncated'][-1] = False
        complete = module.complete_episodes(data)
        self.assertEqual([len(complete[env]) for env in (0, 1)], [1, 1])
        self.assertTrue(all(key[1] == 0 for keys in complete.values() for key in keys))

    def test_environment_split_is_deterministic_and_disjoint(self):
        complete = module.complete_episodes(self.data())
        train_envs, holdout_envs = module.split_environments(complete, 7, 1)
        self.assertEqual((len(train_envs), len(holdout_envs)), (1, 1))
        self.assertEqual((train_envs, holdout_envs), module.split_environments(complete, 7, 1))
        self.assertFalse(set(train_envs) & holdout_envs)
        self.assertEqual(module.split_environments(complete, 7, 0), ([0, 1], set()))
        for count in (-1, 2):
            with self.assertRaisesRegex(ValueError, '留出环境数量'):
                module.split_environments(complete, 7, count)

    def test_split_episodes_only_uses_requested_environments(self):
        complete = module.complete_episodes(self.data())
        train, holdout = module.split_episodes(complete, [0], 43)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(holdout), 1)
        self.assertTrue(all(env == 0 for env, _ in train + holdout))

    def test_world_window_converts_frames_to_transitions(self):
        self.assertEqual(module.world_window(SimpleNamespace(world_horizons=(8,))), (8, 7))
        self.assertEqual(module.world_window(SimpleNamespace(world_horizons=(4, 8))), (8, 7))
        with self.assertRaisesRegex(ValueError, '两帧'):
            module.world_window(SimpleNamespace(world_horizons=(1,)))
        with self.assertRaisesRegex(ValueError, '不能为空'):
            module.world_window(SimpleNamespace(world_horizons=()))

    def test_check_updates_deduplicates_and_includes_endpoints(self):
        self.assertEqual(module.check_updates(500), [0, 100, 250, 500])
        self.assertEqual(module.check_updates(120), [0, 100, 120])
        self.assertEqual(module.check_updates(60), [0, 60])
        with self.assertRaises(ValueError):
            module.check_updates(0)

    def test_freeze_normalization_accepts_keyword_calls(self):
        class WorldNormalizer:
            def __init__(self):
                self.calls = 0
            def update_normalization(self, real_states, real_actions, context):
                self.calls += 1
        class ProprioNormalizer:
            def __init__(self):
                self.calls = 0
            def update_normalization(self, states):
                self.calls += 1
        trainer = SimpleNamespace(world_model=WorldNormalizer(), proprioception=ProprioNormalizer())
        originals = module.freeze_normalization(trainer)
        trainer.world_model.update_normalization(real_states=torch.zeros(1),
                                                 real_actions=torch.zeros(1), context=torch.zeros(1))
        trainer.proprioception.update_normalization(torch.zeros(1))
        self.assertEqual((trainer.world_model.calls, trainer.proprioception.calls), (0, 0))
        module.restore_normalization(trainer, originals)
        trainer.world_model.update_normalization(real_states=torch.zeros(1),
                                                 real_actions=torch.zeros(1), context=torch.zeros(1))
        trainer.proprioception.update_normalization(torch.zeros(1))
        self.assertEqual((trainer.world_model.calls, trainer.proprioception.calls), (1, 1))

    def test_freeze_normalization_tolerates_missing_world_hook(self):
        trainer = SimpleNamespace(world_model=SimpleNamespace(),
                                  proprioception=SimpleNamespace(update_normalization=lambda states: None))
        originals = module.freeze_normalization(trainer)
        module.restore_normalization(trainer, originals)
        trainer.proprioception.update_normalization(torch.zeros(1))

    def test_summarize_reports_balanced_root_position_only(self):
        row = {'holdout': {'balanced': {str(h): {'root_position_rmse_m': .1 * h}
                                        for h in module.EVALUATION_HORIZONS}},
               'train': {'balanced': {str(h): {'root_position_rmse_m': 0.}
                                      for h in module.EVALUATION_HORIZONS}},
               'holdout_reward_direction': {'return_mae': 1.}}
        self.assertEqual(module.summarize(row),
                         {'holdout': {h: round(.1 * h, 5) for h in module.EVALUATION_HORIZONS}})


if __name__ == '__main__':
    unittest.main()
