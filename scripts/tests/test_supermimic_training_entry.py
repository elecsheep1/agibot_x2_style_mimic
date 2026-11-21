"""脚本层固定状态、几何判定与checkpoint恢复的回归检查。"""
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

DIRECTORY = Path(__file__).resolve().parents[1] / 'supermimic'
CHECKS = str(DIRECTORY.parent / 'checks')


def load(name):
    with patch.object(sys, 'path', [str(DIRECTORY), CHECKS] + sys.path):
        spec = importlib.util.spec_from_file_location(name, DIRECTORY / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


evaluation = load('getup_evaluation')
checkpointing = load('getup_checkpointing')
warmup = load('getup_warmup')


class TrainingEntryTests(unittest.TestCase):
    def test_real_reward_and_full_task_snapshot_freeze_after_first_episode(self):
        totals = evaluation.TaskEvaluationTotals(2, 'cpu')
        progress = dict(succeeded=torch.tensor([True, False]),
                        hold_seconds=torch.tensor([3.2, 0.]),
                        max_hold_seconds=torch.tensor([4., 0.]),
                        first_success_time=torch.tensor([5., -1.]))
        # 几何判据与实际任务判据可以不同，直接采用完整任务快照。
        totals.update(torch.tensor([True, True]), torch.tensor([-5., 2.]),
                      torch.tensor([.1, .2]), progress)
        progress = dict(succeeded=torch.tensor([False, True]),
                        hold_seconds=torch.tensor([0., 3.]),
                        max_hold_seconds=torch.tensor([0., 3.]),
                        first_success_time=torch.tensor([-1., 8.]))
        totals.update(torch.tensor([False, True]), torch.tensor([999., 3.]),
                      torch.tensor([999., .3]), progress)
        torch.testing.assert_close(totals.actual_return, torch.tensor([-5., 5.]))
        torch.testing.assert_close(totals.recomputed_return, torch.tensor([.1, .5]))
        torch.testing.assert_close(totals.final_hold, torch.tensor([3.2, 3.]))
        torch.testing.assert_close(totals.max_hold, torch.tensor([4., 3.]))
        torch.testing.assert_close(totals.first_success, torch.tensor([5., 8.]))
        self.assertEqual(totals.success.tolist(), [True, True])

    def make_warmup_runtime(self, completed=False):
        class ToyWorld(torch.nn.Linear):
            def __init__(self):
                super().__init__(2, 1)
                self.register_buffer('stats_ready', torch.tensor(completed))
                self.register_buffer('input_mean', torch.zeros(2))
                self.fit_calls = 0

            def fit_input_stats(self, batch):
                self.stats_ready.fill_(True)
                self.fit_calls += 1

        model = ToyWorld()
        optimizer = torch.optim.Adam(model.parameters())
        collections, updates = [], []

        def update(batch):
            optimizer.zero_grad()
            loss = model(torch.ones(2, 2)).square().mean()
            loss.backward()
            optimizer.step()
            updates.append(float(loss.detach()))
            return {'world_loss': updates[-1]}

        modules = {name: torch.nn.Linear(2, 1) for name in ('actor', 'critic', 'style')}
        runtime = SimpleNamespace(warmup_complete=completed, modules=modules, offline=None,
            trainer=SimpleNamespace(world_model=model, update_world_model=update),
            runner=SimpleNamespace(collect=collections.append, sample_world_batch=lambda *args: {}))
        args = SimpleNamespace(warmup_steps=16, batch_size=4, world_horizons=[8], offline_world_fraction=0.,
                               warmup_updates=3, warmup_collect_interval=1, collection_steps=2)
        return runtime, args, collections, updates

    def test_scratch_warmup_updates_only_world_and_collects_fresh_data(self):
        runtime, args, collections, updates = self.make_warmup_runtime()
        saved = {name: {k: v.clone() for k, v in module.state_dict().items()} for name, module in runtime.modules.items()}
        rows = []
        warmup.warmup_world_model(runtime, args, lambda *row: rows.append(row))
        self.assertEqual(collections, [16, 2, 2])
        self.assertEqual(len(updates), 3)
        self.assertEqual(runtime.trainer.world_model.fit_calls, 1)
        self.assertTrue(runtime.warmup_complete)
        self.assertEqual([r[0] for r in rows], ['warmup'] * 3)
        for name, module in runtime.modules.items():
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, saved[name][key], atol=0, rtol=0)

    def test_completed_resume_refills_replay_without_repeating_warmup(self):
        runtime, args, collections, updates = self.make_warmup_runtime(completed=True)
        warmup.warmup_world_model(runtime, args, lambda *row: self.fail('恢复后不应重新预热'))
        self.assertEqual(collections, [16])
        self.assertEqual(updates, [])
        self.assertEqual(runtime.trainer.world_model.fit_calls, 0)

    def test_uninitialized_resume_cannot_skip_supervised_warmup(self):
        runtime, args, collections, updates = self.make_warmup_runtime()
        args.warmup_updates = 0
        with self.assertRaisesRegex(ValueError, '监督预热'):
            warmup.warmup_world_model(runtime, args, lambda *row: None)
        self.assertEqual(collections, [])
        self.assertEqual(updates, [])

    def test_fixed_cohort_covers_initial_and_rsi(self):
        clips = []
        for _ in range(2):
            pose = torch.zeros(5, 7)
            pose[:, 6] = 1
            clips.append(dict(root_pose=pose, root_velocity=torch.ones(5, 6),
                              joint_pos=torch.full((5, 2), 1.2), joint_vel=torch.full((5, 2), 5.)))
        robot = SimpleNamespace(data=SimpleNamespace(joint_pos_limits=torch.tensor([[[-1., 1.], [-1., 1.]]] * 4),
                                                     joint_vel_limits=torch.full((4, 2), 2.)))
        params = dict(transition_frame_ranges=[(1, 4), (1, 4)], height_offset=.01)
        env = SimpleNamespace(num_envs=4, device='cpu', scene={'robot': robot},
                              cfg=SimpleNamespace(events=SimpleNamespace(reset_from_reference=SimpleNamespace(params=params))))
        runtime = SimpleNamespace(env=env, reset_reference=SimpleNamespace(clips=clips, paths=['a', 'b']))
        a, b = evaluation.build_cohort(runtime, 42), evaluation.build_cohort(runtime, 42)
        self.assertEqual(a['motion_ids'].tolist(), [0, 1, 0, 1])
        self.assertEqual(a['birth_types'].tolist(), [0, 0, 2, 2])
        self.assertTrue(bool((a['frame_ids'][2:] >= 1).all()))
        torch.testing.assert_close(a['frame_ids'], b['frame_ids'])
        self.assertEqual(float(a['states']['root_velocity'][:2].abs().sum()), 0.)
        self.assertEqual(float(a['states']['joint_pos'].max()), 1.)
        self.assertEqual(float(a['states']['joint_vel'][2:].max()), 2.)

    def test_geometric_stability_conditions(self):
        pose = torch.zeros(3, 7)
        pose[:, 2], pose[:, 6] = .65, 1
        velocity = torch.zeros(3, 6)
        velocity[1, 0] = .3
        joints = torch.zeros(3, 2)
        joints[2, 0] = 2.
        fields = dict(root_pose=pose, root_velocity=velocity, joint_position=joints)
        self.assertEqual(evaluation.geometric_stable(fields, torch.ones(3), [0, 1]).tolist(), [True, False, False])

    def test_checkpoint_network_optimizer_and_rng_roundtrip(self):
        module = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(module.parameters())
        module(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        runtime = SimpleNamespace(modules={'actor': module}, optimizers={'actor': optimizer},
                                  runner=SimpleNamespace(total_env_steps=12), context=torch.ones(2, 3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            checkpointing.save_checkpoint(path, runtime, 5, {'seed': 42}, {'task': 'getup'})
            expected = (torch.rand(3), random.random(), np.random.rand())
            loaded = torch.load(path, weights_only=False)
            torch.rand(100)
            checkpointing.restore_rng(loaded)
            actual = (torch.rand(3), random.random(), np.random.rand())
            torch.testing.assert_close(actual[0], expected[0])
            self.assertEqual(actual[1:], expected[1:])
            self.assertEqual(loaded['iteration'], 5)
            self.assertTrue(loaded['optimizers']['actor']['state'])
            module.load_state_dict(loaded['modules']['actor'], strict=True)
            optimizer.load_state_dict(loaded['optimizers']['actor'])
            self.assertFalse((Path(directory) / 'checkpoint.pt.tmp').exists())


if __name__ == '__main__':
    unittest.main()
