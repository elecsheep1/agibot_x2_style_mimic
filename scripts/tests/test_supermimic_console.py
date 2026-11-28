"""终端面板的指标口径、恢复训练 ETA 和输出对齐。"""
import contextlib
import io
from types import SimpleNamespace
import unittest

from test_supermimic_training_entry import load, logging_module


class ConsoleTests(unittest.TestCase):
    def test_plain_panel_wraps_long_path_and_aligns_chinese_label(self):
        console = logging_module.IsaacLabConsole('demo')
        output = console.format_panel('Learning iteration 10/20000', [
            ('Run name', 'demo'), ('输出目录', '/home/panda/' + 'long_path_' * 20),
        ], [('Iteration time', '1.25s')])
        self.assertNotIn('\x1b', output)
        self.assertEqual(output.splitlines()[0], '#' * 100)
        self.assertTrue(all(logging_module.display_width(line) <= 100 for line in output.splitlines()))
        self.assertIn('Learning iteration 10/20000', output)

    def test_long_training_time_does_not_wrap_at_one_day(self):
        self.assertEqual(logging_module.duration_text(49 * 3600 + 62), '49:01:02')
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                logging_module.duration_text(value)

    def test_resume_eta_counts_all_iterations_including_unprinted_ones(self):
        console = logging_module.IsaacLabConsole('resume')
        metrics = {'collection/transitions': 1200, 'imagined_task_reward': .01,
                   'reward/style': .02, 'collection/action_rms': .1}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            console.iteration(41, 100, metrics, 2., 1000, .1, emit=False)
            console.iteration(42, 100, metrics, 4., 2200, .1)
        text = output.getvalue()
        self.assertEqual(console.completed_iterations, 2)
        self.assertEqual(console.training_seconds, 6.)
        self.assertIn('00:02:54', text)
        self.assertIn('Steps per second: 300', text)
        self.assertIn('Mean imagined task reward', text)
        self.assertNotIn('Mean reward:', text)
        self.assertNotIn('Learning iteration 41/', text)

    def test_real_evaluation_is_explicitly_stamped_when_reused(self):
        console = logging_module.IsaacLabConsole('evaluation')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            console.evaluation(500, 20000, {'task_success_rate': .875,
                               'standing_quality_mean': .6}, 1.5)
            console.iteration(510, 20000, {'collection/transitions': 1024}, 2., 10000, .1)
        self.assertIn('Real evaluation 500/20000', output.getvalue())
        self.assertIn('Last real evaluation iteration: 500', output.getvalue())
        self.assertIn('87.50%', output.getvalue())

    def test_iteration_instrumentation_does_not_add_algorithm_updates(self):
        training = load('getup_training')
        run = training.GetupTrainingRun.__new__(training.GetupTrainingRun)
        calls, recorded, displayed = [], [], []
        def train_iteration(reference, **kwargs):
            calls.append(kwargs)
            return {'collection/transitions': 1024, 'imagined_reward': .3, 'reward/style': .1}
        run.runtime = SimpleNamespace(reference=object(), runner=SimpleNamespace(
            train_iteration=train_iteration, total_env_steps=32), env=SimpleNamespace(num_envs=32), offline=None)
        run.cfg = SimpleNamespace(world_horizons=(8,), collection_steps=32, world_batch_size=128,
            policy_batch_size=128, amp_batch_size=128, world_updates=5, discriminator_updates=4,
            offline_world_fraction=0., imagination_horizon=32, log_interval=10, iterations=100, action_noise=.1)
        run.start = 0
        run.logger = SimpleNamespace(record=lambda *args: recorded.append(args), flush=lambda: None)
        run.console = SimpleNamespace(iteration=lambda *args, **kwargs: displayed.append((args, kwargs)))
        run._train_iteration(1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['world_window_frames'], 8)
        self.assertEqual(calls[0]['world_updates'], 5)
        self.assertAlmostEqual(recorded[0][2]['imagined_task_reward'], .2)
        self.assertGreaterEqual(recorded[0][2]['performance/iteration_seconds'], 0.)
        self.assertEqual(displayed[0][0][4], 1024)
        self.assertTrue(displayed[0][1]['emit'])


if __name__ == '__main__':
    unittest.main()
