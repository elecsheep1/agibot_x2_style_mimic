"""实验配方、命令行覆盖和旧检查点配置恢复，不启动 Isaac Sim。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parents[1] / 'supermimic'
spec = importlib.util.spec_from_file_location('getup_config_tests', DIRECTORY / 'getup_config.py')
config = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = config
spec.loader.exec_module(config)


class EntryConfigTests(unittest.TestCase):
    def test_agent_config_is_effective_default_for_nested_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runner.py'
            text = config.DEFAULT_CONFIG.read_text()
            text = text.replace('gamma: float = 0.99', 'gamma: float = 0.91')
            text = text.replace('learning_rate: float = 1e-4', 'learning_rate: float = 3e-5')
            text = text.replace('critic_epochs: int = 5', 'critic_epochs: int = 2')
            path.write_text(text)
            with patch.object(config, 'DEFAULT_CONFIG', path):
                cfg, _ = config.parse_config([])
                self.assertEqual(cfg.gamma, .91)
                self.assertEqual(cfg.discriminator_lr, 3e-5)
                self.assertEqual(cfg.critic_epochs, 2)
                override, _ = config.parse_config(['--gamma', '.95'])
                self.assertEqual(override.gamma, .95)

    def test_default_recipe_matches_previous_explicit_command(self):
        cfg, _ = config.parse_config([])
        expected = dict(world_model_scratch=True, world_hidden_dims=(512, 512, 512), offline_world_fraction=0.,
                        num_envs=16, iterations=200, batch_size=128, world_horizons=(8,), imagination_horizon=32,
                        warmup_steps=512, warmup_updates=500, warmup_collect_interval=50,
                        eval_interval=50, checkpoint_interval=50, eval_seconds=20., seed=43,
                        device='cuda:0', viz='none')
        for key, value in expected.items():
            self.assertEqual(getattr(cfg, key), value, key)

    def test_file_then_command_line_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.toml'
            path.write_text('world_model_scratch=true\niterations=10\nbatch_size=32\n')
            cfg, _ = config.parse_config(['--config', str(path), '--iterations', '20', '--world-horizons', '8', '16'])
        self.assertEqual(cfg.iterations, 20)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.world_horizons, (8, 16))

    def test_pretrained_override_clears_scratch_initialization(self):
        cfg, _ = config.parse_config(['--world-model-init', 'model.pt'])
        self.assertFalse(cfg.world_model_scratch)
        self.assertEqual(cfg.world_model_init, Path('model.pt'))
        self.assertIsNone(cfg.resume)

    def test_resume_loads_actual_previous_arguments_and_resets_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            saved = config.read_config(config.DEFAULT_CONFIG).to_dict()
            saved.update(output=str(path), num_envs=32, batch_size=64, warmup_updates=17,
                         device=None, unused_old_launcher_flag=True)
            (path / 'arguments.json').write_text(json.dumps(saved))
            cfg, _ = config.parse_config(['--resume', str(path / 'latest.pt'), '--iterations', '400'])
            self.assertEqual(cfg.resume, path / 'latest.pt')
            self.assertFalse(cfg.world_model_scratch)
            self.assertIsNone(cfg.world_model_init)
            self.assertEqual((cfg.num_envs, cfg.batch_size, cfg.warmup_updates), (32, 64, 17))
            self.assertEqual(cfg.iterations, 400)
            self.assertEqual(cfg.device, 'cuda:0')
            self.assertIsNone(cfg.output)

    def test_unknown_config_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '未知字段'):
            config.GetupTrainConfig.from_mapping({'world_horzon': 8})

    def test_wrong_field_type_is_rejected(self):
        with self.assertRaises(ValueError):
            config.GetupTrainConfig.from_mapping({'num_envs': '16'})
        with self.assertRaises(ValueError):
            config.GetupTrainConfig.from_mapping({'world_horizons': [True]})

    def test_conflicting_initialization_flags_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            config.parse_config(['--world-model-scratch', '--resume', 'latest.pt'])

    def test_legacy_explicit_command_resolves_to_same_recipe(self):
        cfg, _ = config.parse_config([
            '--world-model-scratch', '--world-hidden-dims', '512', '512', '512',
            '--offline-world-fraction', '0', '--num-envs', '16', '--iterations', '200',
            '--batch-size', '128', '--world-horizons', '8', '--imagination-horizon', '32',
            '--warmup-steps', '512', '--warmup-updates', '500', '--warmup-collect-interval', '50',
            '--eval-interval', '50', '--checkpoint-interval', '50', '--eval-seconds', '20',
            '--seed', '43', '--device', 'cuda:0', '--viz', 'none',
        ])
        self.assertEqual(cfg.to_dict(), config.read_config(config.DEFAULT_CONFIG).to_dict())


if __name__ == '__main__':
    unittest.main()
