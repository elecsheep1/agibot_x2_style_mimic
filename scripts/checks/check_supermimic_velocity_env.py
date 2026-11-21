"""Check the production SuperMimic velocity environment and termination wiring."""
import argparse
import json
from pathlib import Path
import sys

import torch


def main():
    from isaaclab_tasks.utils import (
        add_launcher_args,
        launch_simulation,
        setup_preset_cli,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--output', type=Path, default=Path('logs/supermimic/velocity_env_acceptance.json'))
    add_launcher_args(parser)
    args, rest = setup_preset_cli(parser)
    if rest:
        parser.error(str(rest))
    if args.num_envs < 2 or args.steps < 16:
        parser.error('Use at least two environments and sixteen steps')
    if args.output.exists():
        parser.error('Choose a fresh output path')
    sys.argv = [sys.argv[0]]

    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env_cfg import (
        X2SuperMimicVelocityTaskEnvCfg,
    )

    cfg = X2SuperMimicVelocityTaskEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 43
    if args.device is not None:
        cfg.sim.device = args.device
    # Shorten intervals to exercise command changes and automatic resets.
    cfg.episode_length_s = 0.24
    cfg.commands.base_velocity.resampling_time_range = (0.1, 0.1)

    with launch_simulation(cfg, args):
        from isaaclab.managers import SceneEntityCfg
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.supermimic.terminations import velocity_task_fall
        from x2_style_mimic.algorithms.supermimic.heads import VelocityTask, VelocityTaskCfg

        env = X2SuperMimicEnv(cfg)
        try:
            observations, _ = env.reset()
            robot = env.scene['robot']
            task_cfg = VelocityTaskCfg(**cfg.velocity_task)
            task = VelocityTask(
                env.world_state_spec,
                robot.data.joint_pos_limits.torch[0],
                task_cfg,
            ).to(env.device)
            assert cfg.terminations.fallen.func is velocity_task_fall
            assert cfg.terminations.fallen.params['minimum_height'] == task_cfg.minimum_height
            assert cfg.terminations.fallen.params['maximum_tilt'] == task_cfg.maximum_tilt
            previous = env.command_manager.get_command('base_velocity').clone()
            resets = command_changes = 0
            asset_cfg = SceneEntityCfg('robot')
            observation_dim = 6 + 3 * env.world_state_spec.num_actions
            for _ in range(args.steps):
                state = env.read_world_state()
                actual = velocity_task_fall(
                    env, task_cfg.minimum_height, task_cfg.maximum_tilt, asset_cfg,
                )
                assert torch.equal(task.terminated(state), actual)
                action = torch.zeros(
                    env.num_envs, env.world_state_spec.num_actions, device=env.device,
                )
                observations, reward, terminated, truncated, extras = env.step(action)
                transition = extras['world_model']
                assert torch.equal(task.terminated(transition['next_state']), terminated)
                assert bool(torch.isfinite(transition['next_state']).all())
                assert not bool(reward.any())
                assert observations['policy'].shape == (env.num_envs, 5 * observation_dim)
                current = env.command_manager.get_command('base_velocity')
                assert current.shape == (env.num_envs, 3)
                command_changes += int((current != previous).any(dim=-1).sum())
                previous = current.clone()
                resets += int((terminated | truncated).sum())
            assert resets > 0 and command_changes > 0
            report = {
                'passed': True,
                'config_class': type(cfg).__name__,
                'termination_module': velocity_task_fall.__module__,
                'num_envs': env.num_envs,
                'steps': args.steps,
                'control_dt': env.step_dt,
                'policy_observation_dim': 5 * observation_dim,
                'resets': resets,
                'command_changes': command_changes,
                'zero_environment_reward': True,
                'termination_matches_task': True,
                'test_episode_length_s': cfg.episode_length_s,
                'test_command_interval_s': 0.1,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
        finally:
            env.close()


if __name__ == '__main__':
    main()
