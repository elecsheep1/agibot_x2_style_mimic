#!/usr/bin/env python3
"""Isaac Lab 3-safe RSL-RL launcher for the migrated X2 tasks."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from functools import wraps

from isaaclab.app import AppLauncher


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train a migrated X2 task with RSL-RL.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--logger", choices=("tensorboard", "wandb", "neptune"), default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_run", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--video_interval", type=int, default=2000)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()


def _patch_multimotion_reset() -> None:
    """Convert the Isaac Lab 3 initial ``slice(None)`` reset IDs to a tensor."""
    import torch
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_getup_env import X2MultiMotionGetUpEnv

    original = X2MultiMotionGetUpEnv._reset_idx
    if getattr(original, "_isaaclab3_slice_compatible", False):
        return

    @wraps(original)
    def reset_idx_compat(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        return original(self, env_ids)

    reset_idx_compat._isaaclab3_slice_compatible = True
    X2MultiMotionGetUpEnv._reset_idx = reset_idx_compat


def main() -> None:
    args_cli, hydra_args = _parse_args()
    # Hydra only consumes task overrides such as env.rewards.* and agent.algorithm.*.
    sys.argv = [sys.argv[0], *hydra_args]

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    env = None
    try:
        # This must remain below AppLauncher: importing X2 before Kit preloads pxr/USD.
        import gymnasium as gym
        import torch
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner

        import x2_style_mimic  # noqa: F401
        from isaaclab.envs import DirectMARLEnvCfg, multi_agent_to_single_agent
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import resolve_task_config

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False

        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.seed is not None:
            agent_cfg.seed = args_cli.seed
        if args_cli.max_iterations is not None:
            agent_cfg.max_iterations = args_cli.max_iterations
        if args_cli.run_name is not None:
            agent_cfg.run_name = args_cli.run_name
        if args_cli.experiment_name is not None:
            agent_cfg.experiment_name = args_cli.experiment_name
        if args_cli.logger is not None:
            agent_cfg.logger = args_cli.logger
        if args_cli.resume:
            agent_cfg.resume = True
        if args_cli.load_run is not None:
            agent_cfg.load_run = args_cli.load_run
        if args_cli.checkpoint is not None:
            agent_cfg.load_checkpoint = args_cli.checkpoint
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env_cfg.seed = agent_cfg.seed

        if args_cli.task == "X2-Prone-MultiMotion-v0":
            _patch_multimotion_reset()
            # The old sensor only recognizes env_0, not Isaac Lab 3's regex spelling.
            env_cfg.scene.self_contact.prim_path = "/World/envs/env_0/Robot"

        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root, log_dir)
        print(f"[INFO] Logging experiment in directory: {log_root}")
        env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            env = multi_agent_to_single_agent(env)
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=os.path.join(log_dir, "videos", "train"),
                step_trigger=lambda step: step % args_cli.video_interval == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )

        start_time = time.time()
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported RSL-RL runner: {agent_cfg.class_name}")

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"Training time: {time.time() - start_time:.2f} seconds")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
