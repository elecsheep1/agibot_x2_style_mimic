#!/usr/bin/env python3
"""Run an X2 environment with zero actions under Isaac Lab 3.0."""

from __future__ import annotations

import argparse

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="X2 zero-action contact diagnostic for Isaac Lab 3.")
parser.add_argument("--task", default="Template-X2-Style-Mimic-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true")
# AppLauncher supplies the standard ``--visualizer`` / ``--viz`` option.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import x2_style_mimic  # noqa: F401, E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def main() -> None:
    """Print contact-body resolution and step the environment with zero actions."""
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"]

    print("\n===== X2 bodies =====")
    print("\n".join(f"{index:2d}: {name}" for index, name in enumerate(robot.body_names)))
    print("\n===== Contact sensor bodies =====")
    print("\n".join(f"{index:2d}: {name}" for index, name in enumerate(contact_sensor.body_names)))
    print("\n===== Contact force history shape =====")
    print(contact_sensor.data.net_forces_w_history.torch.shape)

    env.reset()
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    steps = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(actions)
        steps += 1
        if args_cli.headless and steps >= 100:
            break
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
