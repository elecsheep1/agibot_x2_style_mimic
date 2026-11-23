#!/usr/bin/env python3
"""Train an X2 task through the Isaac Lab 3 unified RSL-RL entrypoint."""

from __future__ import annotations

import sys

import gymnasium as gym
import torch

# Register the project's Gym tasks before the unified CLI resolves ``--task``.
import x2_style_mimic  # noqa: F401
from isaaclab_rl.entrypoints.dispatch import run_train_cli


def _apply_x2_prone_compatibility(argv: list[str]) -> list[str]:
    """Add Isaac Lab 3 compatibility overrides for the prone MultiMotion task.

    The original self-contact sensor expands ``{ENV_REGEX_NS}`` to the Isaac
    Lab 3 expression ``env_[^/]+`` but its initialization code only recognizes
    the old ``env_.*`` spelling.  Resolving env_0 here is intentional: the
    sensor then derives each replicated environment path from that root.
    """
    task = None
    for index, argument in enumerate(argv):
        if argument == "--task" and index + 1 < len(argv):
            task = argv[index + 1]
        elif argument.startswith("--task="):
            task = argument.split("=", 1)[1]

    override_prefix = "env.scene.self_contact.prim_path="
    if task == "X2-Prone-MultiMotion-v0":
        _patch_prone_reset_event()
        if not any(arg.startswith(override_prefix) for arg in argv):
            return [*argv, f"{override_prefix}/World/envs/env_0/Robot"]
    return argv


def _patch_prone_reset_event() -> None:
    """Adapt the legacy motion-reset callback to Isaac Lab 3 slice env IDs."""
    from x2_style_mimic.tasks.manager_based.x2_style_mimic import x2_getup_env_cfg
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp import getup

    original = getup.reset_from_motion_dataset
    if getattr(original, "_isaaclab3_slice_compatible", False):
        return

    def reset_from_motion_dataset_compat(env, env_ids, *args, **kwargs):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        return original(env, env_ids, *args, **kwargs)

    reset_from_motion_dataset_compat._isaaclab3_slice_compatible = True
    def make_compatible_cfg():
        cfg = x2_getup_env_cfg.X2MultiMotionGetUpEnvCfg()
        cfg.events.reset_from_reference.func = reset_from_motion_dataset_compat
        return cfg

    # ``configclass`` fields are instance-owned in Isaac Lab 3. Replacing the
    # Gym registration's config factory lets the unified CLI receive a patched
    # instance without mutating the original project configuration class.
    gym.spec("X2-Prone-MultiMotion-v0").kwargs["env_cfg_entry_point"] = make_compatible_cfg


def main() -> int | None:
    """Dispatch to the official Isaac Lab RSL-RL training backend."""
    return run_train_cli(["--rl_library", "rsl_rl", *_apply_x2_prone_compatibility(sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
