#!/usr/bin/env python3
"""Train an X2 task through the Isaac Lab 3 unified RSL-RL entrypoint."""

from __future__ import annotations

import sys
from functools import wraps

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
    """Adapt the legacy MultiMotion environment to Isaac Lab 3 slice IDs."""
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_getup_env import X2MultiMotionGetUpEnv
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv

    def patch_reset_method(env_type) -> None:
        original_reset_idx = env_type._reset_idx
        if getattr(original_reset_idx, "_isaaclab3_slice_compatible", False):
            return

        @wraps(original_reset_idx)
        def reset_idx_compat(self, env_ids):
            if isinstance(env_ids, slice):
                env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
            return original_reset_idx(self, env_ids)

        reset_idx_compat._isaaclab3_slice_compatible = True
        env_type._reset_idx = reset_idx_compat

    # Convert the initial all-environment slice at each outer reset boundary.
    # This covers the motion dataset, assist controller, and terminal-state
    # bookkeeping that were originally written for tensor-only environment IDs.
    patch_reset_method(X2MultiMotionGetUpEnv)
    patch_reset_method(X2SuperMimicEnv)


def main() -> int | None:
    """Dispatch to the official Isaac Lab RSL-RL training backend."""
    return run_train_cli(["--rl_library", "rsl_rl", *_apply_x2_prone_compatibility(sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
