#!/usr/bin/env python3
"""Train an X2 task through the Isaac Lab 3 unified RSL-RL entrypoint."""

from __future__ import annotations

import sys

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
    if task == "X2-Prone-MultiMotion-v0" and not any(arg.startswith(override_prefix) for arg in argv):
        return [*argv, f"{override_prefix}/World/envs/env_0/Robot"]
    return argv


def main() -> int | None:
    """Dispatch to the official Isaac Lab RSL-RL training backend."""
    return run_train_cli(["--rl_library", "rsl_rl", *_apply_x2_prone_compatibility(sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
