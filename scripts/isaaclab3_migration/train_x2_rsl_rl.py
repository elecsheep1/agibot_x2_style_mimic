#!/usr/bin/env python3
"""Train an X2 task through the Isaac Lab 3 unified RSL-RL entrypoint."""

from __future__ import annotations

import sys

# Register the project's Gym tasks before the unified CLI resolves ``--task``.
import x2_style_mimic  # noqa: F401
from isaaclab_rl.entrypoints.dispatch import run_train_cli


def main() -> int | None:
    """Dispatch to the official Isaac Lab RSL-RL training backend."""
    return run_train_cli(["--rl_library", "rsl_rl", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
