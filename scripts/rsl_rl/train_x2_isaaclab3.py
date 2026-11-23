#!/usr/bin/env python3
"""Train an X2 task through the Isaac Lab 3 unified RSL-RL entrypoint.

Importing :mod:`x2_style_mimic` registers the project's Gym tasks before the
Isaac Lab CLI resolves ``--task``.  This replaces the legacy local
``train.py`` launcher, whose launcher-argument API was removed in Isaac Lab
3.0.
"""

from __future__ import annotations

import sys

# Register Template-X2-Style-Mimic-v0 and other project tasks in this process.
import x2_style_mimic  # noqa: F401
from isaaclab_rl.entrypoints.dispatch import run_train_cli


def main() -> int | None:
    """Dispatch to the official Isaac Lab RSL-RL training backend."""
    return run_train_cli(["--rl_library", "rsl_rl", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
