#!/usr/bin/env python3
"""Launch the legacy SuperMimic get-up trainer on Isaac Lab 3.

The original entrypoint imports ``add_launcher_args`` from an Isaac Lab 2-era
location.  Keep that compatibility shim here so the original SuperMimic files
remain untouched.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPERMIMIC_SCRIPTS = PROJECT_ROOT / "scripts" / "supermimic"


@contextmanager
def _launch_simulation_compat(_env_cfg, args_cli):
    """Provide the removed Isaac Lab 2 ``launch_simulation`` context manager."""
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    try:
        yield
    finally:
        app_launcher.app.close()


def _setup_preset_cli_compat(parser):
    """Parse known launcher arguments when preset support is unavailable."""
    return parser.parse_known_args()


def main() -> None:
    """Install the launcher compatibility shim and run the existing trainer."""
    # The SuperMimic scripts use sibling imports (``from getup_config import``).
    sys.path.insert(0, str(SUPERMIMIC_SCRIPTS))

    from isaaclab.app import AppLauncher
    import isaaclab_tasks.utils as task_utils

    # Isaac Lab 3 moved/removed these Isaac Lab 2 helpers.  The SuperMimic
    # runtime only needs ordinary argument parsing and a Kit lifecycle.
    task_utils.add_launcher_args = AppLauncher.add_app_launcher_args
    task_utils.launch_simulation = _launch_simulation_compat
    if not hasattr(task_utils, "setup_preset_cli"):
        task_utils.setup_preset_cli = _setup_preset_cli_compat

    from getup_config import handle_meta_arguments

    if handle_meta_arguments(sys.argv[1:]):
        return

    from getup_runtime import run_training

    run_training()


if __name__ == "__main__":
    main()
