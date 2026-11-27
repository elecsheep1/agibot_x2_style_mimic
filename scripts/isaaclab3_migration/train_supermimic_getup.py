#!/usr/bin/env python3
"""Launch the legacy SuperMimic get-up trainer on Isaac Lab 3.

The original entrypoint imports ``add_launcher_args`` from an Isaac Lab 2-era
location.  Keep that compatibility shim here so the original SuperMimic files
remain untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPERMIMIC_SCRIPTS = PROJECT_ROOT / "scripts" / "supermimic"


def main() -> None:
    """Install the launcher compatibility shim and run the existing trainer."""
    # The SuperMimic scripts use sibling imports (``from getup_config import``).
    sys.path.insert(0, str(SUPERMIMIC_SCRIPTS))

    from isaaclab.app import AppLauncher
    import isaaclab_tasks.utils as task_utils

    # Isaac Lab 3 moved this helper onto AppLauncher.  The other two helpers
    # used by getup_runtime still live in isaaclab_tasks.utils.
    task_utils.add_launcher_args = AppLauncher.add_app_launcher_args

    from getup_config import handle_meta_arguments

    if handle_meta_arguments(sys.argv[1:]):
        return

    from getup_runtime import run_training

    run_training()


if __name__ == "__main__":
    main()
