#!/usr/bin/env python3
"""Launch the SuperMimic imagination audit on Isaac Lab 3.

Reuse the launcher and reset shims from ``train_supermimic_getup.py`` so the
original audit script in ``scripts/checks`` stays untouched.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SUPERMIMIC_SCRIPTS = PROJECT_ROOT / "scripts" / "supermimic"
CHECKS = PROJECT_ROOT / "scripts" / "checks"


def main() -> None:
    """Install the Isaac Lab 3 compatibility shim and run the existing audit."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(SUPERMIMIC_SCRIPTS))
    sys.path.insert(0, str(CHECKS))

    from isaaclab.app import AppLauncher
    import isaaclab_tasks.utils as task_utils

    from train_supermimic_getup import (
        _install_supermimic_runtime_compat,
        _launch_simulation_compat,
        _setup_preset_cli_compat,
    )

    # The audit imports these three names inside main(); install them first.
    task_utils.add_launcher_args = AppLauncher.add_app_launcher_args
    task_utils.launch_simulation = _launch_simulation_compat
    task_utils.setup_preset_cli = _setup_preset_cli_compat

    import getup_runtime

    _install_supermimic_runtime_compat(getup_runtime)

    from audit_getup_imagination import main as audit_main

    try:
        audit_main()
    except Exception:
        # SimulationApp shutdown can obscure exceptions from the audit.
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
