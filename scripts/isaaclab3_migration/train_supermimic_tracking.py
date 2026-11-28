#!/usr/bin/env python3
"""Run the legacy SuperMimic Tracking trainer on Isaac Lab 3.

The original entrypoint is preserved under ``scripts/supermimic_tracking``.
This wrapper supplies only the Isaac Lab 2 -> 3 launcher compatibility layer.
"""

from __future__ import annotations

import importlib.util
import signal
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
TRACKING_ENTRYPOINT = PROJECT_ROOT / "scripts" / "supermimic_tracking" / "train.py"
DEFAULT_URDF = (
    Path.home()
    / "robotics_ws/agibot_model/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra.urdf"
)


@contextmanager
def _launch_simulation_compat(_env_cfg, args_cli):
    """Replacement for Isaac Lab 2's removed ``launch_simulation`` helper."""
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def report_sigint(_signum, frame):
        print("\n[interrupt] Received SIGINT after Kit launch; Python was executing:", file=sys.stderr)
        traceback.print_stack(frame, file=sys.stderr)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, report_sigint)
    try:
        yield
    finally:
        try:
            app_launcher.app.close()
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)


def _setup_preset_cli_compat(parser):
    """Tracking owns its TOML config; do not apply unrelated task presets."""
    return parser.parse_known_args()


def _load_legacy_entrypoint():
    spec = importlib.util.spec_from_file_location("x2_legacy_supermimic_tracking_train", TRACKING_ENTRYPOINT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load tracking trainer: {TRACKING_ENTRYPOINT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _supply_remote_urdf(argv: list[str]) -> None:
    """Use the documented new-machine URDF location when it was not explicit."""
    if "--urdf" in argv or not DEFAULT_URDF.is_file():
        return
    argv.extend(("--urdf", str(DEFAULT_URDF)))
    print(f"[compat] Using migrated X2 URDF: {DEFAULT_URDF}", flush=True)


def main() -> None:
    if not TRACKING_ENTRYPOINT.is_file():
        raise FileNotFoundError(TRACKING_ENTRYPOINT)

    _supply_remote_urdf(sys.argv)

    # The original module imports these from an Isaac Lab 2 location inside
    # ``main``.  Patch that public module before dispatching to it.
    from isaaclab.app import AppLauncher
    import isaaclab_tasks.utils as task_utils

    task_utils.add_launcher_args = AppLauncher.add_app_launcher_args
    task_utils.setup_preset_cli = _setup_preset_cli_compat
    task_utils.launch_simulation = _launch_simulation_compat

    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def report_sigint(_signum, frame):
        print("\n[interrupt] Received SIGINT; Python was executing:", file=sys.stderr)
        traceback.print_stack(frame, file=sys.stderr)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, report_sigint)
    try:
        _load_legacy_entrypoint().main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    main()
