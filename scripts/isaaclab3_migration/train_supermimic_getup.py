#!/usr/bin/env python3
"""Launch the legacy SuperMimic get-up trainer on Isaac Lab 3.

The original entrypoint imports ``add_launcher_args`` from an Isaac Lab 2-era
location.  Keep that compatibility shim here so the original SuperMimic files
remain untouched.
"""

from __future__ import annotations

import sys
import signal
import traceback
from contextlib import contextmanager
from functools import wraps
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


def _patch_supermimic_reset_chain(env) -> None:
    """Normalize Isaac Lab 3's initial ``slice(None)`` reset IDs.

    The legacy X2 get-up events, AMP bookkeeping, and SuperMimic transition
    capture expect a tensor of environment IDs.  Isaac Lab 3 calls the first
    reset with a slice, which must be converted at every custom reset boundary.
    """
    import torch

    for env_type in type(env).__mro__:
        original = env_type.__dict__.get("_reset_idx")
        if original is None or getattr(original, "_isaaclab3_slice_compatible", False):
            continue

        @wraps(original)
        def reset_idx_compat(self, env_ids, _original=original):
            if isinstance(env_ids, slice):
                env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
            return _original(self, env_ids)

        reset_idx_compat._isaaclab3_slice_compatible = True
        setattr(env_type, "_reset_idx", reset_idx_compat)


def _install_supermimic_runtime_compat(runtime_module) -> None:
    """Patch the environment immediately before its first legacy reset."""
    original_build_runtime = runtime_module.build_runtime
    if getattr(original_build_runtime, "_isaaclab3_compatible", False):
        return

    @wraps(original_build_runtime)
    def build_runtime_compat(env, args, resume):
        _patch_supermimic_reset_chain(env)
        if env.num_envs != args.num_envs:
            raise RuntimeError(
                f"SuperMimic environment count mismatch: config requested {args.num_envs}, "
                f"but Isaac Lab created {env.num_envs}."
            )
        print(f"[compat] SuperMimic running with {env.num_envs} environments", flush=True)
        return original_build_runtime(env, args, resume)

    build_runtime_compat._isaaclab3_compatible = True
    runtime_module.build_runtime = build_runtime_compat


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
    # The current helper applies Isaac Lab task presets and overwrites the
    # SuperMimic TOML's ``num_envs`` (32) with 4096.  This trainer does not use
    # Isaac Lab presets, so always use a plain parse-known-args implementation.
    task_utils.setup_preset_cli = _setup_preset_cli_compat

    from getup_config import handle_meta_arguments

    if handle_meta_arguments(sys.argv[1:]):
        return

    import getup_runtime

    _install_supermimic_runtime_compat(getup_runtime)
    run_training = getup_runtime.run_training

    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def report_sigint(_signum, frame):
        """Expose interrupts that would otherwise look like a silent Kit exit."""
        print("\n[interrupt] Received SIGINT; Python was executing:", file=sys.stderr)
        traceback.print_stack(frame, file=sys.stderr)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, report_sigint)
    try:
        run_training()
    except BaseException:
        # SimulationApp shutdown can obscure exceptions from the custom runner.
        traceback.print_exc()
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    main()
