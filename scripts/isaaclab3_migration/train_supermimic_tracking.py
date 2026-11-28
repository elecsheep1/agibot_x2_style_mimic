#!/usr/bin/env python3
"""Run the legacy SuperMimic Tracking trainer on Isaac Lab 3.

The original entrypoint is preserved under ``scripts/supermimic_tracking``.
This wrapper supplies only the Isaac Lab 2 -> 3 launcher compatibility layer.
"""

from __future__ import annotations

import importlib.util
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import signal
import sys
import traceback
from pathlib import Path
import tomllib


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
TRACKING_ENTRYPOINT = PROJECT_ROOT / "scripts" / "supermimic_tracking" / "train.py"
DEFAULT_URDF = (
    Path.home()
    / "robotics_ws/agibot_model/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra.urdf"
)


def _load_legacy_entrypoint():
    spec = importlib.util.spec_from_file_location("x2_legacy_supermimic_tracking_train", TRACKING_ENTRYPOINT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load tracking trainer: {TRACKING_ENTRYPOINT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_config(legacy):
    """Parse the legacy CLI without invoking its obsolete launcher entrypoint."""
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description="SuperMimic Tracking (Isaac Lab 3 migration)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "scripts/supermimic_tracking/configs/train.toml")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--motion-manifest", type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--print-config", action="store_true")
    add_launcher_args(parser)
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(str(unknown))

    from x2_style_mimic.algorithms.supermimic_tracking.config import TrackingTrainCfg

    cfg = TrackingTrainCfg()
    legacy.update_config(cfg, tomllib.loads(args.config.read_text()))
    if args.evaluate and not args.resume:
        parser.error("验收需要 --resume 指定该入口保存的检查点")
    if args.evaluate:
        import torch

        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        legacy.update_config(cfg, checkpoint["config"])
    for name in ("num_envs", "iterations", "run_name", "device"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if args.resume:
        cfg.resume = str(args.resume.resolve())
    cfg.policy_trainer.obs_noise = cfg.obs_noise
    cfg.validate()
    return args, cfg


def _supply_remote_urdf(argv: list[str]) -> None:
    """Use the documented new-machine URDF location when it was not explicit."""
    if "--urdf" in argv or not DEFAULT_URDF.is_file():
        return
    argv.extend(("--urdf", str(DEFAULT_URDF)))
    print(f"[compat] Using migrated X2 URDF: {DEFAULT_URDF}", flush=True)


def _replace_null_manager_configs(env_cfg) -> None:
    """Translate Lab 2's ``None`` manager configs to empty Lab 3 configs.

    Isaac Lab 3 constructs the Event/Curriculum managers and resolves their
    terms even when a legacy environment set the corresponding config to
    ``None``.  An empty config object is the explicit Lab 3 representation of
    "no terms".
    """
    from isaaclab.utils.configclass import configclass

    @configclass
    class EmptyEventsCfg:
        pass

    @configclass
    class EmptyCurriculumCfg:
        pass

    if env_cfg.events is None:
        env_cfg.events = EmptyEventsCfg()
        print("[compat] Replaced legacy events=None with an empty Event config", flush=True)
    if env_cfg.curriculum is None:
        env_cfg.curriculum = EmptyCurriculumCfg()
        print("[compat] Replaced legacy curriculum=None with an empty Curriculum config", flush=True)


def main() -> None:
    if not TRACKING_ENTRYPOINT.is_file():
        raise FileNotFoundError(TRACKING_ENTRYPOINT)

    _supply_remote_urdf(sys.argv)

    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def report_sigint(_signum, frame):
        print("\n[interrupt] Received SIGINT; Python was executing:", file=sys.stderr)
        traceback.print_stack(frame, file=sys.stderr)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, report_sigint)
    try:
        legacy = _load_legacy_entrypoint()
        args, cfg = _parse_config(legacy)
        if args.print_config:
            print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
            return

        from x2_style_mimic.tasks.manager_based.x2_supermimic_tracking.env_cfg import (
            X2SuperMimicTrackingEnvCfg,
        )

        env_cfg = X2SuperMimicTrackingEnvCfg()
        env_cfg.scene.num_envs, env_cfg.seed, env_cfg.sim.device = cfg.num_envs, cfg.seed, cfg.device
        env_cfg.domain_randomization = cfg.domain_randomization and not args.evaluate
        env_cfg.reset_noise = cfg.reset_noise and not args.evaluate
        env_cfg.obs_noise = cfg.obs_noise and not args.evaluate
        env_cfg.actions.joint_pos.noise_std = 0.0 if args.evaluate else cfg.action_noise_std
        env_cfg.observations.policy.history_length = cfg.policy.history
        env_cfg.root_height_threshold = cfg.world_model.height_threshold
        env_cfg.root_tilt_threshold = cfg.world_model.tilt_threshold
        env_cfg.commands.motion.future_frames = cfg.policy_trainer.future_frames
        env_cfg.commands.motion.sampling_mode = "start" if args.evaluate else "return"
        _replace_null_manager_configs(env_cfg)
        if args.motion_manifest:
            env_cfg.motion_manifest = str(args.motion_manifest.resolve())
        if args.urdf:
            env_cfg.urdf = str(args.urdf.resolve())
        if not Path(env_cfg.urdf).is_file():
            raise FileNotFoundError(f"找不到 X2 URDF：{env_cfg.urdf}")
        if args.evaluate:
            import numpy as np

            manifest = Path(env_cfg.motion_manifest)
            entries = json.loads(manifest.read_text())["motions"]
            longest = max(
                len(np.load(manifest.parent / entry["file"], allow_pickle=False)["time_s"])
                for entry in entries
            )
            env_cfg.episode_length_s = (longest + 1) * env_cfg.sim.dt * env_cfg.decimation
        output = PROJECT_ROOT / "logs/supermimic_tracking" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{cfg.run_name}"

        # Arguments have been fully parsed.  Prevent Kit from parsing this
        # training CLI a second time during startup.
        sys.argv = [sys.argv[0]]
        # Isaac Lab 3 chooses and owns the simulator lifecycle from the
        # backend configuration.  Do not recreate the retired Lab 2
        # ``AppLauncher`` wrapper here: it can shut Kit down before the
        # standalone workflow reaches its environment construction.
        from isaaclab.app import launch_simulation

        print("[compat] Starting official Isaac Lab 3 simulation lifecycle", flush=True)
        with launch_simulation(env_cfg, args):
            print("[compat] Kit started; importing Tracking environment", flush=True)
            import random
            import numpy as np
            import torch
            from x2_style_mimic.tasks.manager_based.x2_supermimic_tracking.env import (
                X2SuperMimicTrackingEnv,
            )
            from x2_style_mimic.algorithms.supermimic_tracking.training import (
                TrackingTraining,
                evaluate,
            )

            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            print(f"[compat] Creating Tracking environment ({cfg.num_envs} environments)", flush=True)
            env = X2SuperMimicTrackingEnv(env_cfg)
            try:
                print("[compat] Tracking environment ready; starting workflow", flush=True)
                if args.evaluate:
                    evaluate(env, cfg, args.resume, output)
                else:
                    TrackingTraining(env, cfg, output).run()
            finally:
                env.close()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    main()
