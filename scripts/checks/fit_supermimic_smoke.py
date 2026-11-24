"""Fit a small dynamics model offline; report held-out-episode errors, never update actor.

No Isaac Sim startup is needed. A successful fit is an engineering smoke test,
not evidence that the model is ready for differentiable policy optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import torch


def load_modules():
    # Import the pure PyTorch modules without executing the package's Gym/Isaac registration.
    directory = Path(__file__).resolve().parents[2] / "source/x2_style_mimic/x2_style_mimic/algorithms/supermimic"
    root_package = "_supermimic_offline"
    algorithm_package = f"{root_package}.algorithms.supermimic"
    package_paths = {
        root_package: directory.parent.parent,
        f"{root_package}.algorithms": directory.parent,
        algorithm_package: directory,
    }
    for name, path in package_paths.items():
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package

    state = importlib.import_module(algorithm_package + ".state")
    dynamics = importlib.import_module(algorithm_package + ".world_model")
    return state, dynamics


def valid_windows(data, horizon):
    """Return [start time, environment] pairs; the last transition may terminate."""
    steps, num_envs = data["terminated"].shape
    if steps < horizon:
        raise ValueError("Dataset shorter than requested horizon")
    count = steps - horizon + 1
    mask = torch.ones(count, num_envs, dtype=torch.bool)
    episode = data["episode_ids"][:count]
    first_step = data["step_ids"][:count]
    for offset in range(horizon):
        mask &= data["episode_ids"][offset:offset + count] == episode
        mask &= data["step_ids"][offset:offset + count] == first_step + offset
        if offset < horizon - 1:
            mask &= ~(data["terminated"][offset:offset + count] | data["truncated"][offset:offset + count])
    return mask.nonzero()


def sample(data, candidates, batch_size, horizon, device):
    chosen = candidates[torch.randint(len(candidates), (batch_size,))]
    start, env = chosen.unbind(-1)
    time = start[:, None] + torch.arange(horizon)
    envs = env[:, None].expand_as(time)
    states = torch.cat((data["state"][start, env, None], data["next_state"][time, envs]), dim=1)
    return {"states": states.to(device), "actions": data["action"][time, envs].to(device),
            "context": data["contexts"][env].to(device)}


def sample_training_batch(data, candidates, batch_size, horizon, device, early_fraction):
    if early_fraction == 0.0:
        return sample(data, candidates, batch_size, horizon, device)
    if "episode_steps" not in data or "step_dt" not in data["metadata"]:
        raise ValueError("Early sampling requires episode_steps and metadata.step_dt")
    elapsed = data["episode_steps"][candidates[:, 0], candidates[:, 1]] * data["metadata"]["step_dt"]
    early = elapsed < 3.0
    early_count = round(batch_size * early_fraction)
    parts = []
    for pool, count in ((candidates[early], early_count), (candidates[~early], batch_size - early_count)):
        if count:
            if not len(pool):
                raise ValueError("Requested early/late sampling pool is empty")
            parts.append(sample(data, pool, count, horizon, device))
    return {key: torch.cat([part[key] for part in parts], dim=0) for key in parts[0]}


def consistency_residuals(states, spec, dt, state_module, dynamics):
    """Finite-difference rates minus average endpoint velocities, in physical units."""
    if dt <= 0 or states.ndim != 3 or states.shape[1] < 2:
        raise ValueError("Consistency requires positive dt and [B,H+1,Ds] states")
    before = state_module.unpack_state(states[:, :-1].reshape(-1, spec.dim), spec)
    after = state_module.unpack_state(states[:, 1:].reshape(-1, spec.dim), spec)
    q0 = dynamics.unit_quat(before["root_pose"][:, 3:])
    q1 = dynamics.unit_quat(after["root_pose"][:, 3:])
    inverse = torch.cat((-q0[:, :3], q0[:, 3:]), dim=-1)
    relative = dynamics.unit_quat(dynamics.quat_mul(q1, inverse))
    relative = torch.where(relative[:, 3:] < 0, -relative, relative)
    # The clamped norm gives a finite identity derivative for the quaternion log.
    norm = relative[:, :3].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angle = 2.0 * torch.atan2(norm, relative[:, 3:])
    angular_rate = relative[:, :3] * (angle / norm) / dt
    velocity = (before["root_velocity"] + after["root_velocity"]) * 0.5
    joint_velocity = (before["joint_velocity"] + after["joint_velocity"]) * 0.5
    return {
        "root_linear": (after["root_pose"][:, :3] - before["root_pose"][:, :3]) / dt - velocity[:, :3],
        "root_angular": angular_rate - velocity[:, 3:],
        "joint_velocity": (after["joint_position"] - before["joint_position"]) / dt - joint_velocity,
    }


def soft_consistency_loss(states, spec, dt, margins, state_module, dynamics):
    residuals = consistency_residuals(states, spec, dt, state_module, dynamics)
    scales = {"root_linear": 0.5, "root_angular": 1.0, "joint_velocity": 2.0}
    terms = {}
    for name, residual in residuals.items():
        excess = (residual.abs() - margins[name]).clamp_min(0.0) / scales[name]
        terms[name] = torch.nn.functional.smooth_l1_loss(excess, torch.zeros_like(excess))
    return torch.stack(list(terms.values())).sum(), terms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("logs/supermimic/smoke_v1/transitions.pt"))
    parser.add_argument("--output", type=Path, default=Path("logs/supermimic/fit_v1"))
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-horizons", type=int, nargs="+", default=[8],
                        help="Cycle through these autoregressive training horizons")
    parser.add_argument("--early-fraction", type=float, default=0.0,
                        help="0 keeps uniform sampling; positive values set the fraction from episode time <3s, with the rest >=3s")
    parser.add_argument("--consistency-weight", type=float, default=0.0,
                        help="Weight for rate consistency with training-data-calibrated deadzones")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--reference-run", type=Path,
                        help="Reuse this run's episode split and input statistics; seed sampling independently of network width")
    parser.add_argument("--refit-input-stats", action="store_true",
                        help="配合 reference-run 只沿用回合划分，按当前世界模型特征重新计算输入统计量")
    args = parser.parse_args()
    if args.refit_input_stats and args.reference_run is None:
        parser.error("refit-input-stats 必须与 reference-run 一起使用")
    if args.output.exists():
        parser.error("Output already exists; choose a fresh directory")
    if args.updates < 1 or args.batch_size < 1:
        parser.error("updates and batch-size must be positive")
    if min(args.train_horizons) < 1 or not 0.0 <= args.early_fraction <= 1.0:
        parser.error("Training horizons must be positive and early-fraction must be in [0,1]")
    if not 0.0 <= args.consistency_weight < float("inf"):
        parser.error("consistency-weight must be finite and nonnegative")
    if min(args.hidden_dims) < 1:
        parser.error("hidden-dims must be positive")
    torch.manual_seed(args.seed)
    # This file is produced locally by collect_supermimic_smoke.py.
    data = torch.load(args.data, map_location="cpu", weights_only=True)
    state, dynamics = load_modules()
    spec = state.StateSpec(**data["metadata"]["state_spec"])
    timing = (dict(dt=data['metadata']['step_dt'], decimation=data['metadata']['decimation'])
              if hasattr(dynamics.WorldModel, 'DYNAMICS_MODE') else {})
    model = dynamics.WorldModel(spec, data["contexts"].shape[-1],
                               dynamics.TargetLimiter(**data["limiter"]),
                               hidden_dims=tuple(args.hidden_dims), **timing).to(args.device)
    # Split whole (environment, episode) keys BEFORE constructing windows or normalization.
    keys = torch.stack((torch.arange(data["episode_ids"].shape[1]).expand_as(data["episode_ids"]),
                        data["episode_ids"]), dim=-1)
    unique, inverse = torch.unique(keys.reshape(-1, 2), dim=0, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("Need at least two distinct episodes")
    holdout = torch.zeros(len(unique), dtype=torch.bool)
    reference_checkpoint = None
    if args.reference_run is None:
        holdout[torch.randperm(len(unique))[:max(1, len(unique) // 5)]] = True
    else:
        reference_report = json.loads((args.reference_run / "report.json").read_text())
        reference_checkpoint = torch.load(args.reference_run / "world_model.pt", map_location="cpu", weights_only=True)
        if Path(reference_report["data"]).resolve() != args.data.resolve() or reference_checkpoint["metadata"] != data["metadata"]:
            raise ValueError("Reference run must use the same dataset and metadata")
        training_keys = set(map(tuple, reference_report["train_episodes"]))
        validation_keys = set(map(tuple, reference_report["validation_episodes"]))
        if training_keys & validation_keys or training_keys | validation_keys != set(map(tuple, unique.tolist())):
            raise ValueError("Reference split must partition all dataset episodes")
        holdout = torch.tensor([tuple(key) in validation_keys for key in unique.tolist()], dtype=torch.bool)
        if not bool(holdout.any()) or bool(holdout.all()):
            raise ValueError("Reference split requires both training and validation episodes")
        for name, value in data["limiter"].items():
            torch.testing.assert_close(reference_checkpoint["limiter"][name], value, atol=0, rtol=0)
    validation_mask = holdout[inverse].reshape(data["episode_ids"].shape)

    split_cache = {}

    def split(horizon):
        if horizon in split_cache:
            return split_cache[horizon]
        candidates = valid_windows(data, horizon)
        is_validation = validation_mask[candidates[:, 0], candidates[:, 1]]
        train, validation = candidates[~is_validation], candidates[is_validation]
        if not len(train) or not len(validation):
            raise ValueError(f"Missing train/validation windows for horizon {horizon}")
        split_cache[horizon] = (train, validation)
        return train, validation

    train, validation = split(8)
    if reference_checkpoint is None or args.refit_input_stats:
        model.fit_input_stats(sample(data, train, min(2048, len(train)), 8, args.device))
    else:
        if model.input_mean.shape != reference_checkpoint["model"]["input_mean"].shape:
            raise ValueError("世界模型特征维度已改变；保留回合划分时请增加 --refit-input-stats")
        model.input_mean.copy_(reference_checkpoint["model"]["input_mean"].to(args.device))
        model.input_std.copy_(reference_checkpoint["model"]["input_std"].to(args.device))
        if hasattr(model, 'NORMALIZATION_MODE'):
            for name in ('state_normalizer', 'action_normalizer'):
                prefix = name + '.'
                saved_stats = {key[len(prefix):]: value for key, value in reference_checkpoint['model'].items()
                               if key.startswith(prefix)}
                if not saved_stats:
                    raise ValueError('参考模型缺少运行归一化统计量，请增加 --refit-input-stats 重新初始化')
                getattr(model, name).load_state_dict(saved_stats, strict=True)
        model.stats_ready.fill_(True)
        torch.manual_seed(args.seed)
    consistency_margins = None
    if args.consistency_weight:
        # Calibration uses training episodes only and preserves the sampling RNG.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            calibration = sample_training_batch(data, train, 2048, 8, args.device, args.early_fraction)
            residuals = consistency_residuals(calibration["states"], spec, data["metadata"]["step_dt"], state, dynamics)
            consistency_margins = {name: float(value.abs().flatten().quantile(0.95)) for name, value in residuals.items()}
    optimizer_path = Path(dynamics.__file__).with_name('optimizers.py')
    if optimizer_path.is_file():
        build_optimizers = importlib.import_module(dynamics.__package__ + '.optimizers').build_optimizers
        optimizers = build_optimizers(model, lr=2e-4)
        optimizer_mode = 'muon_adam_v1'
    else:
        optimizers = [torch.optim.Adam(model.parameters(), lr=2e-4)]
        optimizer_mode = 'adam_single_v1'

    def physical_metrics(predicted, target):
        p = state.unpack_state(predicted, spec)
        t = state.unpack_state(target, spec)

        def rmse(a, b):
            return float((a - b).square().mean().sqrt())

        return dict(
            root_position_rmse_m=rmse(p["root_pose"][:, :3], t["root_pose"][:, :3]),
            root_rotation_mean_rad=float(dynamics.rotation_error(
                p["root_pose"][:, 3:], t["root_pose"][:, 3:]).mean()),
            root_linear_rmse_m_s=rmse(p["root_velocity"][:, :3], t["root_velocity"][:, :3]),
            root_angular_rmse_rad_s=rmse(p["root_velocity"][:, 3:], t["root_velocity"][:, 3:]),
            joint_position_rmse_rad=rmse(p["joint_position"], t["joint_position"]),
            joint_velocity_rmse_rad_s=rmse(p["joint_velocity"], t["joint_velocity"]),
        )

    @torch.no_grad()
    def evaluate():
        result = {}
        baseline = {}
        early_result = {}
        early_baseline = {}
        for horizon in (1, 8, 32):
            _, candidates = split(horizon)
            groups = [("all", candidates)]
            # Compare transition dynamics separately from long standing intervals.
            if "episode_steps" in data:
                episode_time = data["episode_steps"][candidates[:, 0], candidates[:, 1]]
                early = episode_time * data["metadata"]["step_dt"] < 3.0
                if bool(early.any()):
                    groups.append(("first_3_seconds", candidates[early]))
            for name, group in groups:
                batch = sample(data, group, 256, horizon, args.device)
                pred = dynamics.rollout_logged_actions(
                    model, batch["states"][:, 0], batch["actions"], batch["context"])
                metrics = physical_metrics(pred[:, -1], batch["states"][:, -1])
                fixed_metrics = physical_metrics(batch["states"][:, 0], batch["states"][:, -1])
                if name == "all":
                    result[str(horizon)] = metrics
                    baseline[str(horizon)] = fixed_metrics
                else:
                    early_result[str(horizon)] = metrics
                    early_baseline[str(horizon)] = fixed_metrics
        return result, baseline, early_result, early_baseline

    # Reuse a seeded validation draw for before/after comparisons.
    rng = torch.random.get_rng_state()
    sampling_rng_sha256 = hashlib.sha256(rng.numpy().tobytes()).hexdigest()
    before, baseline, early_before, early_baseline = evaluate()
    torch.random.set_rng_state(rng)
    for update in range(args.updates):
        horizon = args.train_horizons[update % len(args.train_horizons)]
        train_candidates, _ = split(horizon)
        batch = sample_training_batch(data, train_candidates, args.batch_size, horizon, args.device, args.early_fraction)
        pred = dynamics.rollout_logged_actions(model, batch["states"][:, 0], batch["actions"], batch["context"])
        loss, _ = dynamics.dynamics_loss(pred, batch["states"], spec)
        if args.consistency_weight:
            consistency, _ = soft_consistency_loss(
                pred, spec, data["metadata"]["step_dt"], consistency_margins, state, dynamics)
            loss = loss + args.consistency_weight * consistency
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite dynamics loss")
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        for optimizer in optimizers:
            optimizer.step()
        if (update + 1) % 50 == 0:
            print(f"[WorldModel] {update + 1}/{args.updates} horizon={horizon} loss={float(loss.detach()):.6f}", flush=True)
    torch.random.set_rng_state(rng)
    after, _, early_after, _ = evaluate()
    report = dict(data=str(args.data.resolve()), updates=args.updates, seed=args.seed,
                  hidden_dims=args.hidden_dims, parameter_count=sum(p.numel() for p in model.parameters()),
                  reference_run=str(args.reference_run.resolve()) if args.reference_run else None,
                  sampling_rng_sha256=sampling_rng_sha256,
                  consistency_weight=args.consistency_weight, consistency_margins=consistency_margins,
                  consistency_rate_scales={"root_linear": 0.5, "root_angular": 1.0, "joint_velocity": 2.0},
                  training_horizons=args.train_horizons, early_fraction=args.early_fraction,
                  trained_transition_steps=args.batch_size * sum(args.train_horizons[i % len(args.train_horizons)] for i in range(args.updates)),
                  train_episodes=unique[~holdout].tolist(), validation_episodes=unique[holdout].tolist(),
                  train_windows_8=len(train), validation_windows_8=len(validation), before=before, after=after,
                  fixed_state_baseline=baseline,
                  first_3_seconds=dict(before=early_before, after=early_after, fixed_state_baseline=early_baseline),
                  action_source=data["metadata"].get("action_source", "unspecified"),
                  note="Held-out episodes; errors are end-of-horizon component RMSE; first_3_seconds uses episode elapsed time, including RSI births")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(model=model.state_dict(), metadata=data["metadata"], hidden_dims=tuple(args.hidden_dims),
                    dynamics_mode=getattr(model, 'DYNAMICS_MODE', 'independent_delta_v1'),
                    normalization_mode=getattr(model, 'NORMALIZATION_MODE', 'fixed_input_v1'),
                    optimizer_mode=optimizer_mode,
                    consistency_weight=args.consistency_weight, consistency_margins=consistency_margins,
                    training_horizons=args.train_horizons, early_fraction=args.early_fraction,
                    limiter=data["limiter"], context_dim=data["contexts"].shape[-1]), args.output / "world_model.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
