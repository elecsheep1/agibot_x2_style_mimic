"""Utilities for a fresh-discriminator branch; no simulator imports at module load."""

import json
import math
from pathlib import Path

import torch


def reset_discriminator(algorithm):
    """Reset network/Adam state only, preserving loaded normalization and PPO."""
    required = ("amp_discriminator", "amp_buffer", "amp_num_steps", "amp_reference")
    if not all(hasattr(algorithm, key) for key in required):
        raise TypeError("This entry requires AMPWindowPPO")
    discriminator = algorithm.amp_discriminator
    if not isinstance(discriminator.optimizer, torch.optim.Adam):
        raise TypeError("Expected the AMP discriminator's Adam optimizer")
    with torch.no_grad():
        for module in discriminator.network.modules():
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
    discriminator.optimizer.state.clear()
    discriminator.optimizer.zero_grad(set_to_none=True)
    # Keep the configured discriminator LR, not any historical optimizer override.
    for group in discriminator.optimizer.param_groups:
        group["lr"] = float(algorithm.amp_cfg["learning_rate"])
    algorithm.amp_buffer.clear()
    if algorithm.amp_history is not None:
        algorithm.amp_history.clear()
    algorithm._amp_sample_history = None
    print("[AMP reset] Network and Adam state reset; PPO and reference statistics preserved.", flush=True)


def warmup_discriminator(runner, iterations, log_path):
    """Collect stochastic frozen-policy rollouts and update only the discriminator.

    Uses real pre-reset terminal AMP states and the standard history implementation.
    Does not call PPO act/process/update, update observation statistics, or advance
    the PPO iteration counter. Warmup samples remain in the fresh replay pool.
    """
    if iterations == 0:
        return
    from x2_style_mimic.algorithms.amp.history import AMPHistory

    alg = runner.alg
    if alg.amp_cfg.get("policy_sample_gate_enabled", False):
        raise ValueError("Frozen-policy warmup requires policy_sample_gate_enabled=false")
    batch_size = int(alg.amp_cfg["batch_size"])
    updates = int(alg.amp_cfg["updates_per_iteration"])
    if batch_size <= 0 or updates <= 0:
        raise ValueError("Discriminator batch size and update count must be positive")
    obs = runner.env.get_observations().to(runner.device)
    history = AMPHistory(obs["amp"].shape[0], obs["amp"].shape[-1], alg.amp_num_steps, runner.device)
    alg.eval_mode()  # Also freezes observation-normalization running statistics.
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            for iteration in range(iterations):
                with torch.no_grad():
                    for _ in range(runner.cfg["num_steps_per_env"]):
                        history.begin(obs["amp"])
                        actions = alg.actor(obs, stochastic_output=True)
                        obs, _, dones, extras = runner.env.step(actions.to(runner.env.device))
                        obs = obs.to(runner.device)
                        dones = dones.to(runner.device).bool()
                        terminal = extras["amp_terminal_mask"].to(runner.device).bool()
                        if not torch.equal(terminal, dones):
                            raise RuntimeError("AMP terminal mask differs from environment dones")
                        windows, valid = history.advance(extras["amp_next_state"].to(runner.device), dones)
                        alg.amp_buffer.add(windows, valid)
                totals = {}
                completed = 0
                if alg.amp_buffer.size:
                    for _ in range(updates):
                        reference = alg.amp_reference.sample_amp_windows(
                            batch_size=batch_size, num_steps=alg.amp_num_steps,
                        )["states"]
                        result = alg.amp_discriminator.update(
                            reference_pairs=reference, policy_pairs=alg.amp_buffer.sample(batch_size),
                        )
                        for key, value in result.items():
                            totals[key] = totals.get(key, 0.0) + value
                        completed += 1
                record = {key: value / completed for key, value in totals.items()}
                record.update(warmup_iteration=iteration + 1, updates=completed, buffer_size=alg.amp_buffer.size)
                log.write(json.dumps(record) + "\n")
                log.flush()
                if iteration == 0 or (iteration + 1) % 10 == 0 or iteration + 1 == iterations:
                    print(f"[AMP warmup] {iteration + 1}/{iterations}: {record}", flush=True)
    finally:
        alg.train_mode()
    if not alg.amp_buffer.size:
        raise RuntimeError("Warmup produced no valid AMP windows")


def reset_actor_std(algorithm, value):
    """Reset Gaussian exploration and only its associated Adam state."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError("resume std must be finite and positive")
    distribution = algorithm.actor.distribution
    if distribution.std_type == "scalar" and hasattr(distribution, "std_param"):
        parameter = distribution.std_param
        previous_std = parameter.detach().clone()
        target = value
    elif distribution.std_type == "log" and hasattr(distribution, "log_std_param"):
        parameter = distribution.log_std_param
        previous_std = parameter.detach().exp().clone()
        target = math.log(value)
    else:
        raise TypeError("Exploration reset requires a state-independent scalar/log Gaussian")
    if not bool(torch.isfinite(previous_std).all()) or bool((previous_std <= 0).any()):
        raise ValueError("Loaded action std must be finite and positive")
    previous_entropy = (previous_std.log() + 0.5 * math.log(2 * math.pi * math.e)).sum().item()
    with torch.no_grad():
        parameter.fill_(target)
    # Keep actor/critic momentum, but discard momentum from the previous std.
    algorithm.optimizer.state.pop(parameter, None)
    parameter.grad = None
    distribution._distribution = None  # Rebuilt from the next actor forward pass.
    entropy = parameter.numel() * (math.log(value) + 0.5 * math.log(2 * math.pi * math.e))
    print(
        f"[Exploration reset] Mean std {previous_std.mean().item():.6f} -> {value:.6f}; "
        f"summed Gaussian entropy {previous_entropy:.6f} -> {entropy:.6f}; std optimizer state cleared.",
        flush=True,
    )
    return {"std_before": previous_std.mean().item(), "std_after": value,
            "entropy_before": previous_entropy, "entropy_after": entropy}
