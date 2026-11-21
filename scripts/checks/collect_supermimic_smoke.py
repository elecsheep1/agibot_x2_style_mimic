"""Collect real SuperMimic transitions and audit the simulator interface; no policy update.

Random actions are intentional: this is an interface smoke test, not a recovery dataset.
The runtime-only configuration omits custom get-up hooks and their dependent terms.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch


def shared_vector(value, count, name, device):
    value = torch.as_tensor(value, device=device, dtype=torch.float32)
    if value.ndim == 0:
        return value.expand(count).clone()
    if value.ndim == 2:
        if not torch.allclose(value, value[:1].expand_as(value), atol=1e-7, rtol=0):
            raise RuntimeError(f"{name} varies by environment; TargetLimiter cannot represent it")
        value = value[0]
    if value.shape != (count,):
        raise RuntimeError(f"Unexpected {name} shape: {value.shape}")
    return value.clone()


def policy_action(policy, observations, num_envs, num_actions, noise_std):
    """Evaluate exported policy and return the action that will be sent to the simulator."""
    action = policy(observations["policy"])
    if action.shape != (num_envs, num_actions):
        raise RuntimeError(f"Unexpected policy action shape: {tuple(action.shape)}")
    if not bool(torch.isfinite(action).all()):
        raise RuntimeError("Policy action contains NaN/Inf")
    if noise_std > 0.0:
        action = action + torch.randn_like(action) * noise_std
    return action


def make_collection_config(args):
    """Shared zero-assistance simulator configuration for collection and loop checks."""
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_getup_env_cfg import X2MultiMotionGetUpEnvCfg

    cfg = X2MultiMotionGetUpEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.device is not None:
        cfg.sim.device = args.device
    cfg.assist_curriculum_enabled = False
    cfg.assist_initial_stage = len(cfg.assist_force_levels) - 1
    cfg.assist_zero_probability = 1.0
    # This environment directly inherits ManagerBasedRLEnv. Do not enable terms
    # requiring X2MultiMotionGetUpEnv's per-substep contact/assistance callbacks.
    cfg.scene.self_contact = None
    # Remaining observations/rewards do not read contact sensors. Avoid constructing
    # unused ground filters in this standalone dynamics collection configuration.
    for name in ("contact_forces", "head_yaw_ground_contact", "head_pitch_ground_contact",
                 "left_foot_ground_contact", "right_foot_ground_contact",
                 "left_hand_ground_contact", "right_hand_ground_contact"):
        setattr(cfg.scene, name, None)
    for name in ("standing", "head_contact", "joint_pos_limits", "joint_acc",
                 "self_contact", "out_of_bounds_failure"):
        setattr(cfg.rewards, name, None)
    cfg.terminations.progress = None
    cfg.events.standing_perturbation = None
    cfg.observations.amp = None
    cfg.observations.policy.enable_corruption = False
    reset = cfg.events.reset_from_reference.params
    reset.update(birth_probabilities=(0.5, 0.0, 0.5, 0.0, 0.0),
                 perturb_state_bank=None, recovery_state_bank=None)

    return cfg


def main(policy_mode=False):
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli

    description = "Collect and audit real transitions using an exported X2 policy." if policy_mode else __doc__
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-std", type=float, default=0.0 if policy_mode else 0.10)
    if policy_mode:
        parser.add_argument("--policy-jit", type=Path, help="Exported feed-forward policy.pt, including normalization")
    default_output = "logs/supermimic/policy_clean_v1" if policy_mode else "logs/supermimic/smoke_v1"
    parser.add_argument("--output", type=Path, default=Path(default_output))
    add_launcher_args(parser)
    args, rest = setup_preset_cli(parser)
    if rest:
        parser.error(f"Unrecognized arguments: {rest}")
    if args.num_envs < 2 or args.steps < 1032:
        parser.error("Use >=2 environments and >=1032 steps")
    if not 0.0 <= args.action_std <= 1.0 or (not policy_mode and args.action_std == 0.0):
        parser.error("action-std must be in [0,1] for policy collection, or (0,1] for random collection")
    if policy_mode and (args.policy_jit is None or not args.policy_jit.is_file()):
        parser.error("Provide an existing exported policy with --policy-jit")
    if args.output.exists():
        parser.error("Output already exists; choose a fresh --output directory")
    sys.argv = [sys.argv[0]]

    cfg = make_collection_config(args)

    with launch_simulation(cfg, args):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv
        from x2_style_mimic.algorithms.supermimic.buffer import SequenceReplayBuffer
        from x2_style_mimic.algorithms.supermimic.context import read_sim_context
        from x2_style_mimic.algorithms.supermimic.state import unpack_state
        from x2_style_mimic.algorithms.supermimic.world_model import TargetLimiter

        torch.manual_seed(args.seed)
        env = X2SuperMimicEnv(cfg=cfg)
        try:
            observations, _ = env.reset()
            spec = env.world_state_spec
            policy = None
            if policy_mode:
                policy = torch.jit.load(str(args.policy_jit.resolve()), map_location=env.device)
                policy.eval()
                recurrent_names = {"LSTM", "GRU", "RNN", "LSTMCell", "GRUCell", "RNNCell"}
                if any(getattr(module, "original_name", "") in recurrent_names for module in policy.modules()):
                    raise RuntimeError("This collector supports feed-forward exported policies only")
                with torch.no_grad():
                    policy_action(policy, observations, env.num_envs, spec.num_actions, 0.0)
            contexts, context_spec = read_sim_context(env)
            robot = env.scene["robot"]
            term = env.action_manager.get_term("joint_pos")
            ids = term._joint_ids
            limiter_params = {
                "offset": shared_vector(term._offset, spec.num_actions, "offset", env.device),
                "scale": shared_vector(term._scale, spec.num_actions, "scale", env.device),
                "lower": shared_vector(robot.data.joint_pos_limits.torch[:, ids, 0], spec.num_actions, "lower", env.device),
                "upper": shared_vector(robot.data.joint_pos_limits.torch[:, ids, 1], spec.num_actions, "upper", env.device),
                "max_delta": shared_vector(robot.data.joint_vel_limits.torch[:, ids] * term.cfg.target_velocity_fraction * env.step_dt,
                                           spec.num_actions, "max_delta", env.device),
            }
            limiter = TargetLimiter(**limiter_params).to(env.device)
            buffer = SequenceReplayBuffer(env.num_envs, args.steps, spec.dim,
                                          spec.num_actions, contexts, device="cpu")
            done_count = 0
            timeout_count = 0
            max_continuity = max_target_error = max_quat_error = 0.0
            previous = previous_done = None
            birth_types = []
            motion_ids = []
            episode_steps = []
            reference_frames = []
            height_above_half_m = 0
            for step in range(args.steps):
                # Capture provenance before stepping because done environments auto-reset.
                birth_types.append(env._getup_birth_types.detach().cpu().clone())
                motion_ids.append(env._getup_reference_motion_ids.detach().cpu().clone())
                episode_steps.append(env.episode_length_buf.detach().cpu().clone())
                reference_frames.append(env._getup_reference_frame_ids.detach().cpu().clone())
                with torch.no_grad():
                    if policy is None:
                        action = torch.randn(env.num_envs, spec.num_actions, device=env.device) * args.action_std
                    else:
                        action = policy_action(policy, observations, env.num_envs, spec.num_actions, args.action_std)
                    observations, _, _, _, extras = env.step(action)
                transition = extras["world_model"]
                for name in ("state", "next_state", "action", "reward"):
                    if not bool(torch.isfinite(transition[name]).all()):
                        raise RuntimeError(f"Nonfinite {name} at step {step}")
                torch.testing.assert_close(transition["action"], action, atol=0, rtol=0)
                s = unpack_state(transition["state"], spec)
                following = unpack_state(transition["next_state"], spec)
                height_above_half_m += int((following["root_pose"][:, 2] > 0.5).sum())
                expected = limiter(s["previous_target"], action)
                error = (following["previous_target"] - expected).abs().max().item()
                max_target_error = max(max_target_error, error)
                torch.testing.assert_close(following["previous_target"], expected, atol=2e-6, rtol=1e-6)
                torch.testing.assert_close(following["previous_action"], action, atol=0, rtol=0)
                if previous is not None and bool((~previous_done).any()):
                    mask = ~previous_done
                    error = (transition["state"][mask] - previous[mask]).abs().max().item()
                    max_continuity = max(max_continuity, error)
                    torch.testing.assert_close(transition["state"][mask], previous[mask], atol=2e-6, rtol=1e-6)
                quat_error = (following["root_pose"][:, 3:].norm(dim=-1) - 1).abs().max().item()
                max_quat_error = max(max_quat_error, quat_error)
                if quat_error > 1e-3:
                    raise RuntimeError("Simulator root quaternion is not normalized")
                done = transition["terminated"] | transition["truncated"]
                done_count += int(done.sum())
                timeout_count += int(transition["truncated"].sum())
                buffer.add(transition)
                previous = transition["next_state"].clone()
                previous_done = done.clone()
                if (step + 1) % 100 == 0:
                    print(f"[SuperMimic collection] {step + 1}/{args.steps}; completed={done_count}", flush=True)
            if timeout_count == 0:
                raise RuntimeError("No timeout observed; reset-boundary audit is incomplete")
            final_contexts, final_spec = read_sim_context(env)
            if final_spec != context_spec:
                raise RuntimeError("Context layout changed during collection")
            torch.testing.assert_close(final_contexts, contexts, atol=0, rtol=0)
            # Exercise the actual window sampler, including a 32-step rollout window.
            for horizon in (1, 8, 32):
                window = buffer.sample_windows(64, horizon=horizon)
                if window["states"].shape != (64, horizon + 1, spec.dim):
                    raise RuntimeError("Unexpected sampled window shape")

            metadata = dict(state_spec=asdict(spec), context_spec=asdict(context_spec),
                            joint_names=list(robot.joint_names), body_names=list(robot.body_names),
                            control_joint_names=[robot.joint_names[i] for i in
                                                 (list(range(spec.num_joints))[ids] if isinstance(ids, slice) else ids)],
                            step_dt=env.step_dt, physics_dt=cfg.sim.dt, decimation=cfg.decimation,
                            seed=args.seed, action_std=args.action_std, steps=args.steps,
                            num_envs=args.num_envs,
                            action_source="jit_policy_plus_gaussian_noise" if policy_mode else "gaussian_random_interface_smoke",
                            disabled_terms=["getup rewards except action_rate", "progress", "contact sensors",
                                            "standing_perturbation", "amp observation", "observation corruption"],
                            rewards_note="Only action_rate reward; not the getup task reward",
                            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            if policy_mode:
                metadata.update(policy_jit=str(args.policy_jit.resolve()),
                                policy_sha256=hashlib.sha256(args.policy_jit.read_bytes()).hexdigest())
            data = {name: getattr(buffer, name).clone() for name in
                    ("state", "next_state", "action", "rewards", "terminated", "truncated", "episode_ids", "step_ids")}
            data.update(contexts=contexts.cpu(), limiter={k: v.cpu() for k, v in limiter_params.items()}, metadata=metadata)
            data.update(birth_types=torch.stack(birth_types), motion_ids=torch.stack(motion_ids),
                        episode_steps=torch.stack(episode_steps), reference_frames=torch.stack(reference_frames))
            report = dict(passed=True, transitions=args.steps * args.num_envs, completed=done_count,
                          timeouts=timeout_count, max_continuity_error=max_continuity,
                          max_target_error=max_target_error, max_quaternion_norm_error=max_quat_error,
                          context_fixed=True, action_source=metadata["action_source"],
                          height_above_0_5m_fraction=height_above_half_m / (args.steps * args.num_envs))
            args.output.mkdir(parents=True, exist_ok=False)
            from isaaclab.utils.io import dump_yaml
            dump_yaml(str(args.output / "env.yaml"), cfg)
            torch.save(data, args.output / "transitions.pt")
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
