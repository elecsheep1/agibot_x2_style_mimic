"""Validate model-based actor/critic updates on a local four-joint reference task.

Runs differentiable observation reconstruction, BPTT through a frozen world model,
lambda-return critic updates, and paired PhysX return measurements.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch
from torch import nn

from collect_supermimic_smoke import make_collection_config
from fit_supermimic_smoke import load_modules, sample, valid_windows


def rotate_inverse(quaternion, vector):
    xyz = -quaternion[..., :3]
    cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., 3:] * cross + torch.cross(xyz, cross, dim=-1)


def policy_observation(history, spec, joint_ids, offset, state_module):
    batch, length, _ = history.shape
    parts = state_module.unpack_state(history.reshape(-1, spec.dim), spec)
    quaternion = parts["root_pose"][:, 3:]
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    gravity = torch.zeros_like(parts["root_velocity"][:, :3])
    gravity[:, 2] = -1.0
    fields = (
        rotate_inverse(quaternion, parts["root_velocity"][:, 3:]),
        rotate_inverse(quaternion, gravity),
        parts["joint_position"][:, joint_ids] - offset,
        parts["joint_velocity"][:, joint_ids],
        parts["previous_action"],
    )
    # ObservationManager flattens each term's history before concatenating terms.
    return torch.cat([value.reshape(batch, length, -1).flatten(1) for value in fields], dim=-1)


class ResidualActor(nn.Module):
    def __init__(self, base_policy, joint_ids, offset, task_action_ids, spec, state_module):
        super().__init__()
        self.base_policy = base_policy
        self.spec = spec
        self.state_module = state_module
        self.register_buffer("joint_ids", joint_ids)
        self.register_buffer("offset", offset)
        self.register_buffer("task_action_ids", task_action_ids)
        self.residual = nn.Sequential(nn.Linear(465 + 4, 128), nn.ELU(), nn.Linear(128, 4))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        for parameter in base_policy.parameters():
            parameter.requires_grad_(False)
        base_policy.eval()

    def forward(self, history, goal):
        obs = policy_observation(history, self.spec, self.joint_ids, self.offset, self.state_module)
        parts = self.state_module.unpack_state(history[:, -1], self.spec)
        controlled = parts["joint_position"][:, self.joint_ids[self.task_action_ids]]
        features = torch.cat((self.base_policy.obs_normalizer(obs), (goal - controlled) / 0.1), dim=-1)
        residual = 0.8 * torch.tanh(self.residual(features))
        action = self.base_policy(obs)
        return action.index_add(1, self.task_action_ids, residual)


def task_reward(before, following, action, goal, pose_goal, spec, joint_ids, state_module, dynamics):
    b = state_module.unpack_state(before, spec)
    s = state_module.unpack_state(following, spec)
    tracking = torch.exp(-((s["joint_position"][:, joint_ids] - goal) / 0.1).square().mean(-1))
    height_cost = ((s["root_pose"][:, 2] - pose_goal[:, 2]) / 0.05).square()
    tilt_cost = (dynamics.rotation_error(s["root_pose"][:, 3:], pose_goal[:, 3:]) / 0.3).square()
    velocity_cost = s["root_velocity"][:, :3].square().sum(-1) / 0.5**2
    angular_cost = s["root_velocity"][:, 3:].square().sum(-1)
    action_cost = (action - b["previous_action"]).square().mean(-1)
    return tracking - 0.1 * height_cost - 0.1 * tilt_cost - 0.01 * velocity_cost - 0.005 * angular_cost - 0.001 * action_cost


def critic_value(critic, model, state, context, goal, pose_goal, joint_ids, state_module, dynamics):
    s = state_module.unpack_state(state, model.spec)
    features, _ = model.features(state, s["previous_action"], context)
    normalized = (features - model.input_mean) / model.input_std
    goal_error = (goal - s["joint_position"][:, joint_ids]) / 0.1
    quaternion = dynamics.unit_quat(s["root_pose"][:, 3:])
    inverse = torch.cat((-quaternion[:, :3], quaternion[:, 3:]), -1)
    relative = dynamics.quat_mul(dynamics.unit_quat(pose_goal[:, 3:]), inverse)
    pose_error = torch.cat(((pose_goal[:, :3] - s["root_pose"][:, :3]) / 0.1,
                            dynamics.quat_to_rot6d(relative)), -1)
    return critic(torch.cat((normalized, goal_error, pose_error), -1)).squeeze(-1)


def imagine(actor, model, history, context, goal, pose_goal, horizon, task_joint_ids, state_module, dynamics):
    states = [history[:, -1]]
    rewards = []
    probabilities = []
    for _ in range(horizon):
        action = actor(history, goal)
        following = model(states[-1], action, context)
        rewards.append(task_reward(states[-1], following, action, goal, pose_goal, model.spec,
                                   task_joint_ids, state_module, dynamics))
        # Smooth version of the environment's 2m out-of-bounds termination.
        distance = (following[:, :2] - pose_goal[:, :2]).norm(dim=-1)
        probabilities.append(torch.sigmoid((distance - 2.0) / 0.1))
        states.append(following)
        history = torch.cat((history[:, 1:], following[:, None]), dim=1)
    return torch.stack(states), torch.stack(rewards), torch.stack(probabilities)


def main():
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("logs/supermimic/policy_noise005_v1/transitions.pt"))
    parser.add_argument("--model-run", type=Path, default=Path("logs/supermimic/fit_noise005_capacity256_v1"))
    parser.add_argument("--output", type=Path, default=Path("logs/supermimic/loop_smoke_v1"))
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--actor-lr", type=float, default=2e-4)
    parser.add_argument("--bootstrap-weight", type=float, default=1.0,
                        help="1 uses the critic; 0 is a finite-horizon diagnostic control")
    parser.add_argument("--online-rounds", type=int, default=0)
    parser.add_argument("--wm-updates-per-round", type=int, default=10)
    add_launcher_args(parser)
    args, rest = setup_preset_cli(parser)
    if rest or args.updates < 1 or not 1 <= args.horizon <= 32 or not 0 < args.actor_lr < float("inf") or args.bootstrap_weight not in (0.0, 1.0):
        parser.error("Use positive updates, horizon in [1,32], and recognized arguments")
    if args.output.exists():
        parser.error("Choose a new output directory")
    if args.online_rounds < 0 or args.wm_updates_per_round < 1:
        parser.error("online-rounds must be nonnegative and wm-updates-per-round positive")
    state_module, dynamics = load_modules()
    objectives = importlib.import_module(state_module.__package__ + ".objectives")
    gradients = importlib.import_module(state_module.__package__ + ".gradients")
    data = torch.load(args.data, weights_only=True, map_location="cpu")
    checkpoint = torch.load(args.model_run / "world_model.pt", weights_only=True, map_location="cpu")
    args.num_envs = data["contexts"].shape[0]
    args.seed = data["metadata"]["seed"]
    cfg = make_collection_config(args)
    with launch_simulation(cfg, args):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.getup import _write_reference_state
        from x2_style_mimic.algorithms.supermimic.context import read_sim_context

        torch.manual_seed(args.seed)
        env = X2SuperMimicEnv(cfg=cfg)
        try:
            env.reset()
            context, _ = read_sim_context(env)
            torch.testing.assert_close(context.cpu(), data["contexts"], atol=0, rtol=0)
            spec = state_module.StateSpec(**checkpoint["metadata"]["state_spec"])
            model = dynamics.WorldModel(spec, checkpoint["context_dim"], dynamics.TargetLimiter(**checkpoint["limiter"]),
                                        hidden_dims=checkpoint["hidden_dims"]).to(env.device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            base = torch.jit.load(data["metadata"]["policy_jit"], map_location=env.device)
            action_names = data["metadata"]["control_joint_names"]
            joint_ids = torch.tensor([data["metadata"]["joint_names"].index(name) for name in action_names], device=env.device)
            task_names = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_elbow_joint", "right_elbow_joint"]
            task_action_ids = torch.tensor([action_names.index(name) for name in task_names], device=env.device)
            task_joint_ids = joint_ids[task_action_ids]
            robot = env.scene["robot"]
            torch.testing.assert_close(robot.data.default_joint_vel.torch[:, joint_ids], torch.zeros(env.num_envs, spec.num_actions, device=env.device), atol=0, rtol=0)
            torch.manual_seed(123)
            actor = ResidualActor(base, joint_ids, model.limiter.offset.clone(), task_action_ids, spec, state_module).to(env.device)
            critic = nn.Sequential(nn.Linear(model.input_mean.numel() + 13, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1)).to(env.device)
            nn.init.zeros_(critic[-1].weight)
            nn.init.zeros_(critic[-1].bias)
            actor_optimizer = torch.optim.Adam(actor.residual.parameters(), lr=args.actor_lr, betas=(0.9, 0.95))
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=2e-4)
            goal_offset = torch.tensor([0.08, 0.08, -0.08, -0.08], device=env.device)

            def initial(step):
                value = data["state"][step].to(env.device)
                history = value[:, None].expand(-1, 5, -1).clone()
                parts = state_module.unpack_state(value, spec)
                return history, parts["joint_position"][:, task_joint_ids] + goal_offset, parts["root_pose"].clone()

            observation_errors = []

            def restore(history):
                env.reset()
                parts = state_module.unpack_state(history[:, -1], spec)
                ids = torch.arange(env.num_envs, device=env.device)
                pose = parts["root_pose"].clone()
                pose[:, :3] += env.scene.env_origins
                _write_reference_state(env, ids, robot, pose, parts["root_velocity"], parts["joint_position"], parts["joint_velocity"])
                term = env.action_manager.get_term("joint_pos")
                term._raw_actions.copy_(parts["previous_action"])
                term._previous_target.copy_(parts["previous_target"])
                term._processed_actions.copy_(parts["previous_target"])
                env.action_manager.action.copy_(parts["previous_action"])
                env.action_manager.prev_action.copy_(parts["previous_action"])
                term.apply_actions()
                env.scene.write_data_to_sim()
                env.observation_manager.reset(ids)
                obs = env.observation_manager.compute(update_history=True)["policy"]
                analytic = policy_observation(history, spec, joint_ids, model.limiter.offset, state_module)
                error = float((obs - analytic).abs().max())
                observation_errors.append(error)
                torch.testing.assert_close(obs, analytic, atol=2e-5, rtol=1e-5)
                torch.testing.assert_close(env.read_world_state(), history[:, -1], atol=2e-5, rtol=1e-5)

            @torch.no_grad()
            def real_return(history, goal, pose_goal, horizon):
                restore(history)
                history = history.clone()
                result = torch.zeros(env.num_envs, device=env.device)
                for step in range(horizon):
                    action = actor(history, goal)
                    observations, _, terminated, truncated, extras = env.step(action)
                    if bool((terminated | truncated).any()):
                        raise RuntimeError("Unexpected termination during local tracking check")
                    following = extras["world_model"]["next_state"]
                    result += 0.99**step * task_reward(history[:, -1], following, action, goal, pose_goal,
                                                     spec, task_joint_ids, state_module, dynamics)
                    history = torch.cat((history[:, 1:], following[:, None]), dim=1)
                    analytic = policy_observation(history, spec, joint_ids, model.limiter.offset, state_module)
                    error = float((observations["policy"] - analytic).abs().max())
                    observation_errors.append(error)
                    torch.testing.assert_close(observations["policy"], analytic, atol=2e-5, rtol=1e-5)
                return result

            @torch.no_grad()
            def model_return(history, goal, pose_goal, horizon):
                _, rewards, _ = imagine(actor, model, history, context, goal, pose_goal, horizon, task_joint_ids, state_module, dynamics)
                discount = 0.99**torch.arange(horizon, device=env.device)
                return (rewards * discount[:, None]).sum(0)

            train_history, train_goal, train_pose = initial(500)
            eval_history, eval_goal, eval_pose = initial(700)
            frozen_snapshot = {key: value.clone() for key, value in model.state_dict().items()}
            base_snapshot = {key: value.clone() for key, value in base.state_dict().items()}
            baseline = {h: real_return(eval_history, eval_goal, eval_pose, h).cpu() for h in (8, 32)}
            repeat = real_return(eval_history, eval_goal, eval_pose, 8).cpu()
            model_before = {h: model_return(eval_history, eval_goal, eval_pose, h).cpu() for h in (8, 32)}

            # Check the local model gradient against paired simulator finite differences.
            with gradients.frozen_parameters(model, critic, base):
                _, rewards, done_probability = imagine(actor, model, eval_history, context, eval_goal, eval_pose, 8, task_joint_ids, state_module, dynamics)
                objective = objectives.imagined_actor_loss(rewards, done_probability, torch.zeros(env.num_envs, device=env.device))
                bias_gradient = -torch.autograd.grad(objective, actor.residual[-1].bias)[0]
            directions = [bias_gradient / bias_gradient.norm().clamp_min(1e-9)]
            generator = torch.Generator(device=env.device).manual_seed(456)
            for _ in range(2):
                direction = torch.randn(4, device=env.device, generator=generator)
                directions.append(direction / direction.norm())
            directions.extend(torch.eye(4, device=env.device).unbind(0))
            finite_differences = []
            original_bias = actor.residual[-1].bias.detach().clone()
            for direction in directions:
                returns = []
                for sign in (1., -1.):
                    with torch.no_grad():
                        actor.residual[-1].bias.copy_(original_bias + sign * 0.05 * direction)
                    returns.append((float(model_return(eval_history, eval_goal, eval_pose, 8).mean()),
                                    float(real_return(eval_history, eval_goal, eval_pose, 8).mean())))
                finite_differences.append(dict(direction=direction.cpu().tolist(), analytic_model_derivative=float(bias_gradient @ direction),
                                              model_finite_difference=(returns[0][0] - returns[1][0]) / 0.1,
                                              simulator_finite_difference=(returns[0][1] - returns[1][1]) / 0.1))
            with torch.no_grad():
                actor.residual[-1].bias.copy_(original_bias)

            fresh_buffer = None
            if args.online_rounds:
                buffer_module = importlib.import_module(state_module.__package__ + ".buffer")
                fresh_buffer = buffer_module.SequenceReplayBuffer(env.num_envs, 1024, spec.dim, spec.num_actions, context, device=env.device)
                model_optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
                source_report = json.loads((args.model_run / "report.json").read_text())
                training_mask = torch.zeros_like(data["terminated"])
                for env_id, episode in source_report["train_episodes"]:
                    training_mask[:, env_id] |= data["episode_ids"][:, env_id] == episode
                replay_candidates = valid_windows(data, 8)
                replay_candidates = replay_candidates[training_mask[replay_candidates[:, 0], replay_candidates[:, 1]]]
            updates = args.online_rounds or args.updates
            training = []
            for update in range(updates):
                wm_loss = None
                policy_history, policy_context, policy_goal, policy_pose = train_history, context, train_goal, train_pose
                if fresh_buffer is not None:
                    restore(train_history)
                    if update:
                        fresh_buffer.mark_reset(torch.arange(env.num_envs, device=env.device))
                    history = train_history.clone()
                    with torch.no_grad():
                        for _ in range(32):
                            action = actor(history, train_goal) + torch.randn(env.num_envs, spec.num_actions, device=env.device) * 0.05
                            _, _, terminated, truncated, extras = env.step(action)
                            if bool((terminated | truncated).any()):
                                raise RuntimeError("Unexpected termination during online collection")
                            transition = extras["world_model"]
                            transition["reward"] = task_reward(history[:, -1], transition["next_state"], action, train_goal,
                                                               train_pose, spec, task_joint_ids, state_module, dynamics)
                            fresh_buffer.add(transition)
                            history = torch.cat((history[:, 1:], transition["next_state"][:, None]), dim=1)
                    for _ in range(args.wm_updates_per_round):
                        fresh = fresh_buffer.sample_windows(32, horizon=8)
                        replay = sample(data, replay_candidates, 32, 8, env.device)
                        batch = {key: torch.cat((fresh[key], replay[key]), 0) for key in ("states", "actions", "context")}
                        prediction = dynamics.rollout_logged_actions(model, batch["states"][:, 0], batch["actions"], batch["context"])
                        wm_loss, _ = dynamics.dynamics_loss(prediction, batch["states"], spec)
                        model_optimizer.zero_grad(set_to_none=True)
                        wm_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                        model_optimizer.step()
                    model_optimizer.zero_grad(set_to_none=True)
                    real_history = fresh_buffer.sample_windows(32, horizon=4)
                    policy_history = real_history["states"]
                    policy_context = real_history["context"]
                    policy_goal = train_goal[real_history["context_id"]]
                    policy_pose = train_pose[real_history["context_id"]]
                actor_phase_snapshot = {key: value.clone() for key, value in model.state_dict().items()}
                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                with gradients.frozen_parameters(model, critic, base):
                    states, rewards, done_probability = imagine(actor, model, policy_history, policy_context, policy_goal, policy_pose,
                                                               args.horizon, task_joint_ids, state_module, dynamics)
                    bootstrap = critic_value(critic, model, states[-1], policy_context, policy_goal, policy_pose, task_joint_ids, state_module, dynamics)
                    loss = objectives.imagined_actor_loss(rewards, done_probability, args.bootstrap_weight * bootstrap)
                    loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(actor.residual.parameters(), 1.0, error_if_nonfinite=True)
                for parameter in actor.residual.parameters():
                    if parameter.grad is not None:
                        parameter.grad.clamp_(-0.1, 0.1)
                if any(parameter.grad is not None for parameter in model.parameters()) or any(parameter.grad is not None for parameter in critic.parameters()):
                    raise RuntimeError("Actor backward leaked gradients into frozen model or critic")
                actor_optimizer.step()
                detached = states.detach()
                contexts = policy_context.expand(args.horizon + 1, -1, -1).reshape(-1, context.shape[-1])
                goals = policy_goal.expand(args.horizon + 1, -1, -1).reshape(-1, 4)
                poses = policy_pose.expand(args.horizon + 1, -1, -1).reshape(-1, 7)
                with torch.no_grad():
                    values = critic_value(critic, model, detached.reshape(-1, spec.dim), contexts, goals, poses,
                                          task_joint_ids, state_module, dynamics).reshape(args.horizon + 1, -1)
                    targets = objectives.lambda_returns(rewards.detach(), done_probability.detach(), values)
                fixed_targets = targets.clone()
                for _ in range(5):
                    count = args.horizon * policy_context.shape[0]
                    for indexes in torch.randperm(count, device=env.device).chunk(4):
                        predictions = critic_value(critic, model, detached[:-1].reshape(-1, spec.dim)[indexes],
                                                   contexts[:count][indexes], goals[:count][indexes], poses[:count][indexes],
                                                   task_joint_ids, state_module, dynamics)
                        critic_loss = objectives.critic_loss(predictions, targets.reshape(-1)[indexes])
                        critic_optimizer.zero_grad(set_to_none=True)
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
                        critic_optimizer.step()
                torch.testing.assert_close(targets, fixed_targets, atol=0, rtol=0)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, actor_phase_snapshot[key], atol=0, rtol=0)
                training.append(dict(update=update, actor_loss=float(loss.detach()), actor_gradient_norm=float(norm),
                                     critic_loss=float(critic_loss.detach()), bootstrap_mean=float(bootstrap.detach().mean()),
                                     bootstrap_max=float(bootstrap.detach().max()), target_max=float(targets.max()),
                                     world_model_loss=float(wm_loss.detach()) if wm_loss is not None else None))
                if (update + 1) % 5 == 0:
                    print(f"[SuperMimic loop] {update + 1}/{updates}", flush=True)
            model_after = {h: model_return(eval_history, eval_goal, eval_pose, h).cpu() for h in (8, 32)}
            real_after = {h: real_return(eval_history, eval_goal, eval_pose, h).cpu() for h in (8, 32)}
            if fresh_buffer is None:
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, frozen_snapshot[key], atol=0, rtol=0)
            for key, value in base.state_dict().items():
                torch.testing.assert_close(value, base_snapshot[key], atol=0, rtol=0)
            result = dict(protocol="Four-joint static goal tracking with a bounded residual over exported policy; analytic observation/reward/termination; pretrained world model frozen during policy updates; learned critic and existing BPTT/lambda-return objectives",
                          task_joints=task_names, goal_offset_rad=goal_offset.cpu().tolist(), horizon=args.horizon, updates=updates,
                          training_initial_step=500, evaluation_initial_step=700, num_envs=env.num_envs,
                          actor_lr=args.actor_lr, actor_adam_betas=[0.9, 0.95], bootstrap_weight=args.bootstrap_weight,
                          critic_epochs=5, critic_minibatches=4,
                          online_rounds=args.online_rounds, fresh_transitions=fresh_buffer.total_steps * env.num_envs if fresh_buffer else 0,
                          max_observation_reconstruction_error=max(observation_errors),
                          repeat_real_return_max_difference=float((repeat - baseline[8]).abs().max()),
                          finite_differences=finite_differences, training=training, horizons={})
            simulator_gradient = torch.tensor([entry["simulator_finite_difference"] for entry in finite_differences[-4:]], device=env.device)
            result["local_gradient_check"] = dict(model_gradient=bias_gradient.cpu().tolist(), simulator_gradient=simulator_gradient.cpu().tolist(),
                                                  cosine=float(torch.nn.functional.cosine_similarity(bias_gradient, simulator_gradient, dim=0)),
                                                  simulator_to_model_norm_ratio=float(simulator_gradient.norm() / bias_gradient.norm().clamp_min(1e-9)))
            for h in (8, 32):
                delta = real_after[h] - baseline[h]
                result["horizons"][str(h)] = dict(model_before=float(model_before[h].mean()), model_after=float(model_after[h].mean()),
                                                 real_before=float(baseline[h].mean()), real_after=float(real_after[h].mean()),
                                                 mean_real_improvement=float(delta.mean()), fraction_envs_improved=float((delta > 0).float().mean()))
            args.output.mkdir(parents=True, exist_ok=False)
            world_model_path = args.output / "updated_world_model.pt" if fresh_buffer is not None else args.model_run / "world_model.pt"
            torch.save(dict(actor_residual=actor.residual.state_dict(), critic=critic.state_dict(), model_run=str(args.model_run.resolve()),
                            world_model_checkpoint=str(world_model_path.resolve()), base_policy=data["metadata"]["policy_jit"],
                            state_spec=data["metadata"]["state_spec"], control_joint_names=action_names,
                            task_joints=task_names, goal_offset_rad=goal_offset.cpu(), actor_residual_bound=0.8), args.output / "actor_critic.pt")
            if fresh_buffer is not None:
                torch.save(dict(model=model.state_dict(), metadata=checkpoint["metadata"], hidden_dims=checkpoint["hidden_dims"],
                                limiter=checkpoint["limiter"], context_dim=checkpoint["context_dim"]), args.output / "updated_world_model.pt")
            (args.output / "report.json").write_text(json.dumps(result, indent=2))
            print(json.dumps({key: value for key, value in result.items() if key != "training"}, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
