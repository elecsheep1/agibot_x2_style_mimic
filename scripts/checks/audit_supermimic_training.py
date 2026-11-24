"""Gradient isolation and fresh-real-rollout prediction diagnostics for the smoke runner."""
import math

import torch


def module_state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def assert_unchanged(module, saved):
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, saved[name], atol=0, rtol=0)


def assert_no_grad(module):
    if any(p.grad is not None for p in module.parameters()):
        raise AssertionError('Frozen module accumulated parameter gradients')


def audit_gradients(trainer, replay, batch_size):
    protected = dict(world=trainer.world_model, critic=trainer.critic, style=trainer.style,
                     proprioception=trainer.proprioception, features=trainer.amp_features, task=trainer.task)
    saved = {name: module_state(module) for name, module in protected.items()}
    flags = {name: [p.requires_grad for p in module.parameters()] for name, module in protected.items()}
    actor_before = module_state(trainer.actor)
    batch = replay.sample_policy_histories(batch_size, history_length=5)
    trajectory, metrics = trainer.update_actor(batch['history'], batch['context'], batch['commands'])
    for name, module in protected.items():
        assert_unchanged(module, saved[name])
        assert_no_grad(module)
        assert [p.requires_grad for p in module.parameters()] == flags[name]
    actor_change = max(float((value - actor_before[name]).abs().max())
                       for name, value in trainer.actor.state_dict().items())
    assert actor_change > 0 and metrics['actor_gradient_norm'] > 0
    assert not trajectory.states.requires_grad and not trajectory.rewards.requires_grad
    # The critic must not mutate any actor/dynamics/style parameter or buffer.
    other = dict(actor=trainer.actor, world=trainer.world_model, style=trainer.style)
    other_saved = {name: module_state(module) for name, module in other.items()}
    critic_before = module_state(trainer.critic)
    trainer.update_critic(trajectory, batch['context'])
    for name, module in other.items():
        assert_unchanged(module, other_saved[name])
    critic_change = max(float((value - critic_before[name]).abs().max())
                        for name, value in trainer.critic.state_dict().items())
    assert critic_change > 0
    return dict(passed=True, actor_parameter_max_change=actor_change,
                actor_gradient_norm=metrics['actor_gradient_norm'],
                frozen_modules_unchanged=list(protected), requires_grad_restored=True,
                frozen_parameter_gradients_absent=True, returned_trajectory_detached=True,
                critic_parameter_max_change=critic_change, critic_isolation=True)


def prediction_errors(predicted, target, spec, *, state_module=None, rotation_error_fn=None):
    if state_module is None:
        from x2_style_mimic.algorithms.supermimic.state import unpack_state
    else:
        unpack_state = state_module.unpack_state
    if rotation_error_fn is None:
        from x2_style_mimic.utils.rotation import quaternion_angle_error
    else:
        quaternion_angle_error = rotation_error_fn
    p, t = unpack_state(predicted, spec), unpack_state(target, spec)
    rmse = lambda value: float(value.square().mean().sqrt())
    angles = quaternion_angle_error(p['root_pose'][:, 3:], t['root_pose'][:, 3:])
    return dict(root_position_rmse_m=rmse(p['root_pose'][:, :3] - t['root_pose'][:, :3]),
                root_rotation_rmse_deg=rmse(angles) * 180 / math.pi,
                root_linear_rmse_m_s=rmse(p['root_velocity'][:, :3] - t['root_velocity'][:, :3]),
                root_angular_rmse_rad_s=rmse(p['root_velocity'][:, 3:] - t['root_velocity'][:, 3:]),
                joint_position_rmse_rad=rmse(p['joint_position'] - t['joint_position']),
                joint_velocity_rmse_rad_s=rmse(p['joint_velocity'] - t['joint_velocity']))


@torch.no_grad()
def audit_predictions(env, trainer, context, batch_size, seed):
    from x2_style_mimic.algorithms.supermimic.buffer import SequenceReplayBuffer
    from x2_style_mimic.algorithms.supermimic.runner import SuperMimicRunner
    from x2_style_mimic.algorithms.supermimic.world_model import rollout_logged_actions
    torch.manual_seed(seed)
    spec = env.world_state_spec
    # Separate replay: none of these transitions enters any optimizer update.
    command_name = None if env.cfg.commands is None else 'base_velocity'
    command_dim = 0 if command_name is None else 3
    replay = SequenceReplayBuffer(env.num_envs, 96, spec.dim, spec.num_actions,
                                  context, device=env.device, command_dim=command_dim)
    runner = SuperMimicRunner(env, trainer, replay, command_name, action_noise_std=0.1)
    before = module_state(trainer.world_model)
    proprio_before = module_state(trainer.proprioception)
    proprio_update = trainer.proprioception.update_normalization
    update_normalization = getattr(trainer.world_model, 'update_normalization', None)
    try:
        if update_normalization is not None:
            trainer.world_model.update_normalization = lambda *args, **kwargs: None
        trainer.proprioception.update_normalization = lambda *args, **kwargs: None
        collection = runner.collect(96)
    finally:
        if update_normalization is not None:
            trainer.world_model.update_normalization = update_normalization
        trainer.proprioception.update_normalization = proprio_update
    # Same starts for all horizons; sampler excludes cross-episode windows.
    batch = replay.sample_windows(batch_size, horizon=32)
    truth = batch['states']
    predicted = rollout_logged_actions(trainer.world_model, truth[:, 0], batch['actions'], batch['context'])
    if not bool(torch.isfinite(predicted).all()):
        raise RuntimeError('Nonfinite held-out prediction')
    result = {}
    for horizon in (1, 8, 32):
        result[str(horizon)] = dict(seconds=horizon * env.step_dt,
                                   model=prediction_errors(predicted[:, horizon], truth[:, horizon], spec),
                                   persistence=prediction_errors(truth[:, 0], truth[:, horizon], spec))
    for horizon in (1, 8, 32):
        goals = truth.new_empty((batch_size, 0))
        style_zero = truth.new_zeros(batch_size)
        predicted_reward, predicted_terms = trainer.task.reward(
            predicted[:, horizon - 1], batch['actions'][:, horizon - 1],
            predicted[:, horizon], goals, style_zero)
        real_reward, real_terms = trainer.task.reward(
            truth[:, horizon - 1], batch['actions'][:, horizon - 1],
            truth[:, horizon], goals, style_zero)
        result[str(horizon)]['task_reward_proxy'] = dict(
            predicted_mean=float(predicted_reward.mean()), real_state_mean=float(real_reward.mean()),
            mean_absolute_error=float((predicted_reward - real_reward).abs().mean()),
            bias=float((predicted_reward - real_reward).mean()),
            standing_mean_absolute_error=float((predicted_terms['standing'] - real_terms['standing']).abs().mean()),
            head_penalty_mean_absolute_error=float((predicted_terms['head_contact'] - real_terms['head_contact']).abs().mean()))
    assert_unchanged(trainer.world_model, before)
    assert_unchanged(trainer.proprioception, proprio_before)
    return dict(proprioception_unchanged=True, finite=True, world_model_unchanged=True, collected_transitions=96 * env.num_envs,
                sampled_windows=batch_size, collection=collection, endpoint_errors=result,
                note='Fresh stochastic rollouts from the same reset distribution; overlapping windows are possible. No prediction-quality threshold is imposed.')
