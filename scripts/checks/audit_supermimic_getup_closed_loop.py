"""只读采集当前策略轨迹，对照相同真实起点、相同动作的模型预测。"""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
DIRECTORY = Path(__file__).resolve().parents[1] / 'supermimic'
sys.path.insert(0, str(DIRECTORY))
import torch

from audit_supermimic_training import prediction_errors
from getup_config import parse_config


def gradient_audit(trainer, history, context):
    from x2_style_mimic.algorithms.supermimic.gradients import frozen_parameters
    from x2_style_mimic.algorithms.supermimic.rollout import imagine
    from x2_style_mimic.algorithms.supermimic.state import unpack_state

    with torch.enable_grad(), frozen_parameters(
        trainer.world_model, trainer.critic, trainer.style,
        trainer.proprioception, trainer.amp_features, trainer.task,
    ):
        trajectory = imagine(
            trainer.actor, trainer.world_model, trainer.proprioception, trainer.amp_features,
            trainer.style, trainer.task, history, context,
            torch.empty(len(history), 0, device=history.device), horizon=trainer.cfg.horizon,
        )
        p = trajectory.termination_probabilities
        survival = torch.cat((torch.ones_like(p[:1]), (1 - p).cumprod(0)[:-1]), dim=0)
        discount = p.new_tensor(trainer.cfg.gamma).pow(torch.arange(len(p), device=p.device))[:, None]
        weights = survival * discount
        components = {name: (weights * value).sum(0).mean()
                      for name, value in trajectory.reward_terms.items()}
        terminal = trainer.critic(trajectory.states[-1], context, trajectory.goals[-1])
        components['bootstrap'] = (trainer.cfg.gamma ** len(p) * (1 - p).prod(0) * terminal).mean()
        parameters = tuple(trainer.actor.parameters())
        gradients, result = {}, {}
        for name, objective in components.items():
            grad = torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
            vector = torch.cat([(torch.zeros_like(param) if g is None else g).flatten() for param, g in zip(parameters, grad)])
            gradients[name] = vector
            result[name] = dict(discounted_return=float(objective.detach()), gradient_norm=float(vector.norm()))
        standing = gradients['standing']
        for name, vector in gradients.items():
            result[name]['cosine_with_standing'] = float(torch.dot(standing, vector) / (standing.norm() * vector.norm()).clamp_min(1e-12))
        state = unpack_state(trajectory.states[:-1].flatten(0, 1), trainer.world_model.spec)
        action = trajectory.actions.flatten(0, 1)
        result['imagined_target_derivative_active_fraction'] = active_target_fraction(trainer.world_model.limiter, state, action)
    return result


def active_target_fraction(limiter, state, action):
    desired = action * limiter.scale + limiter.offset
    previous = state['previous_target'].clamp(limiter.lower, limiter.upper)
    inside_hard = (desired > limiter.lower) & (desired < limiter.upper)
    inside_rate = (desired.clamp(limiter.lower, limiter.upper) - previous).abs() < limiter.max_delta
    return float((inside_hard & inside_rate).float().mean().detach())


@torch.no_grad()
def audit(runtime, checkpoint, steps=96, batch_size=64):
    from getup_evaluation import build_cohort
    from x2_style_mimic.algorithms.supermimic.world_model import rollout_logged_actions
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.getup import _write_reference_state

    env, trainer = runtime.env, runtime.trainer
    protected = {name: {key: value.clone() for key, value in module.state_dict().items()}
                 for name, module in runtime.modules.items()}
    cohort = build_cohort(runtime, checkpoint['metadata']['evaluation_seed'])
    env.reset()
    ids = torch.arange(env.num_envs, device=env.device)
    state = cohort['states']
    pose = state['root_pose'].clone()
    pose[:, :3] += env.scene.env_origins
    _write_reference_state(env, ids, env.scene['robot'], pose, state['root_velocity'], state['joint_pos'], state['joint_vel'])
    env.action_manager.reset(ids)
    for name, key in (('_getup_birth_types', 'birth_types'), ('_getup_reference_motion_ids', 'motion_ids'),
                      ('_getup_reference_frame_ids', 'frame_ids')):
        getattr(env, name)[ids] = cohort[key]
    current = env.read_world_state()
    history = current[:, None].expand(-1, 5, -1).clone()
    states, following, actions, dones = [], [], [], []
    for _ in range(steps):
        action = trainer.actor(trainer.proprioception(history), torch.empty(env.num_envs, 0, device=env.device))
        _, _, _, _, extras = env.step(action)
        capture = extras['world_model']
        states.append(current.clone())
        following.append(capture['next_state'].clone())
        actions.append(action.clone())
        done = capture['terminated'] | capture['truncated']
        dones.append(done.clone())
        current = env.read_world_state()
        history = torch.cat((history[:, 1:], current[:, None]), dim=1)
        history[done] = current[done, None].expand(-1, 5, -1)
    states, following, actions, dones = [torch.stack(x) for x in (states, following, actions, dones)]
    valid = []
    for t in range(4, steps - 32 + 1):
        ok = ~dones[t - 4:t + 31].any(dim=0)
        valid += [(t, e) for e in ok.nonzero().flatten().tolist()]
    if not valid:
        raise RuntimeError('没有不跨回合的真实历史及32步窗口')
    generator = torch.Generator().manual_seed(43)
    chosen = torch.randperm(len(valid), generator=generator)[:batch_size].tolist()
    chosen = torch.tensor([valid[i] for i in chosen], device=env.device)
    t, e = chosen.unbind(-1)
    real_history = states[t[:, None] + torch.arange(-4, 1, device=env.device), e[:, None]]
    future = following[t[:, None] + torch.arange(32, device=env.device), e[:, None]]
    truth = torch.cat((states[t, e, None], future), dim=1)
    logged_actions = actions[t[:, None] + torch.arange(32, device=env.device), e[:, None]]
    context = runtime.context[e]
    predicted = rollout_logged_actions(trainer.world_model, truth[:, 0], logged_actions, context)
    if not bool(torch.isfinite(predicted).all()):
        raise RuntimeError('预测包含 NaN/Inf')
    frames = {}
    for name, trajectory in (('real', truth), ('predicted', predicted)):
        sequence = torch.cat((real_history, trajectory[:, 1:]), dim=1)
        frames[name] = trainer.amp_features(sequence.flatten(0, 1)).reshape(len(e), 37, -1)
    result = {}
    for h in (1, 8, 32):
        result[str(h)] = {
            'prediction': prediction_errors(predicted[:, h], truth[:, h], env.world_state_spec),
            'persistence': prediction_errors(truth[:, 0], truth[:, h], env.world_state_spec),
            'real_standing_quality': float(trainer.task.standing_quality(truth[:, h]).mean()),
            'predicted_standing_quality': float(trainer.task.standing_quality(predicted[:, h]).mean()),
        }
        for name in frames:
            window = frames[name][:, h:h + 5]
            reward = trainer.style.evaluate(window)
            result[str(h)][name + '_style_reward'] = float(reward.mean())
            result[str(h)][name + '_style_zero_fraction'] = float((reward == 0).float().mean())
    phase_errors = {}
    height = truth[:, 0, 2]
    for name, mask in (('low', height < .2), ('middle', (height >= .2) & (height < .48)), ('high', height >= .48)):
        if bool(mask.any()):
            phase_errors[name] = dict(count=int(mask.sum()), errors_32=prediction_errors(predicted[mask, 32], truth[mask, 32], env.world_state_spec))
    gradients = gradient_audit(trainer, real_history[:16], context[:16])
    from x2_style_mimic.algorithms.supermimic.state import unpack_state
    physical_states = truth[:, :-1].flatten(0, 1)
    raw_actions = logged_actions.flatten(0, 1)
    feature_context = context[:, None].expand(-1, 32, -1).flatten(0, 1)
    features, _ = trainer.world_model.features(physical_states, raw_actions, feature_context)
    z = ((features - trainer.world_model.input_mean) / trainer.world_model.input_std).abs()
    normalization = dict(abs_z_mean=float(z.mean()), abs_z_p99=float(torch.quantile(z.flatten(), .99)),
                         abs_z_over_10_fraction=float((z > 10).float().mean()))
    real_active = active_target_fraction(trainer.world_model.limiter, unpack_state(physical_states, env.world_state_spec), raw_actions)
    for name, module in runtime.modules.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, protected[name][key], atol=0, rtol=0)
    observations = trainer.proprioception.normalizer(trainer.proprioception.from_states(real_history))
    limiter = trainer.world_model.limiter
    desired = actions * limiter.scale + limiter.offset
    return dict(iteration=checkpoint['iteration'], sampled_windows=len(e), collection_steps=steps,
        endpoint_errors=result, phase_errors=phase_errors,
        gradient_components=gradients, world_input_standardized_values=normalization,
        real_target_derivative_active_fraction=real_active,
        action_rms=float(actions.square().mean().sqrt()),
        action_hard_limit_fraction=float(((desired < limiter.lower) | (desired > limiter.upper)).float().mean()),
        observation_clipped_fraction=float((observations.abs() >= 10).float().mean()),
        network_and_statistics_unchanged=True,
        note='固定评估出生状态；确定性策略；相同真实起点和动作；窗口可重叠；未更新参数、统计量或训练回放。')


def main():
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    add_launcher_args(parser)
    args, rest = setup_preset_cli(parser)
    if rest:
        parser.error(f'未识别参数：{rest}')
    cfg, _ = parse_config(['--resume', str(args.checkpoint_file)])
    if args.device is not None:
        cfg.device = args.device
    sys.argv = [sys.argv[0]]
    from getup_runtime import build_runtime
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_getup_env_cfg import X2SuperMimicGetupEnvCfg
    env_cfg = X2SuperMimicGetupEnvCfg()
    env_cfg.scene.num_envs, env_cfg.seed, env_cfg.sim.device = cfg.num_envs, cfg.seed, cfg.device
    with launch_simulation(env_cfg, args):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv
        env = X2SuperMimicEnv(env_cfg)
        try:
            saved = torch.load(args.checkpoint_file, map_location=cfg.device, weights_only=False)
            runtime = build_runtime(env, cfg, saved)
            report = audit(runtime, saved)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
            print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
