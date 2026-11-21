"""Audit differentiable AMP against PhysX, plus gradients and reset isolation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

from collect_supermimic_smoke import make_collection_config


def errors(predicted, actual):
    difference = (predicted - actual).detach().double()
    return {"max": float(difference.abs().max()), "rms": float(difference.square().mean().sqrt())}


def main():
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--usd-origins', action='store_true', help='Use the production kinematics USD alignment method')
    parser.add_argument('--output', type=Path, default=Path('logs/supermimic/amp_features_acceptance'))
    parser.add_argument('--world-model', type=Path, default=Path('logs/supermimic/fit_noise005_capacity256_v1/world_model.pt'))
    add_launcher_args(parser)
    args, rest = setup_preset_cli(parser)
    if rest:
        parser.error(str(rest))
    if args.num_envs < 2 or args.steps < 12:
        parser.error('Use at least two environments and twelve steps')
    if args.output.exists():
        parser.error('Choose a fresh output directory')
    sys.argv = [sys.argv[0]]
    cfg = make_collection_config(args)
    cfg.episode_length_s = .24
    with launch_simulation(cfg, args):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.amp import AMP_KEY_BODY_NAMES, AMP_STATE_DIM, build_amp_state
        from x2_style_mimic.algorithms.supermimic.amp_features import DifferentiableAMPFeatures
        from x2_style_mimic.algorithms.supermimic.gradients import frozen_parameters
        from x2_style_mimic.algorithms.supermimic.style import DifferentiableStyleReward
        from x2_style_mimic.algorithms.supermimic.context import read_sim_context
        from x2_style_mimic.algorithms.supermimic.state import unpack_state
        from x2_style_mimic.algorithms.supermimic.world_model import WorldModel, TargetLimiter
        from x2_style_mimic.algorithms.amp.discriminator import AMPWindowDiscriminator
        from x2_style_mimic.algorithms.amp.history import AMPHistory
        from x2_style_mimic.utils.rotation import quaternion_angle_error

        torch.manual_seed(args.seed)
        env = X2SuperMimicEnv(cfg=cfg)
        report = {"cases": {}, "scope": "feature and gradient correctness, untrained discriminator, no policy-quality claim"}
        try:
            env.reset()
            robot = env.scene['robot']
            spec = env.world_state_spec
            urdf = '/home/panda/robotics_ws/agibot_model/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra.urdf'
            feature = DifferentiableAMPFeatures.from_urdf(spec, urdf, robot.joint_names).to(device=env.device, dtype=torch.float32)
            if args.usd_origins:
                from x2_style_mimic.assets.x2_ultra_cfg import X2_ULTRA_USD

                feature.kinematics.align_joint_origins_from_usd(
                    X2_ULTRA_USD
                )
                report['usd_alignment'] = {
                    'usd_path': str(X2_ULTRA_USD),
                    'implementation': 'DifferentiableKinematics.align_joint_origins_from_usd',
                }
            body_ids = [robot.body_names.index(name) for name in AMP_KEY_BODY_NAMES]
            all_ids = torch.arange(env.num_envs, device=env.device)
            state_history = []

            def audit(label):
                state = env.read_world_state()
                with torch.no_grad():
                    packed = feature(state)
                    root = robot.data.root_link_pose_w.torch
                    velocity = robot.data.root_link_vel_w.torch
                    body_pose = robot.data.body_link_pose_w.torch[:, body_ids].clone()
                    body_vel = robot.data.body_link_vel_w.torch[:, body_ids]
                    actual = build_amp_state(root[:,:3], root[:,3:], velocity[:,:3], velocity[:,3:], body_pose, body_vel, env.scene.env_origins[:,2:3])
                    fields = unpack_state(state, spec)
                    pose, speed = feature.kinematics(fields['root_pose'], fields['root_velocity'], fields['joint_position'], fields['joint_velocity'])
                    body_pose[:,:,:3] -= env.scene.env_origins[:,None,:]
                    angle = quaternion_angle_error(pose[:,:,3:], body_pose[:,:,3:])
                    parts = {'root':(0,10), 'body_position':(10,43), 'body_rotation6d':(43,109), 'body_linear':(109,142), 'body_angular_scaled':(142,175)}
                    data = {name:errors(packed[:,a:b], actual[:,a:b]) for name,(a,b) in parts.items()}
                    data['pose_position'] = errors(pose[:,:,:3],body_pose[:,:,:3])
                    data['pose_angle_max_rad'] = float(angle.max())
                    data['linear_velocity'] = errors(speed[:,:,:3],body_vel[:,:,:3])
                    data['angular_velocity'] = errors(speed[:,:,3:],body_vel[:,:,3:])
                    linear_difference = (
                        packed[:, 109:142] - actual[:, 109:142]
                    ).reshape(env.num_envs, len(body_ids), 3)
                    largest = int(linear_difference.abs().argmax())
                    environment = largest // (len(body_ids) * 3)
                    body_index = (largest // 3) % len(body_ids)
                    component = largest % 3
                    data['worst_relative_linear'] = {
                        'environment': environment,
                        'body': AMP_KEY_BODY_NAMES[body_index],
                        'component': 'xyz'[component],
                        'difference': float(linear_difference[environment, body_index, component]),
                        'body_world_linear_speed': float(body_vel[environment, body_index, :3].norm()),
                        'body_world_angular_speed': float(body_vel[environment, body_index, 3:].norm()),
                        'root_world_angular_speed': float(velocity[environment, 3:].norm()),
                        'max_joint_speed': float(fields['joint_velocity'][environment].abs().max()),
                    }
                    data['per_body'] = {
                        name: {
                            'position_max_m': float((pose[:, index, :3] - body_pose[:, index, :3]).abs().max()),
                            'angle_max_rad': float(angle[:, index].max()),
                            'world_linear_max_m_s': float((speed[:, index, :3] - body_vel[:, index, :3]).abs().max()),
                            'world_angular_max_rad_s': float((speed[:, index, 3:] - body_vel[:, index, 3:]).abs().max()),
                            'relative_linear_max_m_s': float(linear_difference[:, index].abs().max()),
                        }
                        for index, name in enumerate(AMP_KEY_BODY_NAMES)
                    }
                    report['cases'][label] = data
                    state_history.append(state.detach().clone())
                    return packed, actual

            for noise, label in [(0.,'standing'),(.03,'perturbed_standing')]:
                root = robot.data.default_root_state.torch.clone()
                root[:,:3] += env.scene.env_origins
                root[:,7:] = 0
                q = robot.data.default_joint_pos.torch.clone()
                q += torch.randn_like(q)*noise
                limits = robot.data.joint_pos_limits.torch
                q = q.clamp(limits[:,:,0], limits[:,:,1])
                robot.write_root_state_to_sim(root, env_ids=all_ids)
                robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=all_ids)
                robot.reset(all_ids)
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(env.step_dt)
                audit(label)

            history = AMPHistory(env.num_envs, AMP_STATE_DIM, 5, env.device)
            episode = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            tags = torch.full((env.num_envs,5), -1, dtype=torch.long,device=env.device)
            resets = valid_count = 0
            for step in range(args.steps):
                current = feature(env.read_world_state()).detach()
                history.begin(current)
                fresh = history.counts == 1
                tags[fresh] = -1
                tags[fresh,-1] = episode[fresh]
                with torch.no_grad():
                    action = torch.randn(env.num_envs,spec.num_actions,device=env.device)*.08
                    result = env.step(action)
                    transition = result[-1]['world_model']
                    done = transition['terminated'] | transition['truncated']
                    windows, valid = history.advance(feature(transition['next_state']),done)
                    tags[:,:-1] = tags[:,1:].clone()
                    tags[:,-1] = episode
                    if bool(valid.any()):
                        assert bool((tags[valid] == episode[valid,None]).all())
                    assert bool((history.counts[done] == 0).all())
                    tags[done] = -1
                    episode += done.long()
                    resets += int(done.sum())
                    valid_count += int(valid.sum())
                audit(f'moving_{step:03d}')
            report['reset_isolation'] = dict(resets=resets, valid_windows=valid_count, passed=resets>0 and valid_count>0)
            context, _ = read_sim_context(env)
            checkpoint = torch.load(args.world_model, map_location=env.device, weights_only=True)
            assert asdict(spec) == checkpoint['metadata']['state_spec']
            if 'joint_names' in checkpoint['metadata']:
                assert list(robot.joint_names) == checkpoint['metadata']['joint_names']
            model = WorldModel(spec,checkpoint['context_dim'],TargetLimiter(**checkpoint['limiter']),hidden_dims=checkpoint['hidden_dims']).to(env.device)
            model.load_state_dict(checkpoint['model'])
            samples = torch.stack(state_history[-5:],1)
            reference = feature(samples.flatten(0,1)).detach()
            discriminator = AMPWindowDiscriminator(reference,num_steps=5).to(env.device)
            style = DifferentiableStyleReward(discriminator)
            initial = samples[:,-1]
            action = torch.zeros(env.num_envs,spec.num_actions,device=env.device,requires_grad=True)
            with frozen_parameters(model,style):
                following = model(initial,action,context)
                next_features = feature(following)
                past = feature(samples[:,-4:].reshape(-1,spec.dim)).reshape(env.num_envs,4,AMP_STATE_DIM)
                window = torch.cat((past,next_features[:,None]),1)
                score = discriminator(window)
                reward = style(window)
                gradient = torch.autograd.grad(reward.sum(),action)[0]
            report['gradient'] = dict(score_min=float(score.detach().min()),score_max=float(score.detach().max()),reward_mean=float(reward.detach().mean()),action_gradient_norm=float(gradient.norm()),finite=bool(gradient.isfinite().all()),nonzero=bool(gradient.abs().max()>0),frozen_parameters_have_no_grad=all(p.grad is None for p in list(model.parameters())+list(style.parameters())))
            passed = all(max(c['body_position']['max'],c['body_rotation6d']['max'],c['body_linear']['max'],c['body_angular_scaled']['max']) < 5e-4 for c in report['cases'].values())
            report['passed'] = passed and report['reset_isolation']['passed'] and report['gradient']['finite'] and report['gradient']['nonzero']
            args.output.mkdir(parents=True)
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
            if not report['passed']:
                raise RuntimeError('Acceptance failed; inspect report.json')
        finally:
            env.close()


if __name__ == '__main__':
    main()
