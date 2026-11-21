"""Exercise the production velocity SuperMimic/AMP loop; not a performance benchmark."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from collect_supermimic_smoke import shared_vector


def tensor(value):
    return value.torch if hasattr(value, 'torch') else value


def finite_metrics(metrics):
    result = {name: float(value) for name, value in metrics.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError(f'Nonfinite metrics: {result}')
    return result


def snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def change(module, before):
    return max((float((p.detach() - old).abs().max())
                for p, old in zip(module.parameters(), before)), default=0.0)


def main(task_name='velocity'):
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-manifest', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, default=Path('/home/panda/robotics_ws/agibot_model/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra.urdf'))
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--imagination-horizon', type=int, default=8)
    parser.add_argument('--world-model-init', type=Path, default=None, help='Initialize only world model from a supervised checkpoint')
    parser.add_argument('--offline-world-data', type=Path, default=None)
    parser.add_argument('--offline-world-split', type=Path, default=None)
    parser.add_argument('--offline-world-fraction', type=float, default=0.5)
    parser.add_argument('--warmup-steps', type=int, default=64)
    parser.add_argument('--warmup-updates', type=int, default=50)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--audit', action='store_true', help='Audit gradient isolation and fresh 1/8/32-step predictions')
    parser.add_argument('--output', type=Path, default=Path(f'logs/supermimic/{task_name}_training_smoke_v1'))
    add_launcher_args(parser)
    args, rest = setup_preset_cli(parser)
    if rest:
        parser.error(str(rest))
    if min(args.num_envs, args.iterations, args.batch_size, args.warmup_updates) < 1 or args.warmup_steps < 16:
        parser.error('Positive dimensions and warmup-steps >= 16 required')
    if not 1 <= args.imagination_horizon <= 32:
        parser.error('imagination-horizon must be in [1, 32]')
    if not args.motion_manifest.is_file() or not args.urdf.is_file():
        parser.error('Manifest and URDF must exist')
    if args.output.exists():
        parser.error('Choose a fresh output directory')
    if args.world_model_init is not None and not args.world_model_init.is_file():
        parser.error('world-model-init must exist')
    if not math.isfinite(args.offline_world_fraction) or not 0 <= args.offline_world_fraction <= 1:
        parser.error('offline-world-fraction must be in [0, 1]')
    if (args.offline_world_data is None) != (args.offline_world_split is None):
        parser.error('Provide both offline-world-data and offline-world-split')
    if args.offline_world_data is not None and (not args.offline_world_data.is_file() or not args.offline_world_split.is_file()):
        parser.error('Offline dataset and split report must exist')
    sys.argv = [sys.argv[0]]
    if task_name == 'getup':
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_getup_env_cfg import X2SuperMimicGetupEnvCfg
        cfg = X2SuperMimicGetupEnvCfg()
    else:
        parser.error('速度任务已移除；请使用 train_supermimic_getup_smoke.py 或 scripts/supermimic/train_getup.py')
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if task_name == 'velocity':
        cfg.commands.base_velocity.debug_vis = False
    if args.device is not None:
        cfg.sim.device = args.device

    with launch_simulation(cfg, args):
        from x2_style_mimic.assets.x2_ultra_cfg import X2_ULTRA_USD
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.motion_dataset import MotionDataset
        from x2_style_mimic.algorithms.amp.discriminator import AMPWindowDiscriminator
        from x2_style_mimic.algorithms.supermimic.actor_critic import GoalConditionedActor, GoalConditionedCritic
        from x2_style_mimic.algorithms.supermimic.amp_features import DifferentiableAMPFeatures
        from x2_style_mimic.algorithms.supermimic.buffer import SequenceReplayBuffer
        from x2_style_mimic.algorithms.supermimic.context import read_sim_context
        from x2_style_mimic.algorithms.supermimic.heads import VelocityTask, VelocityTaskCfg
        from x2_style_mimic.algorithms.supermimic.observations import Proprioception
        from x2_style_mimic.algorithms.supermimic.runner import SuperMimicRunner
        from x2_style_mimic.algorithms.supermimic.style import DifferentiableStyleReward
        from x2_style_mimic.algorithms.supermimic.trainer import SuperMimicTrainer, TrainerCfg
        from x2_style_mimic.algorithms.supermimic.world_model import TargetLimiter, WorldModel

        torch.manual_seed(args.seed)
        env = X2SuperMimicEnv(cfg)
        try:
            env.reset()
            device, spec, robot = env.device, env.world_state_spec, env.scene['robot']
            context, context_spec = read_sim_context(env)
            action_term = env.action_manager.get_term('joint_pos')
            ids = torch.arange(spec.num_joints, device=device)[action_term._joint_ids]
            limits = tensor(robot.data.joint_pos_limits)
            values = dict(offset=action_term._offset, scale=action_term._scale,
                          lower=limits[:, ids, 0], upper=limits[:, ids, 1],
                          max_delta=tensor(robot.data.joint_vel_limits)[:, ids] * action_term.cfg.target_velocity_fraction * env.step_dt)
            values = {name: shared_vector(value, spec.num_actions, name, device) for name, value in values.items()}
            proprio = Proprioception(spec, ids, tensor(robot.data.default_joint_pos)[0, ids],
                                    tensor(robot.data.default_joint_vel)[0, ids], history_length=5).to(device)
            features = DifferentiableAMPFeatures.from_urdf(spec, str(args.urdf.resolve()),
                                                          list(robot.joint_names), usd_path=X2_ULTRA_USD).to(device)
            names = [robot.joint_names[i] for i in ids.tolist()]
            reference = MotionDataset(str(args.motion_manifest.resolve()), list(robot.joint_names), names, device, env.step_dt)
            discriminator = AMPWindowDiscriminator(torch.cat(reference.amp_states), num_steps=5).to(device)
            style = DifferentiableStyleReward(discriminator).to(device)
            goal_dim = 0 if task_name == 'getup' else 3
            command_name = None if task_name == 'getup' else 'base_velocity'
            actor = GoalConditionedActor(proprio.output_dim, goal_dim, spec.num_actions).to(device)
            critic = GoalConditionedCritic(spec.dim, context.shape[-1], goal_dim).to(device)
            world_checkpoint = None
            hidden_dims = (256, 256, 256)
            if args.world_model_init is not None:
                from supermimic_checkpoint import read_world_checkpoint
                expected_metadata = dict(state_spec=asdict(spec), context_spec=asdict(context_spec),
                                         joint_names=list(robot.joint_names), control_joint_names=names,
                                         step_dt=env.step_dt, physics_dt=cfg.sim.dt, decimation=cfg.decimation)
                world_checkpoint = read_world_checkpoint(args.world_model_init, expected_metadata,
                                                          values, context.shape[-1], device)
                hidden_dims = tuple(world_checkpoint['hidden_dims'])
            world = WorldModel(spec, context.shape[-1], TargetLimiter(**values).to(device), hidden_dims).to(device)
            if world_checkpoint is not None:
                world.load_state_dict(world_checkpoint['model'], strict=True)
                print(f'[world model] loaded {args.world_model_init}', flush=True)
            trainer_cfg = TrainerCfg(horizon=args.imagination_horizon)
            if task_name == 'getup':
                from x2_style_mimic.algorithms.supermimic.heads import GetupTask, GetupTaskCfg
                task_cfg = GetupTaskCfg(**cfg.getup_task)
                task = GetupTask(spec, limits[0], task_cfg, features.kinematics, ids).to(device)
            else:
                task_cfg = VelocityTaskCfg(**cfg.velocity_task)
                task = VelocityTask(spec, limits[0], task_cfg).to(device)
            trainer = SuperMimicTrainer(actor, critic, world, proprio, features, style, task, trainer_cfg)
            replay = SequenceReplayBuffer(env.num_envs, max(256, args.warmup_steps + 32), spec.dim,
                                          spec.num_actions, context, device=device, command_dim=goal_dim)
            runner = SuperMimicRunner(env, trainer, replay, command_name, action_noise_std=0.1)
            offline_world = None
            if args.offline_world_data is not None:
                if not hasattr(runner, 'sample_world_batch'):
                    raise RuntimeError('Apply the documented runner.py sample_world_batch change before enabling offline-world-data')
                from supermimic_offline_world import OfflineWorldReplay
                offline_world = OfflineWorldReplay(args.offline_world_data, args.offline_world_split,
                    dict(state_spec=asdict(spec), context_spec=asdict(context_spec), joint_names=list(robot.joint_names),
                         control_joint_names=names, step_dt=env.step_dt, physics_dt=cfg.sim.dt, decimation=cfg.decimation),
                    values, device)
                print(f'[offline world] {len(offline_world.train_keys)} train episodes; {len(offline_world.holdout_keys)} held out', flush=True)
            task_setup = dict(task_name=task_name, goal_dim=goal_dim)
            if task_name == 'getup':
                current = env.read_world_state()
                termination = env.termination_manager.get_term_cfg('out_of_bounds')
                torch.testing.assert_close(task.terminated(current), termination.func(env, **termination.params))
                assert cfg.terminations.fallen is None and cfg.commands is None
                task_setup.update(initial_root_heights=current[:, 2].tolist(),
                                  initial_birth_types=env._getup_birth_types.tolist(),
                                  initial_reference_frames=env._getup_reference_frame_ids.tolist(),
                                  birth_probabilities=list(cfg.events.reset_from_reference.params['birth_probabilities']),
                                  assistance_enabled=False, low_height_termination=False,
                                  contact_reward_mode='geometry_and_stability_only')
                print('[任务配置] ' + json.dumps(task_setup), flush=True)
            modules = dict(actor=actor, critic=critic, world_model=world, proprioception=proprio,
                           amp_features=features, style=style, task=task)
            initial = {name: snapshot(module) for name, module in dict(actor=actor, critic=critic, world_model=world, discriminator=discriminator).items()}
            args.output.mkdir(parents=True, exist_ok=False)
            pretrained_audit = None
            if world_checkpoint is not None:
                from audit_supermimic_training import audit_predictions
                pretrained_audit = audit_predictions(env, trainer, context, max(64, args.batch_size), args.seed + 2000)
                (args.output / 'pretrained_world_audit.json').write_text(
                    json.dumps(pretrained_audit, indent=2, allow_nan=False), encoding='utf-8')
                print('[pretrained world audit] ' + json.dumps(pretrained_audit), flush=True)
                # The audit used this simulator: reset the training history as well.
                runner.reset()
            print('[warmup] collecting real transitions', flush=True)
            runner.collect(args.warmup_steps)
            for update in range(args.warmup_updates):
                world_batch = (replay.sample_windows(args.batch_size, horizon=8) if offline_world is None else
                               runner.sample_world_batch(args.batch_size, 8, offline_world, args.offline_world_fraction))
                metrics = finite_metrics(trainer.update_world_model(world_batch))
                if (update + 1) % 10 == 0:
                    print(f'[world warmup {update + 1}] {metrics}', flush=True)
            for iteration in range(args.iterations):
                mixed_args = {} if offline_world is None else dict(offline_world_replay=offline_world,
                                                                       offline_world_fraction=args.offline_world_fraction)
                metrics = finite_metrics(runner.train_iteration(reference, collection_steps=32,
                                        batch_size=args.batch_size, world_horizon=8, world_updates=10, discriminator_updates=4, **mixed_args))
                metrics['iteration'] = iteration + 1
                with (args.output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + '\n')
                print(json.dumps(metrics), flush=True)
            audit = {}
            if args.audit:
                from audit_supermimic_training import audit_gradients, audit_predictions
                audit['gradient_isolation'] = audit_gradients(trainer, replay, args.batch_size)
                audit['prediction'] = audit_predictions(env, trainer, context, max(64, args.batch_size), args.seed + 1000)
                print('[audit] ' + json.dumps(audit), flush=True)
            final_context, final_spec = read_sim_context(env)
            assert final_spec == context_spec
            torch.testing.assert_close(final_context, context, atol=0, rtol=0)
            for name, module in modules.items():
                for value in module.state_dict().values():
                    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                        raise RuntimeError(f'Nonfinite module: {name}')
            changes = {name: change(module, initial[name]) for name, module in
                       dict(actor=actor, critic=critic, world_model=world, discriminator=discriminator).items()}
            if not all(value > 0 for value in changes.values()):
                raise RuntimeError(f'A network did not update: {changes}')
            metadata = dict(offline_world_data=str(args.offline_world_data.resolve()) if args.offline_world_data else None,
                            offline_world_split=str(args.offline_world_split.resolve()) if args.offline_world_split else None,
                            offline_world_fraction=args.offline_world_fraction if offline_world is not None else 0.0,
                            world_model_init=str(args.world_model_init.resolve()) if args.world_model_init else None,
                            world_model_init_sha256=hashlib.sha256(args.world_model_init.read_bytes()).hexdigest() if args.world_model_init else None,
                            task_name=task_name, task_setup=task_setup, goal_dim=goal_dim, state_spec=asdict(spec), context_spec=asdict(context_spec),
                            joint_names=list(robot.joint_names), control_joint_names=names,
                            world_hidden_dims=hidden_dims, trainer_cfg=asdict(trainer_cfg), task_cfg=asdict(task_cfg),
                            motion_manifest=str(args.motion_manifest.resolve()), urdf=str(args.urdf.resolve()),
                            usd=str(X2_ULTRA_USD), step_dt=env.step_dt, seed=args.seed,
                            reference_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in reference.paths})
            checkpoint = dict(kind=f'supermimic_{task_name}_smoke', iteration=args.iterations,
                              total_env_steps=runner.total_env_steps, metadata=metadata,
                              modules={name: module.state_dict() for name, module in modules.items()},
                              optimizers=dict(world=trainer.world_optimizer.state_dict(), actor=trainer.actor_optimizer.state_dict(),
                                              critic=trainer.critic_optimizer.state_dict(), discriminator=discriminator.optimizer.state_dict()))
            checkpoint_path = args.output / 'checkpoint.pt'
            torch.save(checkpoint, checkpoint_path)
            loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
            for name, module in modules.items():
                module.load_state_dict(loaded['modules'][name], strict=True)
            report = dict(passed=True, pretrained_audit=pretrained_audit, task_setup=task_setup, audit=audit, iterations=args.iterations, parameter_max_changes=changes,
                          checkpoint_reload=True, total_transitions=runner.total_env_steps * env.num_envs,
                          reference_note='仅验证起身训练链路，尚未验收起身成功率。' if task_name == 'getup' else 'Reference suitability for locomotion is not assessed; engineering smoke test only.')
            (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
