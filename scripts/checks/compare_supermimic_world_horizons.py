"""固定数据、起点与验收窗口，对比世界模型的预测训练长度。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

from collect_supermimic_smoke import shared_vector
from fit_supermimic_smoke import valid_windows, sample
from supermimic_checkpoint import read_world_checkpoint
from supermimic_offline_world import OfflineWorldReplay
from audit_supermimic_training import prediction_errors


def training_schedules():
    # 同时保留更新次数匹配和预测转移数匹配的两个 8 步对照。
    return {'h8_same_updates': (8, 8, 8),
            'h8_same_transitions': (8, 8, 8, 8, 8, 8, 8),
            'h8_16_32': (8, 16, 32)}


def select_env_windows(data, env_ids, horizon=32):
    candidates = valid_windows(data, horizon)
    allowed = torch.tensor(env_ids, dtype=torch.long)
    result = candidates[torch.isin(candidates[:, 1], allowed)]
    if not len(result):
        raise ValueError('指定环境没有可用的连续窗口')
    return result


def tensor(value):
    return value.torch if hasattr(value, 'torch') else value


def main():
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--actor-checkpoint', type=Path, required=True)
    parser.add_argument('--offline-data', type=Path, required=True)
    parser.add_argument('--offline-split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--collection-steps', type=int, default=512)
    parser.add_argument('--cycles', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=45)
    add_launcher_args(parser)
    args, rest = setup_preset_cli(parser)
    if rest or args.num_envs < 4 or args.collection_steps < 64 or args.cycles < 1 or args.batch_size < 2 or args.batch_size % 2:
        parser.error('请检查参数：至少4个环境、64步采集、正循环数，批量为正偶数')
    if args.output.exists():
        parser.error('输出目录已存在，请指定新目录')
    for path in (args.base_model, args.actor_checkpoint, args.offline_data, args.offline_split):
        if not path.is_file():
            parser.error(f'文件不存在：{path}')
    sys.argv = [sys.argv[0]]
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_getup_env_cfg import X2SuperMimicGetupEnvCfg
    cfg = X2SuperMimicGetupEnvCfg()
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    if args.device is not None:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv
        from x2_style_mimic.algorithms.supermimic.actor_critic import GoalConditionedActor
        from x2_style_mimic.algorithms.supermimic.context import read_sim_context
        from x2_style_mimic.algorithms.supermimic.observations import Proprioception
        from x2_style_mimic.algorithms.supermimic.buffer import SequenceReplayBuffer
        from x2_style_mimic.algorithms.supermimic.runner import SuperMimicRunner
        from x2_style_mimic.algorithms.supermimic.world_model import WorldModel, TargetLimiter, rollout_logged_actions, dynamics_loss

        torch.manual_seed(args.seed)
        args.output.mkdir(parents=True, exist_ok=False)
        env = X2SuperMimicEnv(cfg)
        try:
            env.reset()
            spec, device, robot = env.world_state_spec, env.device, env.scene['robot']
            context, context_spec = read_sim_context(env)
            term = env.action_manager.get_term('joint_pos')
            ids = torch.arange(spec.num_joints, device=device)[term._joint_ids]
            names = [robot.joint_names[i] for i in ids.tolist()]
            limits = tensor(robot.data.joint_pos_limits)
            values = dict(offset=term._offset, scale=term._scale, lower=limits[:, ids, 0], upper=limits[:, ids, 1],
                          max_delta=tensor(robot.data.joint_vel_limits)[:, ids] * term.cfg.target_velocity_fraction * env.step_dt)
            values = {k: shared_vector(v, spec.num_actions, k, device) for k, v in values.items()}
            expected = dict(state_spec=asdict(spec), context_spec=asdict(context_spec), joint_names=list(robot.joint_names),
                            control_joint_names=names, step_dt=env.step_dt, physics_dt=cfg.sim.dt, decimation=cfg.decimation)
            base = read_world_checkpoint(args.base_model, expected, values, context.shape[-1], device)
            actor_checkpoint = torch.load(args.actor_checkpoint, map_location=device, weights_only=False)
            for key in ('state_spec', 'context_spec', 'joint_names', 'control_joint_names'):
                if actor_checkpoint['metadata'][key] != expected[key]:
                    raise ValueError(f'采集策略与当前环境不兼容：{key}')
            if actor_checkpoint['metadata']['goal_dim'] != 0 or actor_checkpoint['metadata']['task_name'] != 'getup':
                raise ValueError('必须使用无外部目标的起身策略 checkpoint')
            proprio = Proprioception(spec, ids, tensor(robot.data.default_joint_pos)[0, ids],
                                    tensor(robot.data.default_joint_vel)[0, ids], history_length=5).to(device)
            proprio.load_state_dict(actor_checkpoint['modules']['proprioception'], strict=True)
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'supermimic'))
            from getup_data import action_bounds_from_calibration
            action_lower, action_upper = action_bounds_from_calibration(values)
            actor = GoalConditionedActor(proprio.output_dim, 0, spec.num_actions,
                                         action_lower=action_lower, action_upper=action_upper).to(device)
            actor.load_state_dict(actor_checkpoint['modules']['actor'], strict=True)
            actor.eval()
            replay = SequenceReplayBuffer(env.num_envs, args.collection_steps, spec.dim, spec.num_actions,
                                          context, device=device, command_dim=0)
            collector = SuperMimicRunner(env, SimpleNamespace(device=device, actor=actor, proprioception=proprio,
                                          style=SimpleNamespace(discriminator=SimpleNamespace(num_steps=5))),
                                          replay, None, action_noise_std=.1)
            print('[采集] 固定当前策略与动作噪声，生成新真实转移', flush=True)
            collection = collector.collect(args.collection_steps)
            final_context, final_spec = read_sim_context(env)
            assert final_spec == context_spec
            torch.testing.assert_close(final_context, context, atol=0, rtol=0)
            online = {name: getattr(replay, name).detach().cpu().clone() for name in
                      ('state', 'next_state', 'action', 'rewards', 'terminated', 'truncated', 'episode_ids', 'step_ids', 'contexts')}
            online['metadata'] = expected
            torch.save(online, args.output / 'online_transitions.pt')
            offline = OfflineWorldReplay(args.offline_data, args.offline_split, expected, values, device)
            holdout_count = max(1, env.num_envs // 4)
            # 整个环境的数据都留出，训练窗口不会经过这些环境。
            permutation = torch.randperm(env.num_envs, generator=torch.Generator().manual_seed(6001))
            test_envs, train_envs = permutation[:holdout_count].tolist(), permutation[holdout_count:].tolist()
            online_train = select_env_windows(online, train_envs)
            online_test = select_env_windows(online, test_envs)
            all_offline = valid_windows(offline.data, 32)
            mask = torch.tensor([(int(e), int(offline.data['episode_ids'][s, e])) in offline.holdout_keys
                                 for s, e in all_offline], dtype=torch.bool)
            offline_test = all_offline[mask]
            if not len(offline_test):
                raise ValueError('旧数据没有可用的32步留出窗口')
            torch.manual_seed(6002)
            tests = dict(online_holdout=sample(online, online_test, 128, 32, device),
                         offline_holdout=sample(offline.data, offline_test, 128, 32, device))
            # 每个循环预采样一批完整32步轨迹。三组模型读取完全相同的起点和动作。
            torch.manual_seed(6003)
            batches = []
            for _ in range(args.cycles):
                old = offline.sample_windows(args.batch_size // 2, 32)
                new = sample(online, online_train, args.batch_size // 2, 32, device)
                batches.append({k: torch.cat((old[k], new[k]), dim=0) for k in ('states', 'actions', 'context')})

            def make_world():
                model = WorldModel(spec, context.shape[-1], TargetLimiter(**values).to(device), base['hidden_dims']).to(device)
                model.load_state_dict(base['model'], strict=True)
                return model

            @torch.no_grad()
            def evaluate(model):
                result = {}
                for name, batch in tests.items():
                    prediction = rollout_logged_actions(model, batch['states'][:, 0], batch['actions'], batch['context'])
                    if not bool(torch.isfinite(prediction).all()):
                        raise RuntimeError('验收预测包含 NaN/Inf')
                    result[name] = {str(h): prediction_errors(prediction[:, h], batch['states'][:, h], spec) for h in (1, 8, 32)}
                return result

            report = dict(seed=args.seed, collection=collection, train_envs=train_envs, holdout_envs=test_envs,
                          fixed_batch_starts=True, fixed_validation_windows=True, offline_fraction=.5,
                          cycles=args.cycles, batch_size=args.batch_size,
                          offline_train_episodes=len(offline.train_keys), offline_holdout_episodes=len(offline.holdout_keys),
                          baseline=evaluate(make_world()), arms={},
                          note='匹配预测转移数不等于匹配优化器更新次数或精确运行时间，因此同时提供两种8步对照。仅比较世界模型，不更新actor/critic/判别器。')
            for name, schedule in training_schedules().items():
                model = make_world()
                optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
                start = time.monotonic()
                steps = 0
                for cycle, full in enumerate(batches):
                    for horizon in schedule:
                        batch = dict(states=full['states'][:, :horizon + 1], actions=full['actions'][:, :horizon], context=full['context'])
                        optimizer.zero_grad(set_to_none=True)
                        prediction = rollout_logged_actions(model, batch['states'][:, 0], batch['actions'], batch['context'])
                        loss, _ = dynamics_loss(prediction, batch['states'], spec)
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError('世界模型训练损失包含 NaN/Inf')
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        steps += 1
                    if (cycle + 1) % 4 == 0:
                        print(f'[{name}] 循环 {cycle + 1}/{args.cycles}，损失 {float(loss.detach()):.6f}', flush=True)
                report['arms'][name] = dict(schedule=schedule, updates=steps,
                    trained_transitions=args.cycles * args.batch_size * sum(schedule),
                    train_seconds=time.monotonic() - start, errors=evaluate(model))
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), metadata=expected,
                                hidden_dims=tuple(base['hidden_dims']), limiter=values, context_dim=context.shape[-1],
                                schedule=schedule, source_actor=str(args.actor_checkpoint.resolve())), args.output / f'{name}.pt')
            report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (args.base_model, args.actor_checkpoint, args.offline_data, args.offline_split)}
            (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            print('[完成] 对照报告：' + str(args.output / 'report.json'), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
