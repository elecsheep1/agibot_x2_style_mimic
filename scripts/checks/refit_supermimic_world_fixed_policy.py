"""固定策略采集完整回合，仅补训世界模型，按回合留出验证长轨迹误差。"""
import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'supermimic'
sys.path.insert(0, str(SCRIPT_DIR))
from getup_data import valid_windows, sample


def reset_collection(runtime):
    """先让 runner 标记手动回合边界，再丢弃预采样转移。"""
    runtime.runner.reset()
    runtime.replay.total_steps = 0


def episode_split(data, seed):
    """每个物理环境保留一个完整回合作验证，其余完整回合作训练。"""
    generator = torch.Generator().manual_seed(seed)
    train, holdout = [], []
    done = data['terminated'] | data['truncated']
    for env in range(done.shape[1]):
        complete = []
        for episode in data['episode_ids'][:, env].unique().tolist():
            ids = torch.where(data['episode_ids'][:, env] == episode)[0]
            steps = data['step_ids'][ids, env]
            if len(ids) < 32 or int(steps[0]) != 0 or not bool(done[ids[-1], env]):
                continue
            if bool(done[ids[:-1], env].any()) or not bool((steps[1:] == steps[:-1] + 1).all()):
                raise ValueError('回合编号或连续步数不一致')
            complete.append((env, episode))
        if len(complete) < 2:
            raise ValueError(f'环境 {env} 不足两个完整回合，不能独立划分训练与留出集')
        selected = int(torch.randint(len(complete), (1,), generator=generator))
        holdout.append(complete[selected])
        train.extend(key for i, key in enumerate(complete) if i != selected)
    assert not set(train) & set(holdout)
    return train, holdout


def candidates_for(data, horizon, episode_keys):
    pool = valid_windows(data, horizon)
    allowed = set(map(tuple, episode_keys))
    mask = torch.tensor([(int(env), int(data['episode_ids'][start, env])) in allowed
                         for start, env in pool], dtype=torch.bool)
    result = pool[mask]
    if not len(result):
        raise ValueError('该回合划分没有有效连续窗口')
    return result


def fixed_evaluation_batch(data, pool, batch_size, seed, dt):
    """固定一半早期、一半后期窗口，防止长时间趴地掩盖起身阶段误差。"""
    generator = torch.Generator().manual_seed(seed)
    elapsed = data['step_ids'][pool[:, 0], pool[:, 1]] * dt
    pools = (pool[elapsed < 3.], pool[elapsed >= 3.])
    if any(not len(part) for part in pools):
        raise ValueError('早期或后期验收窗口不足')
    count = batch_size // 2
    selected = torch.cat([part[torch.randint(len(part), (size,), generator=generator)]
                          for part, size in zip(pools, (count, batch_size - count))])
    start, env = selected.unbind(-1)
    time = start[:, None] + torch.arange(32)
    envs = env[:, None].expand_as(time)
    return dict(states=torch.cat((data['state'][start, env, None], data['next_state'][time, envs]), 1),
                actions=data['action'][time, envs], context=data['contexts'][env],
                phase=torch.cat((torch.zeros(count, dtype=torch.long),
                                 torch.ones(batch_size - count, dtype=torch.long))),
                selected_windows=selected)


@torch.no_grad()
def evaluate_world(model, batch, device, spec):
    from x2_style_mimic.algorithms.supermimic.world_model import rollout_logged_actions
    from audit_supermimic_training import prediction_errors
    states, actions, context = (batch[k].to(device) for k in ('states', 'actions', 'context'))
    predicted = rollout_logged_actions(model, states[:, 0], actions, context)
    if not bool(torch.isfinite(predicted).all()):
        raise RuntimeError('长轨迹预测包含 NaN/Inf')
    report = {}
    for phase, mask in [('balanced', torch.ones(len(states), dtype=torch.bool)),
                        ('early', batch['phase'] == 0), ('late', batch['phase'] == 1)]:
        mask = mask.to(device)
        report[phase] = {str(h): prediction_errors(predicted[mask, h], states[mask, h], spec)
                         for h in (1, 8, 32)}
    return report


@torch.no_grad()
def evaluate_reward_direction(trainer, batch, device):
    """用真实五帧历史起步，对照同动作八步的高度与几何奖励。"""
    from x2_style_mimic.algorithms.supermimic.world_model import rollout_logged_actions
    states, actions, context = (batch[k].to(device) for k in ('states', 'actions', 'context'))
    history, truth, actions = states[:, :5], states[:, 4:13], actions[:, 4:12]
    predicted = rollout_logged_actions(trainer.world_model, truth[:, 0], actions, context)
    def score(sequence):
        window = trainer.amp_features(history.flatten(0, 1)).reshape(len(states), 5, -1)
        total = torch.zeros(len(states), device=device)
        for t in range(8):
            window = torch.cat((window[:, 1:], trainer.amp_features(sequence[:, t + 1])[:, None]), 1)
            reward, _ = trainer.task.reward(sequence[:, t], actions[:, t], sequence[:, t + 1],
                                            torch.empty(len(states), 0, device=device), trainer.style(window))
            total += reward
        return total
    real_return, model_return = score(truth), score(predicted)
    real_delta, model_delta = truth[:, -1, 2] - truth[:, 0, 2], predicted[:, -1, 2] - truth[:, 0, 2]
    moving = real_delta.abs() > .005
    return dict(return_mae=float((model_return - real_return).abs().mean()),
                return_bias=float((model_return - real_return).mean()),
                real_return_mean=float(real_return.mean()), model_return_mean=float(model_return.mean()),
                height_change_mae_m=float((model_delta - real_delta).abs().mean()),
                height_direction_count=int(moving.sum()),
                height_direction_agreement=float((real_delta[moving] * model_delta[moving] > 0).float().mean())
                    if bool(moving.any()) else None)


def main():
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
    from getup_config import parse_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261122)
    parser.add_argument('--actor-probe-updates', type=int, default=0,
                        help='采集前在冻结模型上更新 actor 的次数，默认不更新')
    add_launcher_args(parser)
    args, rest = setup_preset_cli(parser)
    if rest or args.output.exists() or args.actor_probe_updates < 0 or min(args.updates, args.batch_size, args.eval_batch_size) < 2:
        parser.error('输出目录必须不存在，批量和更新次数至少为2，不接受额外参数')
    cfg, _ = parse_config(['--resume', str(args.checkpoint)])
    if args.device is not None:
        cfg.device = args.device
    sys.argv = [sys.argv[0]]
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_getup_env_cfg import X2SuperMimicGetupEnvCfg
    env_cfg = X2SuperMimicGetupEnvCfg()
    env_cfg.scene.num_envs, env_cfg.seed, env_cfg.sim.device = cfg.num_envs, cfg.seed, cfg.device
    with launch_simulation(env_cfg, args):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv
        from getup_runtime import build_runtime
        from getup_training import file_hashes
        source_before = file_hashes((SCRIPT_DIR.parents[1] / 'source').rglob('*.py'))
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        checkpoint = torch.load(args.checkpoint, map_location=cfg.device, weights_only=False)
        env = X2SuperMimicEnv(env_cfg)
        try:
            runtime = build_runtime(env, cfg, checkpoint)
            args.output.mkdir(parents=True)
            trainer = runtime.trainer
            world_update_normalization = getattr(trainer.world_model, 'update_normalization', None)
            if world_update_normalization is not None:
                trainer.world_model.update_normalization = lambda *args: None
            update_normalization = trainer.proprioception.update_normalization
            trainer.proprioception.update_normalization = lambda states: None
            runtime.runner.action_noise_std = 0.
            if args.actor_probe_updates:
                from x2_style_mimic.algorithms.supermimic.rollout import imagine
                from x2_style_mimic.algorithms.supermimic.objectives import imagined_actor_loss
                from x2_style_mimic.algorithms.supermimic.gradients import frozen_parameters
                runtime.runner.collect(256)
                torch.manual_seed(args.seed)
                probe = runtime.replay.sample_windows(args.batch_size, 12)
                optimizer = torch.optim.Adam(trainer.actor.parameters(), lr=.0002, betas=(.9, .95))
                with frozen_parameters(trainer.world_model, trainer.critic, trainer.style,
                                       trainer.proprioception, trainer.amp_features, trainer.task):
                    for _ in range(args.actor_probe_updates):
                        optimizer.zero_grad(set_to_none=True)
                        trajectory = imagine(trainer.actor, trainer.world_model, trainer.proprioception,
                                             trainer.amp_features, trainer.style, trainer.task,
                                             probe['states'][:, :5], probe['context'],
                                             torch.empty(args.batch_size, 0, device=cfg.device), horizon=8)
                        terminal = trainer.critic(trajectory.states[-1], probe['context'], trajectory.goals[-1])
                        loss = imagined_actor_loss(trajectory.rewards, trajectory.termination_probabilities,
                                                    terminal, cfg.gamma)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainer.actor.parameters(), 1., error_if_nonfinite=True)
                        for parameter in trainer.actor.parameters():
                            if parameter.grad is not None:
                                parameter.grad.clamp_(-.1, .1)
                        optimizer.step()
                torch.save(trainer.actor.state_dict(), args.output / 'fixed_actor.pt')
                print(f'[固定新策略] actor 已更新 {args.actor_probe_updates} 次；接下来仅补训模型', flush=True)
                reset_collection(runtime)
            protected = {name: {key: value.detach().clone() for key, value in module.state_dict().items()}
                         for name, module in runtime.modules.items() if name != 'world_model'}
            stats = {name: getattr(trainer.world_model, name).detach().clone()
                     for name in ('input_mean', 'input_std', 'stats_ready')}
            for module in runtime.modules.values():
                module.eval()
            steps = 2 * env.max_episode_length + 4
            if steps > runtime.replay.capacity:
                raise ValueError('回放容量不足以保留两个完整回合')
            print(f'[固定策略采集] {env.num_envs}环境，{steps}控制步，关闭探索噪声并冻结统计量', flush=True)
            for start in range(0, steps, 128):
                runtime.runner.collect(min(128, steps - start))
                print(f'[采集] {min(start + 128, steps)}/{steps}', flush=True)
            trainer.proprioception.update_normalization = update_normalization
            if world_update_normalization is not None:
                trainer.world_model.update_normalization = world_update_normalization
            replay = runtime.replay
            data = {name: getattr(replay, name)[:steps].detach().cpu().clone()
                    for name in ('state', 'next_state', 'action', 'terminated', 'truncated', 'episode_ids', 'step_ids')}
            data['contexts'] = replay.contexts.detach().cpu().clone()
            torch.save(data, args.output / 'collected_data.pt')
            train_keys, holdout_keys = episode_split(data, args.seed)
            train_pool = candidates_for(data, 8, train_keys)
            train_eval = fixed_evaluation_batch(data, candidates_for(data, 32, train_keys),
                                               args.eval_batch_size, args.seed + 1, env.step_dt)
            holdout_eval = fixed_evaluation_batch(data, candidates_for(data, 32, holdout_keys),
                                                 args.eval_batch_size, args.seed + 2, env.step_dt)
            torch.save(dict(data=data, train_keys=train_keys, holdout_keys=holdout_keys,
                            train_evaluation=train_eval, holdout_evaluation=holdout_eval), args.output / 'dataset.pt')
            history = []
            def check(update):
                row = dict(update=update,
                           train=evaluate_world(trainer.world_model, train_eval, cfg.device, env.world_state_spec),
                           holdout=evaluate_world(trainer.world_model, holdout_eval, cfg.device, env.world_state_spec),
                           holdout_reward_direction=evaluate_reward_direction(trainer, holdout_eval, cfg.device))
                history.append(row)
                (args.output / 'progress.json').write_text(json.dumps(history, ensure_ascii=False, indent=2))
                print('[留出误差]', update, json.dumps(row['holdout']['balanced'], ensure_ascii=False), flush=True)
            print(f'[回合划分] train={len(train_keys)}，holdout={len(holdout_keys)}，训练窗口={len(train_pool)}', flush=True)
            check(0)
            torch.manual_seed(args.seed)
            loss_log = []
            for update in range(1, args.updates + 1):
                batch = sample(data, train_pool, args.batch_size, 8, cfg.device)
                metrics = trainer.update_world_model(batch)
                loss_log.append(dict(update=update, **metrics))
                if update == 1 or update % 50 == 0:
                    print(f'[仅模型补训 {update}/{args.updates}] loss={metrics["world_loss"]:.6f}', flush=True)
                if update in (100, 250, 500) or update == args.updates:
                    check(update)
            for name, module in runtime.modules.items():
                if name in protected:
                    for key, value in module.state_dict().items():
                        torch.testing.assert_close(value, protected[name][key], atol=0, rtol=0)
            for name, value in stats.items():
                torch.testing.assert_close(getattr(trainer.world_model, name), value, atol=0, rtol=0)
            assert source_before == file_hashes((SCRIPT_DIR.parents[1] / 'source').rglob('*.py'))
            report = dict(checkpoint=str(args.checkpoint.resolve()), updates=args.updates,
                          actor_probe_updates=args.actor_probe_updates,
                          horizon=8, batch_size=args.batch_size, train_episodes=train_keys,
                          holdout_episodes=holdout_keys, history=history, loss_log=loss_log,
                          frozen_actor_critic_discriminator_and_statistics=True, formal_source_unchanged=True,
                          collection_noise=0., evaluation_sampling='固定窗口，早期<3秒和后期各占一半；窗口可重叠',
                          note='只验证固定策略分布上的模型补训，不选择留出集最优权重，不证明策略会因此学会起身。')
            (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            torch.save(dict(model=trainer.world_model.state_dict(),
                            optimizers={name: optimizer.state_dict() for name, optimizer in runtime.optimizers.items()
                                        if name == 'world' or name.startswith('world/')},
                            base_checkpoint=str(args.checkpoint.resolve()), metadata=runtime.metadata,
                            updates=args.updates), args.output / 'world_model_refit.pt')
            print('[完成]', args.output, flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
