"""固定策略采集完整回合，仅补训世界模型，按回合与按环境留出验证长轨迹误差。"""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'supermimic'
sys.path.insert(0, str(SCRIPT_DIR))
from getup_data import valid_windows, sample

# 预测诊断统一评估的转移步数：1 步为当前步，32 步为 0.64 秒。
EVALUATION_HORIZONS = (1, 8, 16, 32)
# 固定验收窗口长度，取最长评估步数。
HORIZON = max(EVALUATION_HORIZONS)
# 补训检查点默认台阶，实际只取不超过总更新次数的值。
CHECKPOINT_LADDER = (100, 250, 500, 1000, 2000, 5000)
# 回合内早期与后期的分界时间，用于避免长时间趴地掩盖起身阶段误差。
EARLY_PHASE_SECONDS = 3.0


def reset_collection(runtime):
    """先让 runner 标记手动回合边界，再丢弃预采样转移。"""
    runtime.runner.reset()
    runtime.replay.total_steps = 0


def freeze_normalization(trainer):
    """冻结世界模型与本体观测的归一化统计量，返回原函数用于恢复。

    runner 用关键字参数调用世界模型归一化，替换函数必须同时接受
    位置参数和关键字参数，否则采集第一步就会报错。
    """
    world = getattr(trainer.world_model, 'update_normalization', None)
    proprioception = trainer.proprioception.update_normalization
    if world is not None:
        trainer.world_model.update_normalization = lambda *args, **kwargs: None
    trainer.proprioception.update_normalization = lambda *args, **kwargs: None
    return world, proprioception


def restore_normalization(trainer, originals):
    """恢复被 freeze_normalization 替换的归一化更新函数。"""
    world, proprioception = originals
    trainer.proprioception.update_normalization = proprioception
    if world is not None:
        trainer.world_model.update_normalization = world


def world_window(cfg):
    """把配置中的状态帧数换算为转移数；训练配置以状态帧计数。

    world_horizons 表示状态帧数，底层采样接口的 horizon 表示转移数，
    所以 8 帧窗口对应 7 条转移。配置多个窗口时取最长者。
    """
    horizons = tuple(cfg.world_horizons)
    if not horizons:
        raise ValueError('world_horizons 不能为空')
    frames = max(horizons)
    if frames < 2:
        raise ValueError('世界模型窗口至少需要两帧状态')
    return frames, frames - 1


def check_updates(total):
    """返回实际评估的更新次数，保证同一轮不会被重复评估。"""
    if total < 1:
        raise ValueError('补训次数必须为正整数')
    return sorted({0, total, *(value for value in CHECKPOINT_LADDER if value < total)})


def complete_episodes(data):
    """按环境列出完整且连续的回合键 [环境编号, 回合编号]。

    只保留从回合第 0 步开始、最后一步已结束、且中间没有结束标志的回合。
    不完整回合直接丢弃，既不进入训练池，也不进入留出集。
    """
    done = data['terminated'] | data['truncated']
    result = {}
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
        result[env] = complete
    return result


def split_environments(complete, seed, count):
    """整组留出若干环境，用于检验模型对新物理随机化的泛化。

    count 为留出环境数量；0 表示不做环境级留出。留出环境的全部回合
    都不参与补训，否则物理 参数相同的训练回合仍会泄漏信息。
    """
    envs = sorted(complete)
    if not 0 <= count < len(envs):
        raise ValueError(f'留出环境数量必须在 [0, {len(envs)}) 内')
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(envs), generator=generator).tolist()
    holdout = {envs[index] for index in order[:count]}
    train = [env for env in envs if env not in holdout]
    if not train:
        raise ValueError('全部环境都被留出，没有训练环境')
    return train, holdout


def split_episodes(complete, envs, seed):
    """每个环境保留一个完整回合作验证，其余完整回合作训练。"""
    generator = torch.Generator().manual_seed(seed)
    train, holdout = [], []
    for env in envs:
        keys = complete[env]
        if len(keys) < 2:
            raise ValueError(f'环境 {env} 不足两个完整回合，不能独立划分训练与留出集')
        selected = int(torch.randint(len(keys), (1,), generator=generator))
        holdout.append(keys[selected])
        train.extend(key for index, key in enumerate(keys) if index != selected)
    if not train or not holdout:
        raise ValueError('训练回合或留出回合为空')
    assert not set(train) & set(holdout)
    return train, holdout


def episode_split(data, seed):
    """兼容接口：对全部环境做按回合划分。"""
    complete = complete_episodes(data)
    return split_episodes(complete, sorted(complete), seed)


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
    """固定一半早期、一半后期窗口，且批内不重复抽取同一窗口。

    固定窗口保证补训前后比较的是同一批起点；不重复抽取避免同一窗口
    在批内重复出现而放大统计显著性。窗口之间仍可能在时间上重叠。
    """
    if batch_size < 2:
        raise ValueError('验收批量至少为 2，才能覆盖早期和后期')
    generator = torch.Generator().manual_seed(seed)
    elapsed = data['step_ids'][pool[:, 0], pool[:, 1]] * dt
    pools = (pool[elapsed < EARLY_PHASE_SECONDS], pool[elapsed >= EARLY_PHASE_SECONDS])
    if any(not len(part) for part in pools):
        raise ValueError('早期或后期验收窗口不足')
    count = batch_size // 2
    parts = []
    for part, size in zip(pools, (count, batch_size - count)):
        if len(part) < size:
            raise ValueError(f'该阶段可用窗口 {len(part)} 个，不足以无重复抽取 {size} 个')
        parts.append(part[torch.randperm(len(part), generator=generator)[:size]])
    selected = torch.cat(parts)
    if len({tuple(row) for row in selected.tolist()}) != len(selected):
        raise ValueError('验收窗口出现重复')
    start, env = selected.unbind(-1)
    time = start[:, None] + torch.arange(HORIZON)
    envs = env[:, None].expand_as(time)
    return dict(states=torch.cat((data['state'][start, env, None], data['next_state'][time, envs]), 1),
                actions=data['action'][time, envs], context=data['contexts'][env],
                phase=torch.cat((torch.zeros(count, dtype=torch.long),
                                 torch.ones(batch_size - count, dtype=torch.long))),
                selected_windows=selected,
                episodes=len({(int(e), int(data['episode_ids'][s, e])) for s, e in selected.tolist()}))


def root_distance_quantiles(predicted, target, spec):
    """逐样本根位置误差的分位数；均值会掩盖少量长步失败。"""
    from x2_style_mimic.algorithms.supermimic.state import unpack_state
    p, t = unpack_state(predicted, spec), unpack_state(target, spec)
    distance = (p['root_pose'][:, :3] - t['root_pose'][:, :3]).detach().cpu().norm(dim=-1)
    return dict(median_m=float(distance.median()), p90_m=float(distance.quantile(.9)),
                max_m=float(distance.max()))


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
        if not bool(mask.any()):
            raise ValueError(f'验收窗口在 {phase} 阶段为空')
        report[phase] = {str(h): dict(prediction_errors(predicted[mask, h], states[mask, h], spec),
                                      root_distance_m=root_distance_quantiles(
                                          predicted[mask, h], states[mask, h], spec))
                         for h in EVALUATION_HORIZONS}
    return report


@torch.no_grad()
def evaluate_reward_direction(trainer, batch, device):
    """用真实五帧历史起步，对照同动作八步的高度与几何奖励。"""
    from x2_style_mimic.algorithms.supermimic.world_model import rollout_logged_actions
    states, actions, context = (batch[k].to(device) for k in ('states', 'actions', 'context'))
    if states.shape[1] < 13 or actions.shape[1] < 12:
        raise ValueError('奖励方向评估至少需要 12 条转移')
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


def summarize(row):
    """只取平衡阶段的根位置 RMSE，便于逐行比较补训前后的变化。"""
    return {name: {h: round(row[name]['balanced'][str(h)]['root_position_rmse_m'], 5)
                   for h in EVALUATION_HORIZONS}
            for name in ('holdout', 'holdout_environment') if name in row}


def main():
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
    from getup_config import parse_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--holdout-envs', type=int, default=4,
                        help='整组留出的环境数量；0 表示不做环境级留出')
    parser.add_argument('--seed', type=int, default=20261122)
    parser.add_argument('--actor-probe-updates', type=int, default=0,
                        help='采集前在冻结模型上更新 actor 的次数，默认不更新')
    add_launcher_args(parser)
    args, rest = setup_preset_cli(parser)
    if rest or args.output.exists() or args.actor_probe_updates < 0 or args.holdout_envs < 0 \
            or min(args.updates, args.batch_size, args.eval_batch_size) < 2:
        parser.error('输出目录必须不存在，批量和更新次数至少为2，留出环境数不能为负，不接受额外参数')
    cfg, _ = parse_config(['--resume', str(args.checkpoint)])
    cfg.validate()
    if args.device is not None:
        cfg.device = args.device
    if args.holdout_envs >= cfg.num_envs:
        parser.error('留出环境数量必须小于环境总数')
    window_frames, window_transitions = world_window(cfg)
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
            normalization = freeze_normalization(trainer)
            runtime.runner.action_noise_std = 0.
            try:
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
                # 固定策略后重新建立回合边界，并保证记录从第 0 步开始。
                reset_collection(runtime)
                protected = {name: {key: value.detach().clone() for key, value in module.state_dict().items()}
                             for name, module in runtime.modules.items() if name != 'world_model'}
                stats = {name: getattr(trainer.world_model, name).detach().clone()
                         for name in ('input_mean', 'input_std', 'stats_ready')}
                for module in runtime.modules.values():
                    module.eval()
                # 世界模型保持训练模式，与联合训练口径一致；当前网络没有 dropout 或 BN。
                trainer.world_model.train()
                steps = 2 * env.max_episode_length + 4
                if steps > runtime.replay.capacity:
                    raise ValueError(f'回放容量 {runtime.replay.capacity} 步，'
                                     f'不足以保留两个完整回合所需的 {steps} 步')
                print(f'[固定策略采集] {env.num_envs}环境，{steps}控制步（回放容量 {runtime.replay.capacity}），'
                      f'关闭探索噪声并冻结统计量', flush=True)
                for start in range(0, steps, 128):
                    runtime.runner.collect(min(128, steps - start))
                    print(f'[采集] {min(start + 128, steps)}/{steps}', flush=True)
            finally:
                restore_normalization(trainer, normalization)
            data = {name: getattr(runtime.replay, name)[:steps].detach().cpu().clone()
                    for name in ('state', 'next_state', 'action', 'terminated', 'truncated', 'episode_ids', 'step_ids')}
            data['contexts'] = runtime.replay.contexts.detach().cpu().clone()
            torch.save(data, args.output / 'collected_data.pt')
            complete = complete_episodes(data)
            train_envs, holdout_envs = split_environments(complete, args.seed, args.holdout_envs)
            train_keys, holdout_keys = split_episodes(complete, train_envs, args.seed + 1)
            holdout_environment_keys = [key for env_id in sorted(holdout_envs) for key in complete[env_id]]
            train_pool = candidates_for(data, window_transitions, train_keys)
            train_eval = fixed_evaluation_batch(data, candidates_for(data, HORIZON, train_keys),
                                                args.eval_batch_size, args.seed + 2, env.step_dt)
            holdout_eval = fixed_evaluation_batch(data, candidates_for(data, HORIZON, holdout_keys),
                                                  args.eval_batch_size, args.seed + 3, env.step_dt)
            holdout_environment_eval = (fixed_evaluation_batch(
                data, candidates_for(data, HORIZON, holdout_environment_keys),
                args.eval_batch_size, args.seed + 4, env.step_dt) if holdout_environment_keys else None)
            torch.save(dict(data=data, train_keys=train_keys, holdout_keys=holdout_keys,
                            holdout_environment_keys=holdout_environment_keys,
                            train_evaluation=train_eval, holdout_evaluation=holdout_eval,
                            holdout_environment_evaluation=holdout_environment_eval), args.output / 'dataset.pt')
            history = []
            def check(update):
                row = dict(update=update,
                           train=evaluate_world(trainer.world_model, train_eval, cfg.device, env.world_state_spec),
                           holdout=evaluate_world(trainer.world_model, holdout_eval, cfg.device, env.world_state_spec),
                           holdout_reward_direction=evaluate_reward_direction(trainer, holdout_eval, cfg.device))
                if holdout_environment_eval is not None:
                    row['holdout_environment'] = evaluate_world(
                        trainer.world_model, holdout_environment_eval, cfg.device, env.world_state_spec)
                    row['holdout_environment_reward_direction'] = evaluate_reward_direction(
                        trainer, holdout_environment_eval, cfg.device)
                history.append(row)
                (args.output / 'progress.json').write_text(json.dumps(history, ensure_ascii=False, indent=2))
                print('[留出误差]', update, json.dumps(summarize(row), ensure_ascii=False), flush=True)
            print(f'[回合划分] 训练环境={len(train_envs)}，整组留出环境={sorted(holdout_envs)}，'
                  f'训练回合={len(train_keys)}，回合留出={len(holdout_keys)}，训练窗口={len(train_pool)}；'
                  f'窗口帧数={window_frames}，转移数={window_transitions}', flush=True)
            schedule = check_updates(args.updates)
            print(f'[评估台阶] {schedule}', flush=True)
            check(0)
            torch.manual_seed(args.seed)
            loss_log = []
            for update in range(1, args.updates + 1):
                batch = sample(data, train_pool, args.batch_size, window_transitions, cfg.device)
                metrics = trainer.update_world_model(batch)
                loss_log.append(dict(update=update, **metrics))
                if update == 1 or update % 50 == 0:
                    print(f'[仅模型补训 {update}/{args.updates}] loss={metrics["world_loss"]:.6f}', flush=True)
                if update in schedule:
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
                          configured_world_horizons=list(cfg.world_horizons),
                          window_frames=window_frames, transitions_per_window=window_transitions,
                          evaluation_horizons=list(EVALUATION_HORIZONS),
                          batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                          check_updates=schedule,
                          train_envs=train_envs, holdout_envs=sorted(holdout_envs),
                          train_episodes=train_keys, holdout_episodes=holdout_keys,
                          holdout_environment_episodes=holdout_environment_keys,
                          collected_steps=steps, collected_transitions=steps * env.num_envs,
                          replay_capacity=runtime.replay.capacity,
                          frozen_actor_critic_discriminator_and_statistics=True,
                          formal_source_unchanged=True, collection_noise=0.,
                          evaluation_sampling='固定窗口，早期<3秒和后期各占一半，批内不重复；窗口之间可能重叠',
                          note='只验证固定策略分布上的模型补训，不按留出集挑选权重，'
                               '不证明策略会因此学会起身。')
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
