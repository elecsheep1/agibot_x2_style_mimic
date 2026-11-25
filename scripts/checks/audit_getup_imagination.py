"""只读诊断：检查 BPTT 想象轨迹的奖励结构、终止概率、动作和姿态，不更新任何参数。"""
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'supermimic'
sys.path.insert(0, str(SCRIPT_DIR))
from getup_config import parse_config

# 想象轨迹长度与本体历史帧数，必须与训练器 cfg 的口径一致。
IMAGINATION = 32
HISTORY_FRAMES = 5


def term_totals(reward_terms, exclude=('style',)):
    """按项汇总整条轨迹的任务奖励，区分正负来源。"""
    totals = {name: float(values.sum(0).mean())
              for name, values in reward_terms.items() if name not in exclude}
    positive = sum(value for value in totals.values() if value > 0)
    negative = sum(value for value in totals.values() if value < 0)
    gross = positive - negative
    return dict(terms=totals, positive_total=positive, negative_total=negative,
                # 负项占比按绝对值计算，0 表示全部为正。
                negative_fraction=(-negative / gross) if gross > 0 else 0.0)


def termination_summary(probabilities, threshold=0.5):
    """软终止概率的均值、首步值和超过阈值的比例。"""
    return dict(mean=float(probabilities.mean()),
                first_step_mean=float(probabilities[0].mean()),
                maximum=float(probabilities.max()),
                fraction_above_half=float((probabilities > threshold).float().mean()))


def action_summary(actions, lower, upper):
    """动作幅度、撞到 tanh 边界的比例和相邻帧变化量。"""
    span = (upper - lower).clamp_min(1e-6)
    normalized = (actions - lower) / span
    saturated = ((normalized <= 0.01) | (normalized >= 0.99)).float().mean()
    change = actions[1:] - actions[:-1]
    return dict(rms=float(actions.square().mean().sqrt()),
                maximum=float(actions.abs().max()),
                saturated_fraction=float(saturated),
                change_rms=float(change.square().mean().sqrt()))


def task_return(task, states, actions, goals):
    """沿固定动作序列累计任务奖励，风格奖励置零。

    states 为 [B, H+1, Ds]，actions 为 [B, H, A]，与 sample_windows 一致；返回 [B]。
    """
    import torch
    with torch.no_grad():
        total = states.new_zeros(states.shape[0])
        style = states.new_zeros(states.shape[0])
        for step in range(actions.shape[1]):
            reward, _ = task.reward(states[:, step], actions[:, step],
                                    states[:, step + 1], goals, style)
            total += reward
        return total


def main():
    if any(flag in sys.argv for flag in ('--help', '-h', '--print-config')):
        cfg, arguments = parse_config()
        if arguments.print_config:
            print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        return
    import numpy as np
    import torch
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
    from getup_runtime import build_runtime
    cfg, launcher = parse_config(launcher_hooks=(add_launcher_args, setup_preset_cli))
    cfg.validate()
    if cfg.resume is None:
        raise ValueError('想象轨迹诊断必须通过 --resume 指定检查点；不会执行恢复训练')
    cfg.prepare_output()
    sys.argv = [sys.argv[0]]
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_getup_env_cfg import X2SuperMimicGetupEnvCfg
    env_cfg = X2SuperMimicGetupEnvCfg()
    env_cfg.scene.num_envs, env_cfg.seed, env_cfg.sim.device = cfg.num_envs, cfg.seed, cfg.device
    with launch_simulation(env_cfg, launcher):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv
        from x2_style_mimic.algorithms.supermimic.buffer import SequenceReplayBuffer
        from x2_style_mimic.algorithms.supermimic.runner import SuperMimicRunner
        from x2_style_mimic.algorithms.supermimic.rollout import imagine
        from x2_style_mimic.algorithms.supermimic.world_model import rollout_logged_actions
        from x2_style_mimic.algorithms.supermimic.state import unpack_state
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        checkpoint = torch.load(cfg.resume, map_location=cfg.device, weights_only=False)
        env = X2SuperMimicEnv(env_cfg)
        try:
            runtime = build_runtime(env, cfg, checkpoint)
            cfg.output.mkdir(parents=True, exist_ok=False)
            trainer = runtime.trainer
            spec = env.world_state_spec
            horizon = IMAGINATION + HISTORY_FRAMES - 1
            collected = horizon + 33
            replay = SequenceReplayBuffer(env.num_envs, collected + 32, spec.dim, spec.num_actions,
                                          runtime.context, device=env.device, command_dim=0)
            runner = SuperMimicRunner(env, trainer, replay, None, action_noise_std=0.)
            world_hook = getattr(trainer.world_model, 'update_normalization', None)
            proprio_hook = trainer.proprioception.update_normalization
            try:
                if world_hook is not None:
                    trainer.world_model.update_normalization = lambda *args, **kwargs: None
                trainer.proprioception.update_normalization = lambda *args, **kwargs: None
                collection = runner.collect(collected)
            finally:
                if world_hook is not None:
                    trainer.world_model.update_normalization = world_hook
                trainer.proprioception.update_normalization = proprio_hook
            batch = replay.sample_windows(cfg.world_batch_size, horizon=horizon)
            states, actions, context = batch['states'], batch['actions'], batch['context']
            history = states[:, :HISTORY_FRAMES]
            real_states = states[:, HISTORY_FRAMES - 1:]
            real_actions = actions[:, HISTORY_FRAMES - 1:]
            with torch.no_grad():
                goals = states.new_empty((len(states), 0))
                trajectory = imagine(trainer.actor, trainer.world_model, trainer.proprioception,
                                     trainer.amp_features, trainer.style, trainer.task,
                                     history, context, goals, horizon=IMAGINATION)
                logged = rollout_logged_actions(trainer.world_model, real_states[:, 0],
                                                real_actions, context)
                actor_task = sum(trajectory.reward_terms[name].sum(0)
                                 for name in trajectory.reward_terms if name != 'style')
                model_task = task_return(trainer.task, logged, real_actions, goals)
                real_task = task_return(trainer.task, real_states, real_actions, goals)
                terms = term_totals(trajectory.reward_terms)
                style_return = float(trajectory.reward_terms['style'].sum(0).mean())
                height = lambda state: unpack_state(state, spec)['root_pose'][:, 2]
                def standing(state):
                    return float(trainer.task.standing_quality(state).mean())
                # 站立项只表示接近站立后的维持质量，低位恒为零；
                # 高度进展要看势函数。旧 source 没有这个方法时保持兼容。
                potential_before_fn = getattr(trainer.task, 'geometry_potential', None)
                def potential(state):
                    if potential_before_fn is None:
                        return None
                    return float(potential_before_fn(state).mean())
                report = dict(
                    checkpoint=str(cfg.resume.resolve()), iteration=checkpoint['iteration'],
                    imagination_steps=IMAGINATION, history_frames=HISTORY_FRAMES,
                    batch_size=len(states), collected_steps=collected,
                    termination=termination_summary(trajectory.termination_probabilities),
                    actions=action_summary(trajectory.actions, trainer.actor.action_lower,
                                           trainer.actor.action_upper),
                    returns=dict(actor_total=style_return + sum(terms['terms'].values()),
                                 actor_task=float(actor_task.mean()),
                                 actor_style=style_return,
                                 model_logged_task=float(model_task.mean()),
                                 real_task=float(real_task.mean()),
                                 actor_minus_real=float((actor_task - real_task).mean()),
                                 model_logged_minus_real=float((model_task - real_task).mean())),
                    terms=terms,
                    standing=dict(initial=standing(history[:, -1]),
                                  imagined_final=standing(trajectory.states[-1]),
                                  real_final=standing(real_states[:, -1])),
                    potential=dict(initial=potential(history[:, -1]),
                                   imagined_final=potential(trajectory.states[-1]),
                                   real_final=potential(real_states[:, -1])),
                    root_height=dict(initial=float(height(history[:, -1]).mean()),
                                     imagined_final=float(height(trajectory.states[-1]).mean()),
                                     real_final=float(height(real_states[:, -1]).mean())),
                    value=dict(terminal_mean=float(trainer.critic(
                                   trajectory.states[-1], context, goals).mean()),
                               cap=float(trainer.task.max_step_reward / (1 - cfg.gamma))),
                    collection=collection,
                    note='只读诊断：不更新任何参数或统计量。actor 想象与模型内真实动作对照共用一个起始状态；'
                         '动作序列不同，差值只能描述，不能当作因果结论。')
            (cfg.output / 'imagination_report.json').write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            print('[想象诊断] 回报=' + json.dumps(report['returns'], ensure_ascii=False), flush=True)
            print('[想象诊断] 终止=' + json.dumps(report['termination'], ensure_ascii=False), flush=True)
            print('[想象诊断] 姿态=' + json.dumps(report['standing'], ensure_ascii=False), flush=True)
            print('[想象诊断] 势函数=' + json.dumps(report['potential'], ensure_ascii=False), flush=True)
            print(f'[完成] {cfg.output}', flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
