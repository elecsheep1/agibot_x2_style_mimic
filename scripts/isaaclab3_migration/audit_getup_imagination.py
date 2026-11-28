#!/usr/bin/env python3
"""Run the SuperMimic imagination audit on Isaac Lab 3.

Reuse the launcher and reset shims from ``train_supermimic_getup.py`` and the
training runtime's own runner/replay, i.e. the exact path the migrated trainer
already uses.  The original audit script in ``scripts/checks`` is untouched.
"""

from __future__ import annotations

import json
import importlib.util
import sys
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SUPERMIMIC_SCRIPTS = PROJECT_ROOT / "scripts" / "supermimic"
CHECKS = PROJECT_ROOT / "scripts" / "checks"


def _load_original_audit_module():
    """Load the original audit helpers without importing this launcher again."""
    module_path = CHECKS / "audit_getup_imagination.py"
    spec = importlib.util.spec_from_file_location("x2_original_imagination_audit", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load audit helpers: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_launcher_compat():
    """Install the same Isaac Lab 3 compatibility shims as the trainer."""
    from isaaclab.app import AppLauncher
    import isaaclab_tasks.utils as task_utils

    from train_supermimic_getup import (
        _install_supermimic_runtime_compat,
        _launch_simulation_compat,
        _setup_preset_cli_compat,
    )

    task_utils.add_launcher_args = AppLauncher.add_app_launcher_args
    task_utils.launch_simulation = _launch_simulation_compat
    task_utils.setup_preset_cli = _setup_preset_cli_compat

    import getup_runtime

    _install_supermimic_runtime_compat(getup_runtime)
    return getup_runtime


def _prime_runner_from_current_environment(runner) -> None:
    """Initialize a runner history without issuing a second ``env.reset()``.

    ``build_runtime`` already resets the freshly-created environment.  The
    legacy runner's ``reset`` method unconditionally resets it again.  On the
    Isaac Lab 3/PhysX migration path, that redundant manual reset can make Kit
    close cleanly before Python gets a chance to report an exception.  For a
    read-only audit we only need a valid initial world-state history, which is
    exactly what the first reset has already provided.
    """
    current = runner.env.read_world_state().to(runner.device)
    runner.history = current[:, None].expand(-1, runner.history_length, -1).clone()
    runner._read_commands()
    # Be explicit: the audit must not inherit an old replay timeline.
    runner.replay.total_steps = 0
    print("[想象诊断] 已复用环境初始化状态（不执行重复 env.reset）", flush=True)


def main() -> None:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(SUPERMIMIC_SCRIPTS))
    sys.path.insert(0, str(CHECKS))

    getup_runtime = _install_launcher_compat()

    import torch
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli

    from getup_config import parse_config
    audit = _load_original_audit_module()

    cfg, launcher = parse_config(launcher_hooks=(add_launcher_args, setup_preset_cli))
    cfg.validate()
    if cfg.resume is None:
        raise ValueError("想象轨迹诊断必须通过 --resume 指定检查点；不会执行恢复训练")
    cfg.prepare_output()
    sys.argv = [sys.argv[0]]

    env_cfg = getup_runtime.training_env_cfg(cfg)
    with launch_simulation(env_cfg, launcher):
        # Keep every task/algorithm import below Kit launch.  Importing them
        # earlier can preload pxr/USD and break Isaac Sim 6.2 on the remote host.
        from x2_style_mimic.algorithms.supermimic.rollout import imagine
        from x2_style_mimic.algorithms.supermimic.state import unpack_state
        from x2_style_mimic.algorithms.supermimic.world_model import rollout_logged_actions

        with getup_runtime.training_environment(cfg, env_cfg) as (env, checkpoint):
            if checkpoint is None:
                raise RuntimeError("审计入口未能读取 --resume 指定的检查点")
            runtime = getup_runtime.build_runtime(env, cfg, checkpoint)
            cfg.output.mkdir(parents=True, exist_ok=False)

            trainer = runtime.trainer
            runner, replay = runtime.runner, runtime.replay
            spec = env.world_state_spec
            horizon = audit.IMAGINATION + audit.HISTORY_FRAMES - 1
            collected = horizon + 33
            if collected > replay.capacity:
                raise ValueError(f"回放容量 {replay.capacity} 不足以采集 {collected} 步")

            print(f"[想象诊断] 使用 {env.num_envs} 环境，采集 {collected} 步", flush=True)
            runner.action_noise_std = 0.0
            world_hook = getattr(trainer.world_model, "update_normalization", None)
            proprio_hook = trainer.proprioception.update_normalization
            try:
                if world_hook is not None:
                    trainer.world_model.update_normalization = lambda *args, **kwargs: None
                trainer.proprioception.update_normalization = lambda *args, **kwargs: None
                print("[想象诊断] 正在初始化采样历史", flush=True)
                _prime_runner_from_current_environment(runner)
                print("[想象诊断] 正在采集真实物理轨迹", flush=True)
                collection = runner.collect(collected)
                print("[想象诊断] 真实轨迹采集完成，正在计算想象对照", flush=True)
            finally:
                if world_hook is not None:
                    trainer.world_model.update_normalization = world_hook
                trainer.proprioception.update_normalization = proprio_hook

            batch = replay.sample_windows(cfg.world_batch_size, horizon=horizon)
            states, actions, context = batch["states"], batch["actions"], batch["context"]
            history = states[:, :audit.HISTORY_FRAMES]
            real_states = states[:, audit.HISTORY_FRAMES - 1:]
            real_actions = actions[:, audit.HISTORY_FRAMES - 1:]

            with torch.no_grad():
                goals = states.new_empty((len(states), 0))
                trajectory = imagine(trainer.actor, trainer.world_model, trainer.proprioception,
                                     trainer.amp_features, trainer.style, trainer.task,
                                     history, context, goals, horizon=audit.IMAGINATION)
                logged = rollout_logged_actions(trainer.world_model, real_states[:, 0],
                                                real_actions, context)
                actor_task = sum(trajectory.reward_terms[name].sum(0)
                                 for name in trajectory.reward_terms if name != "style")
                model_task = audit.task_return(trainer.task, logged, real_actions, goals)
                real_task = audit.task_return(trainer.task, real_states, real_actions, goals)
                terms = audit.term_totals(trajectory.reward_terms)
                style_return = float(trajectory.reward_terms["style"].sum(0).mean())
                height = lambda state: unpack_state(state, spec)["root_pose"][:, 2]

                def standing(state):
                    return float(trainer.task.standing_quality(state).mean())

                potential_fn = getattr(trainer.task, "geometry_potential", None)

                def potential(state):
                    return None if potential_fn is None else float(potential_fn(state).mean())

                report = dict(
                    checkpoint=str(cfg.resume.resolve()), iteration=checkpoint["iteration"],
                    imagination_steps=audit.IMAGINATION, history_frames=audit.HISTORY_FRAMES,
                    batch_size=len(states), collected_steps=collected,
                    termination=audit.termination_summary(trajectory.termination_probabilities),
                    actions=audit.action_summary(trajectory.actions, trainer.actor.action_lower,
                                                 trainer.actor.action_upper),
                    returns=dict(actor_total=style_return + sum(terms["terms"].values()),
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
                    note="只读诊断：不更新任何参数或统计量。actor 想象与模型内真实动作对照共用一个起始状态；"
                         "动作序列不同，差值只能描述，不能当作因果结论。")

            (cfg.output / "imagination_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            print("[想象诊断] 回报=" + json.dumps(report["returns"], ensure_ascii=False), flush=True)
            print("[想象诊断] 终止=" + json.dumps(report["termination"], ensure_ascii=False), flush=True)
            print("[想象诊断] 姿态=" + json.dumps(report["standing"], ensure_ascii=False), flush=True)
            print("[想象诊断] 势函数=" + json.dumps(report["potential"], ensure_ascii=False), flush=True)
            print(f"[完成] {cfg.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
