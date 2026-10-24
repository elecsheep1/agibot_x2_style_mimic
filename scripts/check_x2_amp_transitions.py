"""检查统一起身任务的五帧 AMP、终止状态捕获和跨回合隔离，不更新策略。"""

import argparse
import sys

TASK = "X2-Prone-MultiMotion-v0"


def main():
    # Delay simulator imports so --help does not launch or initialize physics.
    from isaaclab_tasks.utils import (
        add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser)
    if args.num_envs < 2 or args.steps < 20:
        parser.error("诊断至少使用 2 个环境和 20 个控制步")
    sys.argv = [sys.argv[0]] + remaining

    import gymnasium as gym
    import torch
    import x2_style_mimic.tasks  # noqa: F401

    cfg, agent_cfg = resolve_task_config(TASK, "amp_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    cfg.episode_length_s = 0.20
    cfg.assist_force_levels = (0.0,)
    cfg.assist_initial_stage = 0
    cfg.assist_zero_probability = 1.0
    cfg.assist_curriculum_enabled = False
    if args.device is not None:
        cfg.sim.device = args.device

    with launch_simulation(cfg, args):
        from x2_style_mimic.algorithms.amp import (
            AMPWindowDiscriminator, AMPWindowReplayBuffer,
        )
        from x2_style_mimic.algorithms.amp.history import AMPHistory
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.amp import AMP_STATE_DIM
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.motion_dataset import MotionDataset

        env = gym.make(TASK, cfg=cfg)
        base = env.unwrapped
        try:
            amp_cfg = agent_cfg.algorithm.amp_cfg
            num_steps = amp_cfg["num_steps"]
            if base.max_episode_length < num_steps + 1:
                raise ValueError("诊断回合太短，无法积累完整 AMP 窗口")
            reference = MotionDataset(
                manifest_file=cfg.events.reset_from_reference.params["manifest_file"],
                robot_joint_names=list(base.scene["robot"].joint_names),
                policy_joint_names=amp_cfg["joint_names"],
                device=base.device,
                step_dt=base.step_dt,
            )
            reference_windows = reference.sample_amp_windows(512, num_steps)["states"]
            discriminator = AMPWindowDiscriminator(
                reference_windows.flatten(0, 1), num_steps=num_steps,
            )
            buffer = AMPWindowReplayBuffer(
                capacity=max(1024, args.steps * base.num_envs),
                state_dim=AMP_STATE_DIM, num_steps=num_steps, device=base.device,
            )
            history = AMPHistory(
                base.num_envs, state_dim=AMP_STATE_DIM,
                num_steps=num_steps, device=base.device,
            )
            obs, _ = env.reset()
            base.episode_length_buf[:] = (
                torch.arange(base.num_envs, device=base.device) % base.max_episode_length
            )
            resets = valid_count = 0
            largest_reset_gap = 0.0
            # Independent episode IDs verify the complete history, not only its last state.
            episode_ids = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
            history_ids = torch.full(
                (base.num_envs, num_steps), -1, dtype=torch.long, device=base.device,
            )
            for _ in range(args.steps):
                with torch.no_grad():
                    current = obs["amp"].clone()
                    assert current.shape == (base.num_envs, AMP_STATE_DIM)
                    fresh = history.counts == 0
                    history_ids[fresh] = -1
                    history_ids[fresh, -1] = episode_ids[fresh]
                    history.begin(current)
                    actions = 0.05 * torch.randn(
                        base.num_envs, base.action_manager.total_action_dim, device=base.device,
                    )
                    next_obs, _, terminated, truncated, extras = env.step(actions)
                    done = terminated | truncated
                    following = extras["amp_next_state"]
                    assert torch.equal(extras["amp_terminal_mask"], done)
                    assert torch.equal(following[~done], next_obs["amp"][~done])
                    assert bool(torch.isfinite(following).all())
                    if bool(done.any()):
                        resets += int(done.sum())
                        largest_reset_gap = max(
                            largest_reset_gap,
                            float((following[done] - next_obs["amp"][done]).abs().max()),
                        )
                    history_ids[:, :-1] = history_ids[:, 1:].clone()
                    history_ids[:, -1] = episode_ids
                    windows, valid = history.advance(following, done)
                    assert bool((history_ids[valid] == episode_ids[valid, None]).all())
                    buffer.add(windows, valid)
                    valid_count += int(valid.sum())
                    if bool(valid.any()):
                        reward = discriminator.style_reward(windows[valid])
                        assert bool(torch.isfinite(reward).all())
                        assert bool(((reward >= 0) & (reward <= 1)).all())
                    assert bool((history.counts[done] == 0).all())
                    history_ids[done] = -1
                    episode_ids[done] += 1
                    obs = next_obs

            assert resets > 0, "没有覆盖自动重置路径"
            assert valid_count > 0, "没有采集到完整五帧窗口"
            assert buffer.size == valid_count
            assert largest_reset_gap > 1e-5, "未观察到终止状态与重置状态的差别"
            before = next(discriminator.parameters()).detach().clone()
            for _ in range(4):
                metrics = discriminator.update(
                    reference_pairs=reference.sample_amp_windows(128, num_steps)["states"],
                    policy_pairs=buffer.sample(128),
                )
            assert not torch.equal(before, next(discriminator.parameters()).detach())
            print(f"[AMP] state_dim={AMP_STATE_DIM}, window={num_steps}, windows={buffer.size}")
            print(f"[AMP] automatic_resets={resets}, terminal/reset gap={largest_reset_gap:.6f}")
            print(f"[AMP] discriminator update={metrics}")
            print("[AMP] 五帧采集、真实终止状态和跨回合隔离检查通过")
        finally:
            env.close()


if __name__ == "__main__":
    main()
