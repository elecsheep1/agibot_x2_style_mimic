"""检查真实仿真 AMP 转移及自动重置边界，不更新策略。"""

import argparse
import sys

import gymnasium as gym
import torch

from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

import x2_style_mimic.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
    )

    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser)

    if args.num_envs < 2 or args.steps < 20:
        parser.error(
            "诊断至少使用 2 个环境和 20 个控制步"
        )

    sys.argv = [sys.argv[0]] + remaining

    cfg, _ = resolve_task_config(
        "X2-GetUp-v0",
        "",
    )

    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42

    # 仅本次诊断缩短回合，主动覆盖自动重置路径。
    # 不修改环境配置文件中的 10 秒回合长度。
    cfg.episode_length_s = 0.20

    if args.device is not None:
        cfg.sim.device = args.device

    with launch_simulation(cfg, args):
        from x2_style_mimic.algorithms.amp import (
            AMPDiscriminator,
            AMPReplayBuffer,
        )
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.amp import (
            AMPReferenceMotion,
        )
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_style_mimic_env_cfg import (
            POLICY_JOINT_NAMES,
        )

        env = gym.make(
            "X2-GetUp-v0",
            cfg=cfg,
        )
        base = env.unwrapped

        try:
            reference = AMPReferenceMotion(
                motion_file=(
                    cfg.events.reset_from_reference
                    .params["motion_file"]
                ),
                joint_names=POLICY_JOINT_NAMES,
                device=base.device,
                step_dt=base.step_dt,
            )

            discriminator = AMPDiscriminator(
                reference.states,
            )

            buffer = AMPReplayBuffer(
                capacity=100_000,
                device=base.device,
            )

            obs, _ = env.reset()

            # 错开超时，验证只重置部分环境时的索引处理。
            base.episode_length_buf[:] = (
                torch.arange(
                    base.num_envs,
                    device=base.device,
                )
                % base.max_episode_length
            )

            resets = 0
            largest_reset_gap = 0.0

            with torch.no_grad():
                for _ in range(args.steps):
                    current = obs["amp"].clone()

                    # 小幅随机动作，仅用于采集诊断。
                    actions = 0.05 * torch.randn(
                        base.num_envs,
                        base.action_manager.total_action_dim,
                        device=base.device,
                    )

                    (
                        following_obs,
                        _,
                        terminated,
                        truncated,
                        extras,
                    ) = env.step(actions)

                    done = terminated | truncated
                    following = extras["amp_next_state"]

                    assert torch.equal(
                        extras["amp_terminal_mask"],
                        done,
                    )

                    # 未终止环境，两种步后状态必须一致。
                    assert torch.equal(
                        following[~done],
                        following_obs["amp"][~done],
                    )

                    assert torch.isfinite(following).all()

                    if done.any().item():
                        resets += int(done.sum().item())

                        gap = (
                            following[done]
                            - following_obs["amp"][done]
                        ).abs().max().item()

                        largest_reset_gap = max(
                            largest_reset_gap,
                            gap,
                        )

                    # 已捕获真实终止状态，终止步的转移也可以保留。
                    buffer.add(
                        current,
                        following,
                        torch.ones_like(done),
                    )

                    reward = discriminator.style_reward(
                        current,
                        following,
                    )
                    assert torch.isfinite(reward).all()

                    # 下一次决策从环境返回的观测开始，
                    # 其中已终止环境对应重置后的出生状态。
                    obs = following_obs

            assert resets > 0, (
                "没有覆盖自动重置路径"
            )

            assert buffer.size == (
                args.steps * base.num_envs
            )

            assert largest_reset_gap > 1e-5, (
                "未观察到终止状态与重置状态的差别"
            )

            # 离开 no_grad 后更新判别器。
            # 这里没有创建或更新 Actor/Critic。
            before = (
                next(discriminator.parameters())
                .detach()
                .clone()
            )

            for _ in range(4):
                first, second = reference.sample(128)

                metrics = discriminator.update(
                    reference_pairs=torch.cat(
                        (first, second),
                        dim=-1,
                    ),
                    policy_pairs=buffer.sample(128),
                )

            assert not torch.equal(
                before,
                next(discriminator.parameters()).detach(),
            )

            print(
                f"[AMP] transitions={buffer.size}, "
                f"automatic_resets={resets}"
            )
            print(
                "[AMP] terminal/reset max difference="
                f"{largest_reset_gap:.6f}"
            )
            print(
                f"[AMP] discriminator update={metrics}"
            )
            print(
                "[AMP] Real-transition collection check passed"
            )

        finally:
            env.close()


if __name__ == "__main__":
    main()
