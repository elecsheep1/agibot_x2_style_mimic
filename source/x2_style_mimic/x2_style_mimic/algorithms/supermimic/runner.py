"""真实转移采集与 SuperMimic 交替训练。"""

import math

import torch

from .batches import merge_world_batches


class SuperMimicRunner:
    """SuperMimic 训练器,用于在真实转移采集和 SuperMimic 训练之间交替。"""

    def __init__(
        self,
        env,
        trainer,
        replay,         # 经验回放缓存
        command_name,
        action_noise_std=0.1,
    ):
        if (
            not math.isfinite(action_noise_std)
            or action_noise_std < 0
        ):
            raise ValueError("action_noise_std 必须为非负有限数")
        if env.num_envs != replay.num_envs:
            raise ValueError("环境数量与 replay 不一致")

        self.env = env
        self.trainer = trainer
        self.replay = replay
        self.command_name = command_name
        self.action_noise_std = action_noise_std

        self.device = trainer.device

        # 构造策略观测时，需要使用多少帧历史状态
        self.observation_steps = trainer.proprioception.history_length
        self.amp_steps = trainer.style.discriminator.num_steps

        self.history_length = max(
            self.observation_steps,
            self.amp_steps,
        )

        self.history = None
        # 累计环境交互步数从 0 开始
        self.total_env_steps = 0

    def _read_commands(self):
        """从环境中读取当前命令。"""

        if self.command_name is None:
            if self.replay.command_dim != 0:
                raise ValueError("无外部目标任务要求 command_dim=0")
            return torch.empty(
                self.env.num_envs,
                0,
                device=self.device,
            )

        commands = self.env.command_manager.get_command(self.command_name)

        expected_shape = (
            self.env.num_envs,
            self.replay.command_dim,
        )

        if commands.shape != expected_shape:
            raise ValueError("CommandManager 输出与replay.command_dim 不一致")
        if not bool(torch.isfinite(commands).all()):
            raise ValueError("指令包含 NaN/Inf")

        # 这里创建独立副本，因为环境执行一步时，CommandManager 可能更新指令。保存到 replay 的应当是本步执行动作时的目标。
        return (
            commands.detach()
            .to(self.device)
            .clone()
        )

    @torch.no_grad()
    def reset(self):
        self.env.reset()

        if self.replay.total_steps > 0:
            # 标记重置
            self.replay.mark_reset(
                # 生成所有环境的编号
                torch.arange(
                    self.env.num_envs,
                    device=self.replay.device,
                )
            )

        # 读取重置后的状态
        current = (
            self.env.read_world_state()
            .to(self.device)
        )

        # 用初始状态填满历史窗口
        self.history = (
            current[:, None]
            .expand(
                -1,
                self.history_length,
                -1,
            )
            .clone()
        )

        # 尽早检查环境指令配置是否正确。
        self._read_commands()

    @torch.no_grad()
    def collect(self, num_steps):
        if (
            not isinstance(num_steps, int)
            or num_steps < 1
        ):
            raise ValueError("num_steps 必须为正整数")

        if self.history is None:
            self.reset()

        terminated_count = 0
        truncated_count = 0
        action_square_sum = 0.0

        for _ in range(num_steps):
            # 保存执行本步动作前的目标。
            commands = self._read_commands()

            # 每个真实控制步只统计当前帧。
            # 不重复统计历史窗口中的旧帧。
            self.trainer.proprioception.update_normalization(
                self.history[:, -1]
            )

            observation = (
                self.trainer.proprioception(
                    self.history[
                        :, -self.observation_steps:
                    ]
                )
            )

            action = self.trainer.actor(
                observation,
                commands,
            )

            # 世界模型记录的是加入噪声后实际发送的动作。
            if self.action_noise_std > 0:
                noise = (
                    torch.randn_like(action)
                    * self.action_noise_std
                )
                action = action + noise

            if action.shape != (
                self.env.num_envs,
                self.replay.action_dim,
            ):
                raise ValueError("Actor 输出动作形状不正确")
            if not bool(torch.isfinite(action).all()):
                raise RuntimeError("Actor 输出包含 NaN/Inf")

            action = action.clamp(
                min=self.trainer.actor.action_lower,
                max=self.trainer.actor.action_upper,
            )

            _, _ , _, _ , extras = self.env.step(action)

            # X2SuperMimicEnv 已捕获真实终止前状态。
            """
            环境结束时有两个不同的状态，都需要用到。
            - 旧回合的最后状态：用来保存真实转移、训练世界模型。
            - 自动重置后的新初始状态：用来让 Actor 开始下一回合。
            """
            # 保存真实的动作转移
            captured = extras["world_model"]

            # 创建一个浅拷贝字典，加上本步指令，存入回放缓存。
            """
            transition：所有并行环境在本控制步产生的转移数据。
            state：动作执行前的状态。
            action：本步执行的动作。
            next_state：动作执行后的状态，包含真实终止前的物理状态。
            reward：环境任务奖励，不含算法侧额外计算的 AMP 奖励。
            terminated：是否触发任务终止条件。
            truncated：是否因时间限制等原因截断。
            commands：本步目标指令。
            schema_version：转移数据格式版本，当前为 4。
            task_memory：动作执行前的任务记忆。
            next_task_memory：动作执行后的任务记忆。
            amp_reward_gate：AMP 风格奖励的缩放系数。
            amp_policy_sample_mask：是否允许作为 AMP 判别器的策略样本。
            对于本步结束的环境，下一状态及对应辅助数据保留重置前的旧回合数据。
            """
            transition = dict(captured)
            transition["commands"] = commands
            self.replay.add(transition)

            # 判断哪些环境结束了
            terminated = captured["terminated"].to(self.device)
            truncated = captured["truncated"].to(self.device)
            done = terminated | truncated

            # 所有环境先正常更新历史
            following = captured["next_state"].to(self.device)
            self.history = torch.cat(
                (
                    self.history[:, 1:],
                    following[:, None],
                ),
                dim=1,
            )

            # 对已经结束的环境，重新填充历史
            if bool(done.any()):
                # 读取重置后的状态
                reset_state = (
                    self.env.read_world_state()
                    .to(self.device)
                )
                # 用重置后的状态填充历史窗口
                self.history[done] = (
                    reset_state[done, None]
                    .expand(
                        -1,
                        self.history_length,
                        -1,
                    )
                )

            terminated_count += int(terminated.sum())
            truncated_count += int(truncated.sum())
            action_square_sum += float(action.square().mean())

        self.total_env_steps += num_steps

        return {
            "collection/env_steps": num_steps,
            "collection/transitions": (
                num_steps * self.env.num_envs
            ),
            "collection/terminated": (
                terminated_count
            ),
            "collection/truncated": (
                truncated_count
            ),
            "collection/action_rms": math.sqrt(
                action_square_sum / num_steps
            ),
            "collection/total_transitions": (
                self.total_env_steps
                * self.env.num_envs
            ),
        }

    @torch.no_grad()
    def sample_policy_windows(
        self,
        batch_size,
    ):
        # 从回放缓存中采样策略窗口
        batch = self.replay.sample_windows(
            batch_size=batch_size,
            horizon=self.amp_steps - 1      # n 帧状态有n-1个转移
        )

        states = batch["states"].to(self.device)

        features = self.trainer.amp_features(
            states.flatten(0, 1)
        )

        return features.reshape(
            batch_size,
            self.amp_steps,
            -1,
        )

    @torch.no_grad()
    def sample_world_batch(
        self,
        batch_size,
        horizon,
        offline_world_replay=None,      # 离线数据缓存
        offline_world_fraction=0.0,     # 采样批次中，离线数据所占的比例。
    ):
        """仅供世界模型监督训练使用的混合真实转移。"""
        if (
            not isinstance(batch_size, int)
            or batch_size < 1
            or not isinstance(horizon, int)
            or horizon < 1
        ):
            raise ValueError("batch_size 和 horizon 必须为正整数")
        if (
            not math.isfinite(offline_world_fraction)
            or not 0.0 <= offline_world_fraction <= 1.0
        ):
            raise ValueError("offline_world_fraction 必须在 [0, 1] 内")
        if (
            offline_world_fraction > 0
            and offline_world_replay is None
        ):
            raise ValueError("离线数据比例大于零，但没有提供离线回放数据")

        offline_count = (
            int(batch_size * offline_world_fraction)
            if offline_world_replay is not None
            else 0
        )
        online_count = batch_size - offline_count

        batches = []

        if online_count > 0:
            batches.append(
                self.replay.sample_windows(
                    batch_size=online_count,
                    horizon=horizon,
                )
            )
        if offline_count > 0:
            batches.append(
                offline_world_replay.sample_windows(
                    batch_size=offline_count,
                    horizon=horizon,
                )
            )

        return merge_world_batches(
            batches=batches,
            batch_size=batch_size,
            horizon=horizon,
            state_dim=self.replay.state_dim,
            action_dim=self.replay.action_dim,
            context_dim=self.replay.contexts.shape[-1],
            extra_fields=getattr(
                self.replay,
                "extra_fields",
                {},
            ),
            device=self.device,
        )

    def train_iteration(
        self,
        reference,
        collection_steps=32,
        batch_size=256,
        world_horizon=8,
        world_updates=10,
        discriminator_updates=4,
        offline_world_replay=None,
        offline_world_fraction=0.0,
    ):
        for value in (
            collection_steps,
            batch_size,
            world_horizon,
            world_updates,
            discriminator_updates,
        ):
            if (
                not isinstance(value, int) or value < 1
            ):
                raise ValueError("采集、采样与更新次数必须为正整数")
        if (
            not math.isfinite(offline_world_fraction)
            or not 0.0 <= offline_world_fraction <= 1.0
        ):
            raise ValueError("offline_world_fraction 必须在 [0, 1] 内")

        metrics = self.collect(collection_steps)

        for _ in range(world_updates):
            batch = self.sample_world_batch(
                batch_size=batch_size,
                horizon=world_horizon,
                offline_world_replay=offline_world_replay,
                offline_world_fraction=offline_world_fraction,
            )

            metrics.update(
                self.trainer.update_world_model(batch)
            )

        offline_count = (
            int(batch_size * offline_world_fraction)
            if offline_world_replay is not None
            else 0
        )

        metrics["world/offline_fraction"] = (
            offline_count / batch_size
        )

        for _ in range(discriminator_updates):
            policy_windows = self.sample_policy_windows(batch_size)

            # 这里是从参考动作数据中抽取一批 AMP 历史窗口，并取出返回结果中的 "states" 数据。
            reference_windows = reference.sample_amp_windows(
                batch_size=batch_size,
                num_steps=self.amp_steps,
            )["states"]

            metrics.update(
                self.trainer.update_discriminator(
                    reference_windows,
                    policy_windows,
                )
            )

        policy_batch = (
            self.replay.sample_policy_histories(
                batch_size=batch_size,
                history_length=self.history_length,
            )
        )

        trajectory, actor_metrics = (
            self.trainer.update_actor(
                history=policy_batch["history"],
                context=policy_batch["context"],
                goals=policy_batch["commands"],
            )
        )

        # 把 actor_metrics 中的键值对合并到 metrics 中。
        metrics.update(actor_metrics)

        metrics.update(
            self.trainer.update_critic(
                trajectory,
                context=policy_batch["context"],
            )
        )

        return metrics
