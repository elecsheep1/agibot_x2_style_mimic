"""世界模型、AMP 判别器、actor 和 critic 的交替更新。"""

from dataclasses import dataclass
import math

import torch

from .batches import detach_world_batch
from .optimizers import build_optimizers

from .gradients import frozen_parameters
from .objectives import (
    imagined_actor_loss,
    lambda_returns,
    critic_loss,
)
from .rollout import (
    ImaginedTrajectory,
    imagine,
)
from .world_model import (
    rollout_logged_actions,
    dynamics_loss,
)


@dataclass(frozen=True)
class TrainerCfg:
    horizon: int = 32
    gamma: float = 0.99
    lam: float = 0.95

    world_learning_rate: float = 2e-4
    actor_learning_rate: float = 2e-4
    critic_learning_rate: float = 2e-4

    nominal_error_root_pose_pos: float = 0.01
    nominal_error_root_pose_ori: float = 0.02
    nominal_error_joint_pos: float = 0.1
    nominal_error_joint_vel: float = 3.14

    # 梯度范数裁剪阈值
    gradient_norm: float = 1.0
    actor_gradient_value: float = 0.1

    critic_epochs: int = 5
    critic_minibatches: int = 4


class SuperMimicTrainer:
    """训练世界模型、AMP 判别器、actor 和 critic。"""

    OPTIMIZER_MODE = "muon_adam_v1"

    def __init__(
        self,
        actor,
        critic,
        world_model,
        proprioception,
        amp_features,
        style,
        task,
        cfg=None,
    ):
        self.actor = actor
        self.critic = critic
        self.world_model = world_model
        self.proprioception = proprioception
        self.amp_features = amp_features
        self.style = style
        self.task = task
        self.cfg = (
            TrainerCfg()    # 创建默认配置
            if cfg is None
            else cfg
        )
        cfg = self.cfg

        if not all(
            math.isfinite(value)
            for value in vars(cfg).values()
        ):
            raise ValueError("Trainer 配置必须为有限数值")
        for value in (
            cfg.horizon,
            cfg.critic_epochs,
            cfg.critic_minibatches,
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{value} 必须为大于 0 的整数")
        if (
            not 0 <= cfg.gamma < 1
            or not 0 <= cfg.lam <= 1
        ):
            raise ValueError("折扣因子 gamma 必须在 [0, 1)，lambda 必须在 [0, 1]")
        if min(
            cfg.nominal_error_root_pose_pos,
            cfg.nominal_error_root_pose_ori,
            cfg.nominal_error_joint_pos,
            cfg.nominal_error_joint_vel,
            cfg.world_learning_rate,
            cfg.actor_learning_rate,
            cfg.critic_learning_rate,
            cfg.gradient_norm,
            cfg.actor_gradient_value,
        ) <= 0:
            raise ValueError("学习率、梯度裁剪阈值和世界模型名义误差必须为正")

        max_step_reward = float(
            self.task.max_step_reward
        )

        if (
            not math.isfinite(max_step_reward)
            or max_step_reward < 0.0
        ):
            raise ValueError("任务单步奖励上限必须为有限非负数")

        # 计算累计折扣回报的理论上限。
        self.value_max = (
            max_step_reward
            / (1.0 - cfg.gamma)
        )

        # 读取世界模型第一个参数所在的设备，并保存到 self.device
        # next() 的意思是：从迭代器中取出下一个元素。
        self.device = next(
            world_model.parameters()
        ).device

        self.world_optimizers = build_optimizers(
            world_model,
            lr=cfg.world_learning_rate,
        )
        self.actor_optimizers = build_optimizers(
            actor,
            lr=cfg.actor_learning_rate,
            betas=(0.9, 0.95),
        )
        self.critic_optimizers = build_optimizers(
            critic,
            lr=cfg.critic_learning_rate,
        )

    def update_world_model(self, batch):
        """
        batch 来自真实转移的连续窗口。

        states:  [B, K+1, Ds]
        actions: [B, K, A]
        context: [B, Dc]
        """

        data = detach_world_batch(
            batch,
            self.device,
        )

        # 若输入归一化统计量尚未初始化，则用当前数据计算并保存各特征的均值和标准差。
        if not bool(self.world_model.stats_ready):
            self.world_model.fit_input_stats(data)

        # 清除世界模型上一步的梯度，设为 None，避免本次反向传播时累加旧梯度。
        for optimizer in self.world_optimizers:
            optimizer.zero_grad(
                set_to_none=True,
            )

        rollout = rollout_logged_actions(
            model=self.world_model,
            initial_state=data["states"][:, 0],
            actions=data["actions"],
            context=data["context"],
        )

        loss, terms = dynamics_loss(
            predicted_states=rollout,
            target_states=data["states"],       # 训练的是世界模型：让它预测的状态尽量接近环境中实际发生的状态。
            state_spec=self.world_model.spec,

            nominal_error_root_pose_pos=(self.cfg.nominal_error_root_pose_pos),
            nominal_error_root_pose_ori=(self.cfg.nominal_error_root_pose_ori),
            nominal_error_joint_pos=(self.cfg.nominal_error_joint_pos),
            nominal_error_joint_vel=(self.cfg.nominal_error_joint_vel),
        )

        if not bool(torch.isfinite(loss)):
            raise RuntimeError("世界模型损失包含 NaN/Inf")

        loss.backward()

        # 梯度裁剪
        norm = torch.nn.utils.clip_grad_norm_(
            self.world_model.parameters(),
            self.cfg.gradient_norm,
            error_if_nonfinite=True,
        )

        for optimizer in self.world_optimizers:
            optimizer.step()

        for optimizer in self.world_optimizers:
            optimizer.zero_grad(
                set_to_none=True,
            )

        return {
            "world_loss": float(loss.detach()),
            "world_grad_norm": float(norm),
            **{
                f"world/{name}": float(value.detach())
                for name, value in terms.items()
            },
        }


    def update_discriminator(
        self,
        reference_windows,
        policy_windows,
    ):
        result = self.style.update_discriminator(
            reference_windows.detach().to(self.device),
            policy_windows.detach().to(self.device),
        )

        self.style.discriminator.optimizer.zero_grad(
            set_to_none=True,
        )

        return {
            f"amp/{name}": float(value)
            for name, value in result.items()
        }

    def update_actor(
        self,
        history,
        context,
        goals,
    ):
        if not bool(self.world_model.stats_ready):
            raise RuntimeError("先训练世界模型，或加载包含输入统计量的 checkpoint")


        history = history.detach().to(self.device)
        context = context.detach().to(self.device)
        goals = goals.detach().to(self.device)
        for optimizer in self.world_optimizers:
            optimizer.zero_grad(
                set_to_none=True,
            )
        for optimizer in self.actor_optimizers:
            optimizer.zero_grad(
                set_to_none=True,
            )
        for optimizer in self.critic_optimizers:
            optimizer.zero_grad(
                set_to_none=True,
            )
        self.style.discriminator.optimizer.zero_grad(
            set_to_none=True,
        )

        with frozen_parameters(
            self.world_model,
            self.critic,
            self.style,
            self.proprioception,
            self.amp_features,
            self.task,
        ):
            trajectory = imagine(
                actor=self.actor,
                world_model=self.world_model,
                proprioception=self.proprioception,
                amp_features=self.amp_features,
                style=self.style,
                task=self.task,
                history=history,
                context=context,
                goals=goals,
                horizon=self.cfg.horizon,
            )

            # 最终状态 s_H 和该时刻的目标 g_H。
            terminal_value = self.critic(
                trajectory.states[-1],
                context,
                trajectory.goals[-1],
            ).clamp(
                max=self.value_max,
            )

            loss = imagined_actor_loss(
                rewards=trajectory.rewards,
                termination_prob=trajectory.termination_probabilities,
                terminal_value=terminal_value,
                gamma=self.cfg.gamma,
            )

            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Actor 损失包含 NaN/Inf")

            # 梯度经过奖励、判别器和世界模型传回 actor。
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.cfg.gradient_norm,
            error_if_nonfinite=True,
        )

        # 全局范数裁剪后，再限制单个梯度元素。
        for param in self.actor.parameters():
            if param.grad is not None:
                param.grad.clamp_(
                    -self.cfg.actor_gradient_value,
                    self.cfg.actor_gradient_value,
                )

        for optimizer in self.actor_optimizers:
            optimizer.step()

        metrics = {
            "actor_loss": float(loss.detach()),
            "actor_gradient_norm": float(norm),
            "imagined_reward": float(trajectory.rewards.mean().detach()),
            "imagined_termination": float(
                trajectory
                .termination_probabilities
                .mean()
                .detach()
            ),
            "terminal_value": float(terminal_value.mean().detach()),
            **{
                f"reward/{name}": float(value.mean().detach())
                for name, value in trajectory.reward_terms.items()
            }
        }

        # Critic 使用这些数值训练，无需保留 actor 计算图。
        detached = ImaginedTrajectory(
            states=trajectory.states.detach(),
            actions=trajectory.actions.detach(),
            rewards=trajectory.rewards.detach(),
            termination_probabilities=trajectory.termination_probabilities.detach(),
            goals=trajectory.goals.detach(),
            reward_terms={
                name: value.detach()
                for name, value
                in trajectory.reward_terms.items()
            },
        )

        return detached, metrics

    def update_critic(
        self,
        trajectory,
        context,
    ):
        state = trajectory.states.detach().to(self.device)
        goals = trajectory.goals.detach().to(self.device)
        rewards = trajectory.rewards.detach().to(self.device)
        termination_prob = trajectory.termination_probabilities.detach().to(self.device)
        context = context.detach().to(self.device)

        horizon, batch_size = rewards.shape

        # 同一条想象轨迹使用固定物理 context。
        contexts = context[None].expand(
            horizon + 1,
            -1,
            -1,
        )

        # 所有训练目标在 critic 更新前一次生成。
        with torch.no_grad():
            values = self.critic(
                # 先合并时间维和批次维，方便一次性计算所有状态的价值。
                state=state.flatten(0, 1),
                context=contexts.flatten(0, 1),
                goals=goals.flatten(0, 1),
            ).reshape(
                horizon + 1,
                batch_size,
            )

            targets = lambda_returns(
                rewards=rewards,
                termination_prob=termination_prob,
                values=values,
                gamma=self.cfg.gamma,
                lam=self.cfg.lam,
            ).clamp(
                max=self.value_max,
            ).flatten()

        # s_H 只参与 bootstrap。
        # critic 拟合 s_0 到 s_{H-1} 的价值。所以这里去掉最后一帧。
        flat_states = state[:-1].flatten(0, 1)
        flat_contexts = contexts[:-1].flatten(0, 1)
        flat_goals = goals[:-1].flatten(0, 1)

        sample_count = horizon * batch_size

        # 取“配置的小批次数”和“样本总数”中的较小值，避免小批次数超过样本数，无法保证每批至少一个样本。
        minibatches = min(
            self.cfg.critic_minibatches,
            sample_count,
        )

        losses = []
        norms = []

        # 同一批想象轨迹只打乱一次。
        order = torch.randperm(
            sample_count,
            device=self.device,
        )

        minibatch_indexes = order.chunk(
            minibatches,
        )

        # 多轮拟合复用相同的小批次划分。
        for _ in range(self.cfg.critic_epochs):
            for indexes in minibatch_indexes:
                # Critic（C 网络）输出的状态价值预测：
                predictions = self.critic(
                    flat_states[indexes],
                    flat_contexts[indexes],
                    flat_goals[indexes],
                )

                loss = critic_loss(
                    predictions=predictions,
                    targets=targets[indexes],
                )

                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Critic 损失包含 NaN/Inf")

                for optimizer in self.critic_optimizers:
                    optimizer.zero_grad(
                        set_to_none=True,
                    )

                loss.backward()

                norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.cfg.gradient_norm,
                    error_if_nonfinite=True,
                )

                for optimizer in self.critic_optimizers:
                    optimizer.step()

                # 存的是每个小批次的损失和梯度范数
                losses.append(
                    float(loss.detach())
                )
                norms.append(
                    float(norm)
                )

        for optimizer in self.critic_optimizers:
            optimizer.zero_grad(
                set_to_none=True,
            )

        return {
            # losses、norms 都是浮点数列表：
            "critic_loss": (
                sum(losses) / len(losses)
            ),
            "critic_gradient_norm": (
                sum(norms) / len(norms)
            ),
            "critic_target_mean": float(
                targets.mean()
            ),
            "critic_updates": len(losses),
        }

