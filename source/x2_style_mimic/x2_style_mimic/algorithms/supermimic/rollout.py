"""任务奖励与 AMP 风格奖励共同驱动的可微想象轨迹。"""

from dataclasses import dataclass

import torch

from .imagined_env import ImaginedGetupEnv


@dataclass
class ImaginedTrajectory:
    """保存各时间步的状态、动作、奖励和软终止概率，后续供 actor 和 critic 更新使用。"""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    termination_probabilities: torch.Tensor
    goals: torch.Tensor
    reward_terms: dict[str, torch.Tensor]


def imagine(
    actor,
    world_model,
    proprioception,
    amp_features,
    style,
    task,
    history,
    context,
    goals,
    task_memory,
    assistance_cfg,
    horizon=32,
):
    """
    在世界模型内展开想象轨迹，保留完整计算图。

    history:
        [B, L, Ds]，来自同一回合的连续真实状态。

    context:
        [B, Dc]，对应环境的物理参数。

    goals:
        [B, G] 表示整个 rollout 使用固定目标。
        [B, H+1, G] 表示提供逐步变化的目标。

    返回张量采用时间维在前的排列。
    """

    if horizon < 1:
        raise ValueError("horizon 必须为正整数")
    if history.ndim != 3:
        raise ValueError("history 必须为 [B, L, Ds]")

    # 解包 history.shape 中的三个维度大小，分别赋给三个变量
    batch_size, history_length, state_dim = history.shape

    observation_steps = (
        proprioception.history_length
    )
    amp_steps = style.discriminator.num_steps
    required_history = max(
        observation_steps,
        amp_steps,
    )
    if history_length < required_history:
        raise ValueError("历史长度不足以构成本体观测和 AMP 窗口")
    if (
        context.ndim != 2
        or context.shape[0] != batch_size
    ):
        raise ValueError("context 必须为 [B, Dc]")

    if(
        goals.ndim == 2
        and goals.shape[0] == batch_size
    ):
        # 固定目标展开为 H+1 个时间步。
        # goals[:, None] 会在第 0 维后面插入一个新维度，使它变成三维。
        goal_sequence = goals[:, None].expand(
            -1,
            horizon + 1,
            -1,
        )
    elif(
        goals.ndim == 3
        and goals.shape[:2]
        == (batch_size, horizon + 1)
    ):
        goal_sequence = goals
    else:
        raise ValueError("goals 必须为 [B, G] 或 [B, H+1, G]")

    # 使用真实历史构造初始 AMP 窗口。
    amp_window = amp_features(
        # 取最后amp_steps帧历史状态，reshape成 [B * amp_steps, Ds]，再送入 amp_features
        history[:, -amp_steps:].reshape(
            -1,
            state_dim,
        )
    ).reshape(
        batch_size,
        amp_steps,
        -1,
    )

    # 本体历史和 AMP 历史可以具有不同长度。
    state_history = history[
        :, -observation_steps:
    ]

    current = state_history[:, -1]  # 当前状态为历史的最后一帧

    imagined_env = ImaginedGetupEnv(
        world_model=world_model,
        assistance_cfg=assistance_cfg,
    )

    current = imagined_env.reset(
        initial_state=current,
        context=context,
        task_memory=task_memory,
    )

    states = [current]
    actions = []
    rewards = []
    probabilities = []
    reward_terms = {}

    for step in range(horizon):
        observation = proprioception(
            state_history
        )

        current_goal = goal_sequence[:, step]

        action = actor(
            observation,
            current_goal,
        )

        following = imagined_env.step(action)

        # 预测下一帧的可微 AMP 特征。
        next_amp_frame = amp_features(
            following
        )
        # 删掉最旧的一帧，把最新的一帧加到末尾。
        amp_window = torch.cat(
            (
                amp_window[:, 1:],
                next_amp_frame[:, None],
            ),
            dim=1,
        )

        # 此处保留判别器对窗口输入的梯度。
        style_reward = style(
            amp_window
        )

        # 用本步目标评价动作执行后的状态 s_{t+1}
        reward, terms = task.reward(
            before=current,
            action=action,
            following=following,
            goals=current_goal,
            style_reward=style_reward,
        )

        probability = (
            task.termination_probability(
                following
            )
        )

        states.append(following)
        actions.append(action)
        rewards.append(reward)
        probabilities.append(probability)

        # 把每一步的分项奖励，按名称收集到列表里。
        for name, value in terms.items():
            # 如果 name 不存在，就在字典中添加：reward_terms[name] = []
            reward_terms.setdefault(
                name,
                [],
            ).append(value)         # .append(value)：把本步奖励添加到列表。

        # 将预测状态追加到本体历史。
        state_history = torch.cat(
            (
                state_history[:, 1:],
                following[:, None],
            ),
            dim=1,
        )

        current = following

    return ImaginedTrajectory(
        # imagined_actor_loss() 和 lambda_returns() 都接收 [H, B] 的奖励，时间维在前。
        # 把列表堆叠成张量，时间维在前
        states=torch.stack(
            states,
            dim=0,
        ),
        actions=torch.stack(
            actions,
            dim=0,
        ),
        rewards=torch.stack(
            rewards,
            dim=0,
        ),
        termination_probabilities=torch.stack(
            probabilities,
            dim=0,
        ),
        # 交换 goal_sequence 的第 0 维和第 1 维，然后把结果作为 goals 字段传给 ImaginedTrajectory。
        goals=goal_sequence.transpose(0, 1),
        # 把每个分项奖励的列表堆叠成张量，时间维在前
        reward_terms={
            name: torch.stack(
                values,
                dim=0,
            )
            for name, values in reward_terms.items()
        },
    )


