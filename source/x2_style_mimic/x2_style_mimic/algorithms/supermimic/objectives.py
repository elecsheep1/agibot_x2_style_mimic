import torch


def imagined_actor_loss(
    rewards: torch.Tensor,              # imagined轨迹每一步奖励，形状为 (H, B)。
    termination_prob: torch.Tensor,     # imagined轨迹每一步终止概率，形状为 (H, B)。
    terminal_value: torch.Tensor,       # 想象到第 H 步之后，预计还能获得多少未来收益，形状为 (B,)。
    gamma: float=0.99,
) -> torch.Tensor:
    """
    计算imagined actor 损失。最小化该值以更新 actor。

    Args:
        rewards: imagined奖励，形状为 (H, B)。
        termination_prob: imagined终止概率，形状为 (H, B)。
        terminal_value: imagined预计未来收益，形状为 (B)。
        gamma: 折扣因子。
    Returns:
        imagined actor 损失。

    本函数不 detach，保留通过奖励、存活权重和末端价值的梯度。
    """
    if rewards.ndim != 2 or rewards.shape[0] == 0:
        raise ValueError("rewards 必须为非空 [H, B]")
    if termination_prob.shape != rewards.shape:
        raise ValueError("termination_prob 与 rewards 形状必须一致")
    if terminal_value.shape != rewards.shape[1:]:
        raise ValueError("terminal_value 必须为 [B]")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma 必须在 [0, 1)")

    survival = torch.ones_like(terminal_value)          # 到当前步之前仍未终止的概率
    total_return = torch.zeros_like(terminal_value)     # 累计预计收益
    discount = 1.0                                      # 当前步的折扣系数

    for t in range(rewards.shape[0]):
        # 本次奖励使用动作执行前的存活权重。
        # 当前奖励 × 时间折扣 × 能活到这一步的概率
        total_return += discount * survival * rewards[t]
        # 更新存活概率
        survival *= (1.0 - termination_prob[t])
        # 更新时间折扣
        discount *= gamma

    # 加上末端价值
    total_return += discount * survival * terminal_value

    return -total_return.mean()


@torch.no_grad()
def lambda_returns(
    rewards: torch.Tensor,
    termination_prob: torch.Tensor,
    values: torch.Tensor,
    gamma: float=0.99,
    lam: float=0.95,
) -> torch.Tensor:
    """
    给想象轨迹中的每一步计算一个收益目标，让 critic 学习预测这些目标。lambda returns。

    Args:
        rewards: imagined奖励，形状为 (H, B)。
        termination_prob: imagined终止概率，形状为 (H, B)。
        values: critic 对状态 s_t 的价值预测。形状为 (H + 1, B)。
        gamma: 折扣因子。
        lam: lambda参数。
    Returns:
        lambda returns，形状为 (H, B)。
    """
    if rewards.ndim != 2 or rewards.shape[0] == 0:
        raise ValueError("rewards 必须为非空 [H, B]")
    if termination_prob.shape != rewards.shape:
        raise ValueError("termination_prob 与 rewards 形状必须一致")
    if values.shape != (
        rewards.shape[0] + 1,
        rewards.shape[1],
    ):
        raise ValueError("values 必须为 [H + 1, B]")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma 必须在 [0, 1)")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam 必须在 [0, 1]")

    targets = torch.empty_like(rewards)         # 保存每一步的目标，形状为 [H, B]
    following_return = values[-1]               # 想象窗口结束状态的价值，即 values[H]。
    # 当前目标 = 本步奖励 + gamma * 存活概率 * 未来收益
    for t in reversed(range(rewards.shape[0])):
        following_return = (
            rewards[t]
            + gamma
            * (1.0 - termination_prob[t])
            * ((1.0 - lam) * values[t + 1] + lam * following_return)
        )
        targets[t] = following_return

    return targets


def critic_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    计算 critic 的预测值与目标值之间的均方误差（MSE），用于训练 critic，让它对未来收益的估计更准确。
    """
    if predictions.shape != targets.shape:
        raise ValueError("critic predictions 与 targets 形状不一致")

    return (predictions - targets.detach()).square().mean()