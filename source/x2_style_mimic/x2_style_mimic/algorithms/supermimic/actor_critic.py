import torch
from torch import nn


class GoalConditionedActor(nn.Module):
    """目标条件化的 actor 模型,用于生成动作。输出范围由动作标定和关节限位共同确定。"""

    def __init__(
        self,
        observation_dim,
        goal_dim,
        num_actions,
        *,
        action_lower,
        action_upper,
    ):
        super().__init__()

        lower = torch.as_tensor(
            action_lower,
            dtype=torch.float32,
        ).detach().clone()

        upper = torch.as_tensor(
            action_upper,
            dtype=torch.float32,
        ).detach().clone()

        if (
            lower.shape != (num_actions,)
            or upper.shape != (num_actions,)
            or not bool(torch.isfinite(lower).all())
            or not bool(torch.isfinite(upper).all())
            or bool((lower >= 0).any())
            or bool((upper <= 0).any())
        ):
            raise ValueError("动作上下界必须为有限的 [动作数量]，且严格包含零")

        self.register_buffer("action_lower", lower)
        self.register_buffer("action_upper", upper)

        self.network = nn.Sequential(
            nn.Linear(observation_dim + goal_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_actions),
        )

        # 初始化网络最后一层的权重和偏置，让网络初始输出接近 0。
        nn.init.normal_(
            self.network[-1].weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observation, goals):
        inputs = torch.cat(
            (observation, goals),
            dim=-1,
        )

        raw = self.network(inputs)

        # 分别使用正、负方向的有效动作范围。
        scale = torch.where(
            raw >= 0.0,
            self.action_upper,
            -self.action_lower,
        )

        return scale * torch.tanh(raw / scale)


class GoalConditionedCritic(nn.Module):
    """物理状态、目标与任务条件对应的价值网络。"""

    def __init__(
        self,
        state_dim,
        context_dim,
        goal_dim,
        task_context_dim=0, # 任务条件化的上下文维度,默认为 0。
    ):
        super().__init__()

        if task_context_dim < 0:
            raise ValueError("任务条件维度不能为负")

        self.task_context_dim = int(task_context_dim)

        input_dim = (
            state_dim
            + context_dim
            + goal_dim
            + self.task_context_dim
        )


        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        state,
        context,
        goals,
        task_context=None,
        ):
        batch_size = state.shape[0]

        if task_context is None:
            if self.task_context_dim != 0:
                raise ValueError("Critic 缺少任务条件")
            task_context = state.new_empty(
                batch_size,
                0,
            )

        if task_context.shape != (
            batch_size,
            self.task_context_dim,
        ):
            raise ValueError("Critic 任务条件形状错误，要求 [B, task_context_dim]")

        inputs = torch.cat(
            (
                state,
                context,
                goals,
                task_context,
            ),
            dim=-1,
        )
        return self.network(inputs).squeeze(-1)