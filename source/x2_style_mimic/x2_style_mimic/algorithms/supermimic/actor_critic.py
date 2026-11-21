import torch
from torch import nn


class GoalConditionedActor(nn.Module):
    """目标条件化的 actor 模型,用于生成动作。"""

    def __init__(
        self,
        observation_dim,
        goal_dim,
        num_actions,
    ):
        super().__init__()

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
        return self.network(inputs)


class GoalConditionedCritic(nn.Module):
    """目标条件化的 critic 模型,用于评估动作价值。"""

    def __init__(
        self,
        state_dim,
        context_dim,
        goal_dim,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                state_dim + context_dim + goal_dim,
                256,
            ),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, state, context, goals):
        inputs = torch.cat(
            (state, context, goals),
            dim=-1,
        )
        return self.network(inputs).squeeze(-1)