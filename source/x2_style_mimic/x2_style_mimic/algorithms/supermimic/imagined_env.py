"""想象起身环境的动力学与辅助状态。"""

import torch

from ...tasks.manager_based.x2_style_mimic.mdp.supermimic.assistance import (
    height_assistance,
)


class ImaginedGetupEnv:
    def __init__(
        self,
        world_model,
        assistance_cfg,
    ):
        self.world_model = world_model
        self.assistance_cfg = assistance_cfg

    def reset(
        self,
        initial_state,
        context,
        task_memory,
    ):
        batch_size = initial_state.shape[0]

        force_cap = task_memory[
            "assist_force_cap_n"
        ]
        withdrawn = task_memory[
            "assist_withdrawn"
        ]

        if (
            force_cap.shape != (batch_size,)
            or withdrawn.shape != (batch_size,)
        ):
            raise ValueError("想象起点辅助状态必须为 [B]")
        if withdrawn.dtype != torch.bool:
            raise ValueError("想象起点撤力标记必须为布尔型")
        if (
            not bool(torch.isfinite(force_cap).all())
            or bool((force_cap < 0).any())
        ):
            raise ValueError("辅助力上限必须为非负有限值")

        self.state = initial_state.detach().clone()
        self.context = context.detach().clone()

        self.force_cap = force_cap.detach().clone()
        self.withdrawn = withdrawn.detach().clone()

        return self.state

    def step(self, action):
        height = self.state[:, 2]

        force, next_withdrawn = (
            height_assistance(
                height=height,
                force_cap=self.force_cap,
                withdrawn=self.withdrawn,
                cfg=self.assistance_cfg,
            )
        )

        zeros = torch.zeros_like(force)

        # 世界坐标、作用于 torso 质心。
        """
        [Fx, Fy, Fz, Tx, Ty, Tz]
        [ 0,  0, force, 0, 0, 0]
        """
        wrench = torch.stack(
            (
                zeros,
                zeros,
                force,
                zeros,
                zeros,
                zeros,
            ),
            dim=-1,
        )

        # 想象环境的下一步状态。
        following = self.world_model(
            state=self.state,
            action=action,
            context=self.context,
            assist_wrench_w=wrench,
        )

        # 不 detach following，保留跨时间步梯度。
        self.state = following
        self.withdrawn = next_withdrawn
        self.applied_wrench_w = wrench

        return following