"""想象起身环境的动力学。"""

class ImaginedGetupEnv:
    def __init__(
        self,
        world_model,
    ):
        self.world_model = world_model

    def reset(
        self,
        initial_state,
        context,
    ):
        self.state = initial_state.detach().clone()
        self.context = context.detach().clone()

        return self.state

    def step(self, action):
        # 想象环境的下一步状态。
        following = self.world_model(
            state=self.state,
            action=action,
            context=self.context,
        )

        # 不 detach following，保留跨时间步梯度。
        self.state = following

        return following