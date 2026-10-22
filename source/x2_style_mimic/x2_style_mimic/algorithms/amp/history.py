"""按环境维护 AMP 时序，禁止窗口跨回合。"""

import torch

class AMPHistory:
    def __init__(
        self,
        num_envs,
        state_dim=80,
        num_steps=5,
        device="cpu",
    ):
        if num_envs <= 0 or state_dim <= 0 or num_steps < 2:
            raise ValueError("环境数量、状态维度或窗口长度无效")

        self.num_envs = num_envs
        self.state_dim = state_dim
        self.num_steps = num_steps
        self.device = torch.device(device)

        # 记录每个环境的时序窗口，窗口内的帧按时间顺序排列。刚出生的环境窗口全为 0。
        self.history = torch.zeros(
            (num_envs, num_steps, state_dim),
            device=self.device,
        )
        # 记录每个环境已捕获的帧数。0 表示刚出生，1 表示已捕获第一帧，依此类推。
        self.counts = torch.zeros(
            num_envs,
            dtype=torch.long,
            device=self.device,
        )

    def _check_state(self, state):
        if state.shape != (self.num_envs, self.state_dim):
            raise ValueError("状态必须为 [环境数, 特征维度]")
        if not bool(torch.isfinite(state).all()):
            raise ValueError("状态包含 NaN/Inf")

    @torch.no_grad()
    def begin(self, current):
        """动作前调用：仅为刚出生的环境写入第一帧。"""
        self._check_state(current)
        current = current.to(self.device)

        # 仅为刚出生的环境写入第一帧。
        fresh = self.counts == 0
        self.history[fresh] = 0.0
        self.history[fresh, -1] = current[fresh]
        self.counts[fresh] = 1

    @torch.no_grad()
    def advance(self, following, dones):
        """追加真实步后状态，返回窗口后清空已结束环境。"""
        self._check_state(following)

        if (dones.shape != (self.num_envs,) or dones.dtype != torch.bool):
            raise ValueError("dones 必须为 [环境数] 布尔张量")
        if bool((self.counts == 0).any()):
            raise RuntimeError("advance 不能在刚出生的环境上调用,必须先调用 begin")

        following = following.to(self.device)
        dones = dones.to(self.device)

        self.history[:, :-1] = self.history[:, 1:].clone()      # 左移一帧
        self.history[:, -1] = following                         # 追加最新帧
        self.counts.add_(1).clamp_(max=self.num_steps)          # 增加计数

        # 未积累满五个真实状态的窗口不能用于 AMP。
        valid_mask = self.counts >= self.num_steps
        windows = self.history.clone()

        # following 必须是终止前状态，而不是重置后的出生状态。
        # 完整终止窗口可以保留，下一回合必须重新积累。
        self.history[dones] = 0.0
        self.counts[dones] = 0

        return windows, valid_mask

    def clear(self):
        """清空所有环境的时序窗口。"""
        self.history.zero_()
        self.counts.zero_()
