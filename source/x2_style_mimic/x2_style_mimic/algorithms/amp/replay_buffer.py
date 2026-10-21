"""AMP 策略转移回放缓冲区。"""

import torch


class AMPReplayBuffer:
    """保存有效相邻状态，达到容量后循环覆盖旧转移。"""

    def __init__(
        self,
        capacity=100_000,   # 保存状态转移容量
        state_dim=80,       # 单帧状态维度
        device="cpu",
    ):
        if capacity <= 0:
            raise ValueError("capacity 必须为正数")

        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device

        # 每条转移保存 current 和 following，共 160 维。
        self.data = torch.empty(
            capacity,
            state_dim * 2,
            device=device,
        )

        self.size = 0       # 当前保存了多少条有效转移
        self.position = 0   # 下一次从哪个槽位开始写

    @torch.no_grad()
    def add(self, current, following, valid_mask):
        """由采集端提供 valid_mask，排除跨重置等无效转移。"""
        if (current.ndim != 2 or current.shape[1] != self.state_dim):
            raise ValueError("current 形状错误")

        # current.shape[:1] 表示从形状中取第一个维度，结果仍然是一个形状序列。
        if (following.shape != current.shape or valid_mask.shape != current.shape[:1]):
            raise ValueError("following 或 valid_mask 形状错误")

        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask 必须为布尔张量")

        # 拼接两帧，并筛选有效数据
        pairs = torch.cat(
            (current, following),
            dim=-1,
        ).detach().to(self.device)

        # 这是布尔索引：只留下掩码为 True 的行。count 就是本次实际准备加入的转移数量。
        pairs = pairs[valid_mask.to(self.device)]
        count = len(pairs)

        if count == 0:
            return

        if not torch.isfinite(pairs).all().item():
            raise ValueError("策略转移包含 NaN/Inf")

        # 单次输入超过容量时，只保留最后 capacity 条。
        if count >= self.capacity:
            self.data.copy_(pairs[-self.capacity:])     # 取最后 capacity 行。将它们复制到已分配的存储空间。
            self.size = self.capacity
            self.position = 0
            return

        indices = (
            torch.arange(count, device=self.device)
            + self.position
        ) % self.capacity

        self.data[indices] = pairs  # 写入数据

        self.position = (
            self.position + count
        ) % self.capacity

        self.size = min(
            self.size + count,
            self.capacity,
        )

    def sample(self, batch_size):
        """从已存储转移中有放回地随机采样。"""
        if self.size == 0 or batch_size <= 0:
            raise ValueError("缓冲区为空或 batch_size 非正数")

        indices = torch.randint(
            self.size,
            (batch_size,),
            device=self.device,
        )

        return self.data[indices]

class AMPWindowReplayBuffer:
    """保存完整时序窗口；窗口是否跨重置由采集端判断。"""
    def __init__(
        self,
        capacity=100_000,   # 保存状态转移容量
        state_dim=80,       # 单帧状态维度
        num_steps=5,        # 窗口长度
        device="cpu",
    ):
        if capacity <= 0 or state_dim <= 0 or num_steps < 2:
            raise ValueError("容量、状态维度或窗口长度无效")

        self.capacity = capacity
        self.state_dim = state_dim
        self.num_steps = num_steps
        self.device = torch.device(device)

        # 每条数据保存完整窗口，不在缓冲区中拼接时序。
        self.data = torch.empty(
            (capacity, num_steps, state_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.size = 0       # 当前保存了多少条有效窗口
        self.position = 0   # 下一次从哪个槽位开始写

    @torch.no_grad()
    def add(self, windows, valid_mask):
        """只保存真实、完整且未跨回合的窗口。"""
        if (windows.ndim != 3 or windows.shape[1:] != (self.num_steps, self.state_dim)):
            raise ValueError("windows 必须为 [批量, 窗口长度, 状态维度]")
        if valid_mask.shape != windows.shape[:1] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask 必须为 [批量] 布尔张量")

        values = windows.detach().to(
            device=self.device,
            dtype=self.data.dtype,
        )
        values = values[valid_mask.to(self.device)]
        count = len(values)

        if count == 0:
            return
        if not bool(torch.isfinite(values).all()):
            raise ValueError("策略窗口包含 NaN/Inf")

        if count >= self.capacity:
            self.data.copy_(values[-self.capacity:])
            self.size = self.capacity
            self.position = 0
            return

        indices = (
            torch.arange(count, device=self.device)
            + self.position
        ) % self.capacity

        self.data[indices] = values
        self.position = (
            self.position + count
        ) % self.capacity
        self.size = min(
            self.size + count,
            self.capacity,
        )

    def sample(self, batch_size):
        """返回 [批量, 窗口长度, 状态维度]。"""
        if self.size == 0 or batch_size <= 0:
            raise ValueError("缓冲区为空或 batch_size 非正数")

        indices = torch.randint(
            self.size,
            (batch_size,),
            device=self.device,
        )

        return self.data[indices]

    def clear(self):
        """清空缓冲区。"""
        self.size = 0
        self.position = 0