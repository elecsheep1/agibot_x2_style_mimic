import torch
from torch import nn

from .state import StateSpec, pack_state, unpack_state
from ...utils.rotation import (
    normalize_quaternion as unit_quat,
    quaternion_multiply as quat_mul,
    rotation_vector_to_quaternion as rotation_increment,
    quaternion_to_rotation_6d as quat_to_rot6d,
    quaternion_angle_error as rotation_error,
)


class TargetLimiter(nn.Module):
    """复用当前 LimitedJointPositionAction 的目标计算规则。

    当前版本要求控制参数在不同环境间一致。
    """
    def __init__(
        self,
        offset,
        scale,
        lower,
        upper,
        max_delta,
    ):
        super().__init__()

        num_actions = torch.as_tensor(offset).numel()

        values = {
            "offset": offset,
            "scale": scale,
            "lower": lower,
            "upper": upper,
            "max_delta": max_delta,
        }

        for name, value in values.items():
            value = torch.as_tensor(value, dtype=torch.float32)
            if value.ndim == 0:
                value = value.expand(num_actions)
            if (
                value.shape != (num_actions,) or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{name} 必须为标量或长度为 {num_actions} 的有限向量")

            self.register_buffer(name, value.clone())

        if bool((self.lower > self.upper).any()):
            raise ValueError("关节目标上下限不合法")
        if bool((self.max_delta < 0.0).any()):
            raise ValueError("max_delta 不能为负")

    def forward(self, previous_target, action):
        desired = (action * self.scale + self.offset).clamp(self.lower, self.upper)
        previous = previous_target.clamp(self.lower, self.upper)

        delta = (desired - previous).clamp(-self.max_delta, self.max_delta)

        return (previous + delta).clamp(self.lower, self.upper)


class WorldModel(nn.Module):

    def __init__(
        self,
        state_spec: StateSpec,
        context_dim: int,
        limiter: TargetLimiter,
        hidden_dims=(2048, 2048, 2048),
    ):
        """
        保存配置和约束模块：
            状态结构 state_spec、
            物理上下文的维度 context_dim
            和动作目标限制器 limiter
        构建预测网络：
            确定输入、输出维度，
            根据 hidden_dims 创建隐藏层，
            最后输出物理状态增量。
        初始化输出层：
            权重设为小随机值，
            偏置设为零
        注册归一化相关 buffer：
            input_mean、input_std 保存输入均值和标准差，
            stats_ready 标记统计量是否准备好。
        注册输出缩放 buffer：
            delta_scale 用于把网络输出转换为对应物理量的增量尺度。
        """
        super().__init__()

        self.spec = state_spec
        self.context_dim = int(context_dim)
        self.limiter = limiter

        if limiter.offset.numel() != self.spec.num_actions:
            raise ValueError("limiter 与动作维度不一致")
        if (
            self.context_dim <= 0
            or not hidden_dims
            or min(hidden_dims) <= 0
        ):
            raise ValueError("输入维度和隐藏层宽度必须为正")

        # 动力学网络只接收：
        # 根位置、根旋转6D、根速度、关节位置、关节速度、
        # 本步实际执行目标和物理上下文。
        #
        # 历史动作仍保存在状态中，但不输入动力学网络。
        imput_dim = (
            15
            + 2 * self.spec.num_joints
            + self.spec.num_actions
            + self.context_dim
        )

        # 根位置增量3 + 旋转向量3 + 根速度增量6
        # + 关节位置增量J + 关节速度增量J。
        physical_dim = (
            12 + 2 * self.spec.num_joints
        )

        layers = []             # 创建空列表，用来保存各个网络层
        width = imput_dim       # 记录下一层的输入维度，初始为模型输入维度

        # 根据 hidden_dims 自动构建多层网络，并保证相邻层的维度匹配。
        for following in hidden_dims:
            layers.extend(
                (
                    nn.Linear(width, following),
                    nn.SiLU(),
                )
            )
            width = following

        # 预测物理状态的变化量的总维度
        layers.append(
            nn.Linear(width, physical_dim)
        )
        self.network = nn.Sequential(*layers)

        # 初始化最后一层，让模型刚开始预测的状态增量较小，同时保留输出对动作的梯度。
        nn.init.normal_(
            self.network[-1].weight,
            std=1e-3,
        )
        nn.init.zeros_(
            self.network[-1].bias,
        )

        self.register_buffer(
            "input_mean",
            torch.zeros(imput_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "input_std",
            torch.ones(imput_dim, dtype=torch.float32),
        )
        # statistics统计量
        self.register_buffer(
            "stats_ready",
            torch.tensor(False, dtype=torch.bool),
        )

        # X2 初版工程尺度，不是论文超参数。
        # 作用是把网络输出映射回各物理量的增量单位。
        # * 重复列表，+ 拼接列表。
        scales = (
            [0.1] * 3                           # 根部位置增量的尺度
            + [0.3] * 3                         # 根部旋转增量的尺度
            + [1.0] * 3                         # 根部线速度增量的尺度
            + [2.0] * 3                         # 根部角速度增量的尺度
            + [0.2] * self.spec.num_joints      # 关节位置增量的尺度
            + [2.0] * self.spec.num_joints      # 关节速度增量的尺度
        )

        self.register_buffer(
            "delta_scale",
            torch.as_tensor(scales, dtype=torch.float32),
        )

    def features(
        self,
        state,
        action,
        context,
    ):
        """
        拆分状态
        检查形状
        计算受约束目标
        拼接特征
        """
        # 将状态向量拆分为各个字段。
        s = unpack_state(state, self.spec)
        # 状态数量
        batch_size = state.shape[0]

        if action.shape != (
            batch_size, self.spec.num_actions
        ):
            raise ValueError("action 形状错误")
        if context.shape != (
            batch_size, self.context_dim
        ):
            raise ValueError("context 形状错误")

        # 对应TargetLimiter的forward方法，计算新的目标。
        target = self.limiter(
            s["previous_target"],       # 上一步下发的目标，作为变化幅度的基准
            action,                     # 当前动作，用来计算新的期望目标
        )

        features = torch.cat(
            (
                s["root_pose"][:, :3],          # 根位置
                quat_to_rot6d(s["root_pose"][:, 3:]),  # 根旋转四元数转换为6D表示
                s["root_velocity"],             # 根速度
                s["joint_position"],            # 关节位置
                s["joint_velocity"],            # 关节速度
                target,                         # 当前处理后的目标
                context,                        # 物理上下文
            ),
            dim=-1,
        )

        return features, target

    @torch.no_grad()
    def fit_input_stats(self, batch):
        """
        处理多批次数据，构造输入特征，计算并保存每一维特征的样本均值和标准差
        """
        states = batch["states"]
        actions = batch["actions"]
        contexts = batch["context"]
        # actions.shape == (批量大小, 序列长度, 单步动作维度)
        batch_size, horizon, _ = actions.shape
        features, _ = self.features(
            # 指定第一维：序列数量 × 时间步数，第二维由 PyTorch 根据元素总数自动计算
            # buffer 的 states 有 H+1 帧，但动作只有 H 步。计算输入统计量时，应取每个动作执行前的状态，排除最后一帧。
            state=states[:, :-1].reshape(batch_size * horizon, -1),
            action=actions.reshape(batch_size * horizon, -1),
            # 把上下文扩展到每个时间步，再展开成与状态、动作对应的二维输入。None 表示在这里新增一个维度，作为时间维。
            context=contexts[:, None].expand(-1, horizon, -1).reshape(batch_size * horizon, -1),
        )

        if not bool(torch.isfinite(features).all()):
            raise ValueError("归一化数据包含 NaN/Inf")

        # features.shape == [B * H, D]
        # 沿样本维求平均
        self.input_mean.copy_(
            features.mean(dim=0)
        )
        # 计算每个特征在所有样本中的标准差。
        self.input_std.copy_(
            features.std(dim=0, unbiased=False).clamp_min(1e-3)
        )
        self.stats_ready.fill_(True)

    def forward(
        self,
        state,
        action,
        context,
        ):
        """
        从状态向量里提取特征然后归一化
        由网络输出计算状态增量，从而得到新的状态
        """
        if not bool(self.stats_ready):
            raise RuntimeError("输入统计量尚未准备好，请先调用 fit_input_stats")

        # 将状态向量拆分为各个字段。
        s = unpack_state(state, self.spec)
        features, target = self.features(
            state=state,
            action=action,
            context=context,
        )
        normalized = (features - self.input_mean) / self.input_std

        # 将网络输出与 delta_scale 按元素相乘，得到对应物理尺度的状态增量。
        dalta = self.network(normalized) * self.delta_scale

        dp, dr, dv, dq, dqd = torch.split(
            dalta,
            (
                3,                          # 根位置增量
                3,                          # 根旋转向量增量
                6,                          # 根速度增量
                self.spec.num_joints,      # 关节位置增量
                self.spec.num_joints,      # 关节速度增量
            ),
            dim=-1,
        )

        # dr 表示世界坐标系下的旋转增量，采用左乘。
        orientation = unit_quat(
            quat_mul(
                rotation_increment(dr),             # 根旋转向量增量转换为四元数
                unit_quat(s["root_pose"][:, 3:]),
            ),
        )

        root_pose = torch.cat(
            (
                s["root_pose"][:, :3] + dp,          # 根位置增量
                orientation,                         # 根旋转四元数
            ),
            dim=-1,
        )

        return pack_state(
            root_pose=root_pose,
            root_velocity=s["root_velocity"] + dv,
            joint_position=s["joint_position"] + dq,
            joint_velocity=s["joint_velocity"] + dqd,
            previous_action=action,
            previous_target=target,
        )


def rollout_logged_actions(
    model,
    initial_state,
    actions,
    context,
):
    """按记录动作进行自回归预测。

    initial_state: [B, Ds]
    actions: [B, H, A]
    context: [B, Dc]

    返回: [B, H+1, Ds]
    """
    if (
        actions.ndim != 3
        or actions.shape[1] == 0
    ):
        raise ValueError("actions 必须为非空 [B, H, A]")

    predicted = initial_state       # 当前预测状态，最开始是初始状态 s₀
    states = [predicted]            # 先保存 s₀

    for t in range(actions.shape[1]):   # 遍历每个时间步的动作，进行自回归预测
        # 使用当前预测状态和动作进行下一步预测
        predicted = model(
            predicted,
            actions[:, t],
            context,
        )
        states.append(predicted)

    return torch.stack(states, dim=1)



def dynamics_loss(
    predicted_states,
    target_states,
    state_spec,
):
    if (
        predicted_states.shape != target_states.shape
        or predicted_states.ndim != 3
    ):
        raise ValueError("预测和目标必须同形状 [B, K+1, Ds]")
    if predicted_states.shape[1] < 2:
        raise ValueError("至少需要一条转移")

    # 第0帧是给定的真实起点，不计入预测损失。
    p = unpack_state(
        predicted_states[:, 1:].reshape(-1, state_spec.dim),
        state_spec,
    )

    # 由真实下一步状态得到的目标状态
    t = unpack_state(
        target_states[:, 1:]
        .detach()
        .reshape(-1, state_spec.dim),
        state_spec,
    )

    # 以下为初版误差归一化尺度，后续根据真实误差分析调整。
    terms = {
        "root_position": (
            (
                p["root_pose"][:, :3]
                - t["root_pose"][:, :3]
            ) / 0.05
        ).square().mean(),

        "root_rotation": (
            rotation_error(
                p["root_pose"][:, 3:],
                t["root_pose"][:, 3:],
            ) / 0.1
        ).square().mean(),

        "root_linear": (
            (
                p["root_velocity"][:, :3]
                - t["root_velocity"][:, :3]
            ) / 0.5
        ).square().mean(),

        "root_angular": (
            (
                p["root_velocity"][:, 3:]
                - t["root_velocity"][:, 3:]
            ) / 1.0
        ).square().mean(),

        "joint_position": (
            (
                p["joint_position"]
                - t["joint_position"]
            ) / 0.1
        ).square().mean(),

        "joint_velocity": (
            (
                p["joint_velocity"]
                - t["joint_velocity"]
            ) / 1.0
        ).square().mean(),
    }

    total_loss = torch.stack(
        list(terms.values()),
    ).sum()

    return total_loss, terms


