import math

import torch
from torch import nn
from rsl_rl.modules import EmpiricalNormalization

from .state import StateSpec, pack_state, unpack_state
from .representations import GenCoordRepresentation
from ...utils.rotation import (
    normalize_quaternion as unit_quat,
    quaternion_multiply as quat_mul,
    rotation_vector_to_quaternion as rotation_increment,
    quaternion_to_rotation_6d as quat_to_rot6d,
    quaternion_angle_error as rotation_error,
    quaternion_to_rotation_matrix,
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

    DYNAMICS_MODE = "gencoord_integrated_v1"
    NORMALIZATION_MODE = "running_state_action_v1"

    def __init__(
        self,
        state_spec: StateSpec,
        context_dim: int,
        limiter: TargetLimiter,
        hidden_dims=(2048, 2048, 2048),
        *,
        dt,
        decimation,
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

        self.representation = GenCoordRepresentation(
            state_spec=state_spec,
            dt=dt,
            decimation=decimation,
        )

        self.state_normalizer = EmpiricalNormalization(
            self.representation.rep_state_dim
        )
        self.action_normalizer = EmpiricalNormalization(
            state_spec.num_actions
        )

        # 动力学网络只接收：
        # 根位置、根旋转6D、根速度、关节位置、关节速度、
        # 本步实际执行目标和物理上下文。
        #
        # 历史动作仍保存在状态中，但不输入动力学网络。
        input_dim = (
            self.representation.rep_state_dim
            + state_spec.num_actions
            + self.context_dim
        )

        layers = []             # 创建空列表，用来保存各个网络层
        width = input_dim       # 记录下一层的输入维度，初始为模型输入维度

        # 根据 hidden_dims 自动构建多层网络，并保证相邻层的维度匹配。
        for following in hidden_dims:
            layers.extend(
                (
                    nn.Linear(width, following),
                    nn.LayerNorm(following),
                    nn.GELU(),
                )
            )
            width = following

        # 预测物理状态的变化量的总维度
        layers.append(
            nn.Linear(width, self.representation.prediction_dim)
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
            torch.zeros(input_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "input_std",
            torch.ones(input_dim, dtype=torch.float32),
        )
        # statistics统计量
        self.register_buffer(
            "stats_ready",
            torch.tensor(False, dtype=torch.bool),
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
                self.representation.represent(
                    state
                ),
                target,
                context,
            ),
            dim=-1,
        )

        return features, target

    @torch.no_grad()
    def update_normalization(
        self,
        real_states,
        real_actions,
        context,
    ):
        """仅由真实采样显式调用。

        用真实采样数据更新世界模型输入的均值和标准差，供后续归一化使用。
        """
        features, targets = self.features(
            real_states,
            real_actions,
            context,
        )

        if not bool(torch.isfinite(features).all()):
            raise ValueError("归一化数据包含 NaN/Inf")

        if real_states.shape[0] == 0:
            return

        width = self.representation.rep_state_dim   # 局部状态表示与动作目标维度的的分界线
        end = width + self.spec.num_actions         # 动作目标与物理上下文维度的分界线

        # 物理参数只在第一次计算统计量
        if not bool(self.stats_ready):
            # X2 当前的 context 是原始物理参数，
            # 先保留固定缩放，仅在首次初始化时拟合。
            # 沿样本维计算均值和标准差：
            self.input_mean[end:].copy_(
                context.mean(dim=0)
            )
            self.input_std[end:].copy_(
                context.std(
                    dim=0,
                    unbiased=False,
                ).clamp_min(1e-3)
            )

        for normalizer, values in (
            (
                self.state_normalizer,
                features[:, :width],
            ),
            (
                self.action_normalizer,
                targets,
            ),
        ):
            previous_mode = normalizer.training     # 记住原来的模式。

            # EmpiricalNormalization 在 eval 模式下
            # 不更新；这里是显式的真实数据更新入口。
            normalizer.train()      # 切到训练模式，临时打开统计更新开关。

            try:
                normalizer.update(      # 用这批真实数据更新均值、方差。
                    values.detach()
                )
            finally:
                normalizer.train(       # 恢复之前的开关状态。
                    previous_mode
                )

        # 同步现有输入缩放接口。
        # 与 EmpiricalNormalization.forward() 一致：
        # (x - mean) / (std + eps)。
        self.input_mean[:width].copy_(
            self.state_normalizer.mean
        )
        self.input_std[:width].copy_(
            self.state_normalizer.std
            + self.state_normalizer.eps
        )

        self.input_mean[width:end].copy_(
            self.action_normalizer.mean
        )
        self.input_std[width:end].copy_(
            self.action_normalizer.std
            + self.action_normalizer.eps
        )

        self.stats_ready.fill_(True)

    @torch.no_grad()
    def fit_input_stats(self, batch):
        """用于离线数据的首次初始化。

        在线训练由真实采样调用
        update_normalization()，不重复统计 replay。
        """
        if bool(self.stats_ready):
            raise RuntimeError("统计量已经初始化；真实采样应调用 update_normalization")

        states = batch["states"]
        actions = batch["actions"]
        contexts = batch["context"]
        # actions.shape == (批量大小, 序列长度, 单步动作维度)
        batch_size, horizon, _ = actions.shape
        self.update_normalization(
            # 指定第一维：序列数量 × 时间步数，第二维由 PyTorch 根据元素总数自动计算
            # buffer 的 states 有 H+1 帧，但动作只有 H 步。计算输入统计量时，应取每个动作执行前的状态，排除最后一帧。
            real_states=states[:, :-1].reshape(batch_size * horizon, -1),
            real_actions=actions.reshape(batch_size * horizon, -1),
            # 把上下文扩展到每个时间步，再展开成与状态、动作对应的二维输入。None 表示在这里新增一个维度，作为时间维。
            context=contexts[:, None].expand(-1, horizon, -1).reshape(batch_size * horizon, -1),
        )

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

        features, target = self.features(
            state=state,
            action=action,
            context=context,
        )
        normalized = (features - self.input_mean) / self.input_std

        prediction = self.network(
            normalized
        )

        aux_state = (
            self.representation.extract_aux_state(
                state
            )
        )
        rep_state = (
            self.representation.represent(
                state
            )
        )

        next_aux, next_rep = (
            self.representation.apply_prediction(
                aux_state,
                rep_state,
                prediction,
            )
        )

        return (
            self.representation.reconstruct_full_state(
                next_aux,
                next_rep,
                previous_action=action,
                previous_target=target,
            )
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
    *,
    # 世界模型训练损失中各类误差的参考尺度
    nominal_error_root_pose_pos=0.01,
    nominal_error_root_pose_ori=0.02,
    nominal_error_joint_pos=0.1,
    nominal_error_joint_vel=3.14,
):
    """官方广义坐标损失。

    监督四组通道：
    根位置、根旋转矩阵、关节位置、关节速度。

    返回：
    - 总加权损失。
    - 各组未加权的原始 MSE。

    第零帧是给定初态，不参与监督。

    从六组改为官方广义坐标版本的四组。根速度通过积分后的根位姿获得监督，不再额外加入独立根速度损失。
    根旋转使用旋转矩阵误差。乘上 sqrt(0.5) 后，其平方和在小角度下近似角度误差平方。
    所有通道整体平均，不是将四组均值直接相加。
    """
    if (
        predicted_states.shape != target_states.shape
        or predicted_states.ndim != 3
        or predicted_states.shape[-1] != state_spec.dim
    ):
        raise ValueError("预测和目标必须同形状 [B, K+1, Ds]")
    if (
        predicted_states.shape[0] < 1
        or predicted_states.shape[1] < 2
    ):
        raise ValueError("至少需要一个样本和一条转移")

    scales = {
        "root_pos": nominal_error_root_pose_pos,
        "root_ori": nominal_error_root_pose_ori,
        "joint_pos": nominal_error_joint_pos,
        "joint_vel": nominal_error_joint_vel,
    }

    if any(
        not math.isfinite(value)
        or value <= 0
        for value in scales.values()
    ):
        raise ValueError("世界模型损失的名义误差必须为有限正数")

    # 第 0 帧是传入的真实初始状态，不是模型预测出来的，所以不参与预测误差计算。
    predicted = unpack_state(
        predicted_states[:, 1:].reshape(-1,state_spec.dim,),
        state_spec,
    )
    target = unpack_state(
        target_states[:, 1:].detach().reshape(-1, state_spec.dim,),
        state_spec,
    )

    def blocks(fields):
        """把各个字段拆分为监督的四组通道。"""
        rotation = quaternion_to_rotation_matrix(
            fields["root_pose"][:, 3:7]
        )

        return {
            "root_pos": fields["root_pose"][:, :3],
            "root_ori": (
                rotation.flatten(-2, -1)
                * math.sqrt(0.5)
            ),
            "joint_pos": fields["joint_position"],
            "joint_vel": fields["joint_velocity"],
        }

    predicted_blocks = blocks(predicted)
    target_blocks = blocks(target)

    squared_errors = {
        name: (
            predicted_blocks[name]
            - target_blocks[name]
        ).square()
        for name in scales
    }

    # 日志记录原始 MSE，便于区分各物理量误差。
    terms = {
        name: error.mean()
        for name, error
        in squared_errors.items()
    }

    # 每个通道按对应名义误差的平方缩放。
    # 拼接后整体平均，对齐官方按通道数归约的方式。
    total_loss = torch.cat(
        [
            error / scales[name] ** 2
            for name, error
            in squared_errors.items()
        ],
        dim=-1,
    ).mean()

    return total_loss, terms


