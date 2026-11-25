"""平地速度任务的可微奖励与终止判断。"""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .state import unpack_state
from ...utils.rotation import (
    normalize_quaternion,
    inverse_rotate_vector,
)


@dataclass(frozen=True)
class GetupTaskCfg:
    step_dt: float = 0.02
    standing_weight: float = 1.0
    style_weight: float = 1.0
    action_rate_weight: float = 0.005
    joint_limit_weight: float = 0.5
    joint_acc_weight: float = 1e-7
    boundary_weight: float = 5.0
    soft_limit_fraction: float = 0.9
    max_distance: float = 2.0
    boundary_transition: float = 0.1
    # 工程初值，单位米；后续根据仿真核对。
    contact_distance: float = 0.005
    contact_transition: float = 0.005
    # 头部接近地面的几何惩罚，不表示接触力。
    head_contact_weight: float = 2.0
    # 离地达到此距离时，惩罚为零，单位米。
    head_contact_distance: float = 0.005
    # 从零惩罚到满惩罚的距离跨度，单位米。
    head_contact_transition: float = 0.010
    # 势函数进展奖励
    progress_weight: float = 5.0
    # 势函数折扣；1.0 时整段求和等于终态势减初态势，无法靠悬停刷取。
    progress_gamma: float = 1.0


class GetupTask(nn.Module):
    """倒地起身任务的可微奖励与终止判断。"""

    def __init__(
        self,
        spec,
        joint_limits,
        cfg,
        kinematics,
        control_joint_ids,
        contact_geometry,
    ):
        super().__init__()

        self.spec = spec
        self.cfg = cfg
        self.kinematics = kinematics
        self.contact_geometry = contact_geometry

        if not all(
            math.isfinite(value)
            for value in vars(cfg).values()
        ):
            raise ValueError("起身任务配置必须全部为有限数值")
        if min(
            cfg.step_dt,
            cfg.max_distance,
            cfg.boundary_transition,
            cfg.contact_transition,
            cfg.head_contact_transition,
        ) <= 0:
            raise ValueError("控制周期、越界距离和过渡宽度必须大于零")
        if not 0.0 < cfg.soft_limit_fraction <= 1.0:
            raise ValueError("soft_limit_fraction 必须在 (0, 1] 内")

        weights = (
            cfg.standing_weight,
            cfg.style_weight,
            cfg.action_rate_weight,
            cfg.joint_limit_weight,
            cfg.joint_acc_weight,
            cfg.boundary_weight,
            cfg.head_contact_weight,
            cfg.progress_weight,
        )

        if min(weights) < 0:
            raise ValueError("奖励和惩罚权重必须为非负数")
        if not 0.0 <= cfg.progress_gamma <= 1.0:
            raise ValueError("progress_gamma 必须在 [0, 1] 内")
        if cfg.contact_distance < 0:
            raise ValueError("接触距离阈值不能为负")

        limits = (
            torch.as_tensor(joint_limits)
            .detach()
            .clone()
            .float()
        )

        if limits.shape != (spec.num_joints, 2):
            raise ValueError("joint_limits 必须为 [全部关节数量, 2]")
        if (
            not bool(torch.isfinite(limits).all())
            or bool(
                (limits[:, 0] >= limits[:, 1]).any()
            )
        ):
            raise ValueError("关节限位必须有限，且下限小于上限")

        center = limits.mean(dim=-1)
        half_range = (
            (limits[:, 1] - limits[:, 0])
            * 0.5
            * cfg.soft_limit_fraction
        )
        self.register_buffer(
            "lower_limits",
            center - half_range,
        )
        self.register_buffer(
            "upper_limits",
            center + half_range,
        )

        joint_ids = (
            torch.as_tensor(
                control_joint_ids,
                dtype=torch.long,
            )
            .detach()
            .clone()
        )

        if (
            joint_ids.shape != (spec.num_actions,)
            or joint_ids.unique().numel() != joint_ids.numel()
            or bool(
                (
                    (joint_ids < 0) | (joint_ids >= spec.num_joints)
                ).any()
            )
        ):
            raise ValueError("控制关节索引必须数量正确、不重复且未越界")

        self.register_buffer(
            "control_joint_ids",
            joint_ids,
        )

        body_names = list(kinematics.body_names)

        if "torso_link" not in body_names:
            raise ValueError("可微运动学必须包含 torso_link")

        self.torso_index = body_names.index(
            "torso_link"
        )

    @staticmethod
    def upright_component(quaternion):
        """计算刚体局部竖直轴在世界竖直方向上的分量。"""
        quaternion = normalize_quaternion(
            quaternion
        )

        # 1 - 2sin²(θ/2) = cos(θ)
        # 计算的是机器人自身 Z 轴与世界 Z 轴夹角的余弦，用来衡量直立程度
        return 1.0 - 2.0 * (
            quaternion[..., 0].square()
            + quaternion[..., 1].square()
        )

    def contact_proximity(self, state, clearance=None):
        """几何接近程度，不代表承重或接触力。"""
        if clearance is None:
            clearance = self.contact_geometry(state)

        return torch.sigmoid(
            (self.cfg.contact_distance - clearance)
            / self.cfg.contact_transition
        )

    def head_contact_cost(self, state, clearance=None):
        """头部接近地面的有界距离惩罚，不表示接触力。"""
        if clearance is None:
            clearance = self.contact_geometry(state)
        clearance = clearance[:, 4:6]

        level = (
            (self.cfg.head_contact_distance - clearance)
            / self.cfg.head_contact_transition
        ).clamp(0.0, 1.0)

        proximity = level.square() * (3.0 - 2.0 * level)
        return proximity.amax(dim=-1)

    @property
    def max_step_reward(self):
        """单步总奖励理论上限；势函数项已抵消公共 dt。"""
        return (
            self.cfg.step_dt * (
                self.cfg.standing_weight
                + self.cfg.style_weight
            )
            + self.cfg.progress_weight
        )

    def upright_fields(self, state):
        """一次 FK 返回状态字段、高度和根/躯干直立程度。"""
        fields = unpack_state(state, self.spec)

        body_pose, _ = self.kinematics(
            root_pose=fields["root_pose"],
            root_velocity=fields["root_velocity"],
            joint_position=fields["joint_position"],
            joint_velocity=fields["joint_velocity"],
        )

        height = fields["root_pose"][:, 2]
        root_up = self.upright_component(fields["root_pose"][:, 3:7])
        torso_up = self.upright_component(body_pose[:, self.torso_index, 3:7])

        return fields, height, root_up, torso_up

    @staticmethod
    def posture_potential(height, root_up, torso_up):
        """与真实环境 standing_geometry_potential 同一口径，范围 [0, 1]。"""
        height_progress = (
            (height - 0.08) / (0.63 - 0.08)
        ).clamp(0.0, 1.0)

        posture_quality = 0.5 * (
            (root_up.clamp(-1.0, 1.0) + 1.0) * 0.5
            + (torso_up.clamp(-1.0, 1.0) + 1.0) * 0.5
        )

        return height_progress * (0.25 + 0.75 * posture_quality)

    def geometry_potential(self, state):
        """势函数只需要姿态，不计算接触几何。"""
        _, height, root_up, torso_up = self.upright_fields(state)
        return self.posture_potential(height, root_up, torso_up)

    def standing_and_potential(self, state):
        """一次 FK、一次接触几何，同时返回维持奖励、势函数和 clearance。

        高度进展只由势函数项提供；站立项只负责接近站立后的维持，
        与真实环境一致，避免重复奖励同一种进展。
        """
        fields, height, root_up, torso_up = self.upright_fields(state)

        clearance = self.contact_geometry(state)
        potential = self.posture_potential(height, root_up, torso_up)

        near_standing = (
            (height - 0.48) / (0.60 - 0.48)
        ).clamp(0.0, 1.0)
        near_standing = (
            near_standing.square() * (3.0 - 2.0 * near_standing)
        )

        upright_gate = (
            (torch.minimum(root_up, torso_up) - 0.50)
            / (math.cos(math.radians(10.0)) - 0.50)
        ).clamp(0.0, 1.0)

        linear_velocity = fields["root_velocity"][:, :3]
        angular_velocity = fields["root_velocity"][:, 3:6]

        stability_quality = torch.exp(
            -linear_velocity.square().sum(dim=-1) / 0.25**2
            - angular_velocity.square().sum(dim=-1) / 0.60**2
        )

        proximity = self.contact_proximity(state, clearance)

        feet_support = proximity[:, :2].prod(dim=-1)
        hands_clear = (1.0 - proximity[:, 2:4]).prod(dim=-1)
        support_quality = feet_support * hands_clear

        standing_score = stability_quality * support_quality

        quality = near_standing * upright_gate * standing_score
        return quality, potential, clearance

    def standing_quality(self, state):
        """对外保持原接口；只表示接近站立后的维持质量。"""
        return self.standing_and_potential(state)[0]

    def reward(
        self,
        before,
        action,
        following,
        goals,
        style_reward,
    ):
        batch_size = following.shape[0]

        if before.shape != following.shape:
            raise ValueError("before 与 following 的形状必须一致")
        if action.shape != (
            batch_size,
            self.spec.num_actions,
        ):
            raise ValueError("action 必须为 [B, 动作数量]")
        if goals.shape != (batch_size, 0):
            raise ValueError("起身任务不使用外部目标，goals 必须为 [B, 0]")
        if style_reward.shape != (batch_size,):
            raise ValueError("style_reward 必须为 [B]")

        previous = unpack_state(
            before,
            self.spec,
        )
        current = unpack_state(
            following,
            self.spec,
        )

        cfg = self.cfg
        joint_ids = self.control_joint_ids

        joint_position = (
            current["joint_position"][:, joint_ids]
        )

        lower_violation = (
            self.lower_limits[joint_ids]
            - joint_position
        ).clamp_min(0.0)

        upper_violation = (
            joint_position
            - self.upper_limits[joint_ids]
        ).clamp_min(0.0)

        joint_violation = (
            lower_violation + upper_violation
        ).sum(dim=-1)

        action_rate = (
            action - previous["previous_action"]
        ).square().sum(dim=-1)

        # 根据相邻状态的真实时间间隔估算关节加速度。
        joint_acceleration = (
            (
                current["joint_velocity"][:, joint_ids]
                - previous["joint_velocity"][:, joint_ids]
            )
            / cfg.step_dt
        )

        joint_acceleration_cost = (
            joint_acceleration.square().sum(dim=-1)
        )

        standing, potential_after, clearance = self.standing_and_potential(following)
        potential_before = self.geometry_potential(before)

        # RewardManager 和这里的公共乘 dt 会抵消，使 progress_weight
        # 直接表示势函数差分系数。注意符号约定：真实环境用 γΦ' − Φ，
        # 这里是 Φ' − γΦ；当前 progress_gamma=1.0 时两者一致。
        progress = (
            potential_after - cfg.progress_gamma * potential_before
        ) / cfg.step_dt

        head_contact = self.head_contact_cost(following, clearance)

        terms = {
            "standing": (
                cfg.standing_weight * standing
            ),
            "style": (
                cfg.style_weight * style_reward
            ),
            "action_rate": (
                -cfg.action_rate_weight * action_rate
            ),
            "joint_limits": (
                -cfg.joint_limit_weight
                * joint_violation
            ),
            "joint_acc": (
                -cfg.joint_acc_weight
                * joint_acceleration_cost
            ),
            "out_of_bounds": (
                -cfg.boundary_weight
                * self.termination_probability(
                    following
                )
                / cfg.step_dt
            ),
            "head_contact": (
                -cfg.head_contact_weight
                * head_contact
            ),
            "progress": (
                cfg.progress_weight * progress
            ),
        }

        # 只在这里乘一次控制周期。
        terms = {
            name: value * cfg.step_dt
            for name, value in terms.items()
        }

        total = torch.stack(
            tuple(terms.values()),
            dim=0,
        ).sum(dim=0)

        return total, terms

    def termination_probability(self, state):
        """前向使用真实越界事件，反向使用平滑代理梯度。"""
        fields = unpack_state(state, self.spec)

        distance = torch.linalg.vector_norm(
            fields["root_pose"][:, :2],
            dim=-1,
        )

        hard = (
            distance > self.cfg.max_distance
        ).to(distance.dtype)

        soft = torch.sigmoid(
            (
                distance - self.cfg.max_distance
            )
            / self.cfg.boundary_transition
        )

        return hard + (soft - soft.detach())

    def terminated(self, state):
        """倒地本身不终止，水平越界才终止。"""
        fields = unpack_state(state, self.spec)

        distance = torch.linalg.vector_norm(
            fields["root_pose"][:, :2],
            dim=-1,
        )

        return distance > self.cfg.max_distance