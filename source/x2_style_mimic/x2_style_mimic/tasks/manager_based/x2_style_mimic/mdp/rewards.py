# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]

    # 接触力历史：[环境数, 历史长度, 脚数, 3]。
    contact_forces = contact_sensor.data.net_forces_w_history.torch[
        :, :, sensor_cfg.body_ids, :
    ]

    in_contact = (
        contact_forces.norm(dim=-1).amax(dim=1)
        >force_threshold
    )

    foot_velocity_xy = robot.data.body_lin_vel_w.torch[
        :, asset_cfg.body_ids, :2
    ]

    speed_squared = torch.sum(
        torch.square(foot_velocity_xy),
        dim=-1,
    )

    return torch.sum(speed_squared * in_contact, dim=1)


def self_contact_history_count(
    force_history: torch.Tensor,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """统计最近物理采样中出现自身接触的次数，每个采样最多计一次。

    force_history: [num_envs, history_length]，每个采样所有配对的最大力模长。
    不按接触对数量加倍，不使用持续时间门控，也不在此乘 dt 或负权重。
    """
    return (force_history > force_threshold).sum(dim=-1).float()


# def self_contact_penalty(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
# ) -> torch.Tensor:
#     """历史自身接触计数，读取缓存而不推进传感器。"""
#     sensor = env.scene.sensors[sensor_cfg.name]
#     return self_contact_history_count(
#         sensor.force_history, sensor.cfg.force_threshold
#     )

def self_contact_pressure_cost(
    pair_force: torch.Tensor,           # 每个刚体配对之间的接触力大小。
    strong_duration: torch.Tensor,
    force_threshold: float,
    force_scale: float,
    grace_seconds: float,               # 允许刚体短时间接触而不受惩罚的宽限时间。
    ramp_seconds: float,                # 超过宽限时间后，持续时间因子从 0 增加到 1 所需的时间。
) -> torch.Tensor:
    """每个配对的强接触力与持续时间共同决定代价。"""
    force_factor = (
        (pair_force - force_threshold) / force_scale
    ).clamp(0.0, 1.0)

    duration_factor = (
        (strong_duration - grace_seconds) / ramp_seconds
    ).clamp(0.0, 1.0)

    return (force_factor * duration_factor).sum(dim=-1)


def impact_force_cost(force, threshold, scale):
    """超过阈值后平滑增长，不在固定力值处提前饱和。"""
    excess = (force - threshold).clamp_min(0.0) / scale

    # 小超限平滑起步，大超限仍增长，但不会像平方项一样急剧放大。
    return torch.log1p(excess.square())


def self_contact_control_cost(sensor) -> torch.Tensor:
    """持续项取物理子步平均，冲击项取每个配对的周期峰值。"""
    persistent_cost = (
        sensor.pressure_cost_sum
        / sensor._updates.clamp_min(1).to(
            sensor.pressure_cost_sum.dtype
        )
    )

    impact_cost = impact_force_cost(
        sensor.peak_force,
        sensor.cfg.impact_force_threshold,
        sensor.cfg.impact_force_scale,
    ).sum(dim=-1)

    return persistent_cost + sensor.cfg.impact_weight * impact_cost


def self_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """只读取缓存，不推进传感器或接触计时。"""
    return self_contact_control_cost(
        env.scene.sensors[sensor_cfg.name]
    )