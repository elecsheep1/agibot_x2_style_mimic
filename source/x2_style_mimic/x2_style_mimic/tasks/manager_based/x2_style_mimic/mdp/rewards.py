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
