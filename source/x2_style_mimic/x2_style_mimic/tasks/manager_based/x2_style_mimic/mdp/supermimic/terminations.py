import math

import torch

from x2_style_mimic.utils.rotation import (
    normalize_quaternion,
    inverse_rotate_vector,
)


def velocity_task_fall(
    env,
    minimum_height,
    maximum_tilt,
    asset_cfg,
):
    robot = env.scene[asset_cfg.name]

    pose = robot.data.root_link_pose_w.torch

    height = (
        pose[:, 2]
        - env.scene.env_origins[:, 2]
    )

    quaternion = normalize_quaternion(
        pose[:, 3:7]
    )

    gravity = torch.zeros_like(
        pose[:, :3]
    )
    gravity[:, 2] = -1.0

    root_up = -inverse_rotate_vector(
        quaternion,
        gravity,
    )[:, 2]

    low = height < minimum_height

    tilted = (
        root_up < math.cos(maximum_tilt)
    )

    return low | tilted