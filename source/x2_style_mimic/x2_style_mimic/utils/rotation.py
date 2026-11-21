"""可微旋转运算。

四元数统一采用 XYZW。
旋转矩阵的 6D 表示按前两列逐列展开。
旋转函数要求输入为单位四元数。
"""

import math

import torch


def normalize_quaternion(quaternion, eps=1e-8):
    """归一化四元数，调用方应保证输入非零。"""
    norm = quaternion.norm(
        dim=-1,
        keepdim=True,
    )

    return quaternion / norm.clamp_min(eps)


def quaternion_conjugate(quaternion):
    """单位四元数的共轭同时表示逆旋转。"""
    return torch.cat(
        (
            -quaternion[..., :3],
            quaternion[..., 3:4],
        ),
        dim=-1,
    )


def quaternion_multiply(first, second):
    """Hamilton 乘法，支持 PyTorch 广播。

    first ⊗ second 表示先应用 second，再应用 first。
    """
    first, second = torch.broadcast_tensors(
        first,
        second,
    )

    first_xyz = first[..., :3]
    first_w = first[..., 3:4]

    second_xyz = second[..., :3]
    second_w = second[..., 3:4]

    xyz = (
        first_w * second_xyz
        + second_w * first_xyz
        + torch.cross(
            first_xyz,
            second_xyz,
            dim=-1,
        )
    )

    w = (
        first_w * second_w
        - (first_xyz * second_xyz).sum(
            dim=-1,
            keepdim=True,
        )
    )

    return torch.cat((xyz, w), dim=-1)


def rotate_vector(quaternion, vector):
    """使用单位四元数正向旋转向量。"""
    xyz, vector = torch.broadcast_tensors(
        quaternion[..., :3],
        vector,
    )

    w = quaternion[..., 3:4]

    cross = 2.0 * torch.cross(
        xyz,
        vector,
        dim=-1,
    )

    return (
        vector
        + w * cross
        + torch.cross(xyz, cross, dim=-1)
    )


def inverse_rotate_vector(quaternion, vector):
    """使用单位四元数，将世界向量旋转到局部坐标。"""
    xyz, vector = torch.broadcast_tensors(
        quaternion[..., :3],
        vector,
    )

    w = quaternion[..., 3:4]

    cross = 2.0 * torch.cross(
        xyz,
        vector,
        dim=-1,
    )

    return (
        vector
        - w * cross
        + torch.cross(xyz, cross, dim=-1)
    )


def rotation_vector_to_quaternion(vector):
    """旋转向量 [rad] 转为 XYZW 四元数。"""
    angle = vector.norm(
        dim=-1,
        keepdim=True,
    )

    xyz = (
        0.5
        * torch.sinc(angle / (2.0 * math.pi))
        * vector
    )

    w = torch.cos(0.5 * angle)

    return torch.cat((xyz, w), dim=-1)


def quaternion_to_rotation_6d(quaternion):
    """返回旋转矩阵前两列，按列展开为六维向量。"""
    x, y, z, w = normalize_quaternion(
        quaternion
    ).unbind(dim=-1)

    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),

            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ),
        dim=-1,
    )


def quaternion_angle_error(first, second):
    """姿态角误差 [rad]，范围为 [0, pi]。"""
    first = normalize_quaternion(first)
    second = normalize_quaternion(second)

    relative = quaternion_multiply(
        first,
        quaternion_conjugate(second),
    )

    return 2.0 * torch.atan2(
        relative[..., :3].norm(dim=-1),
        relative[..., 3].abs(),
    )