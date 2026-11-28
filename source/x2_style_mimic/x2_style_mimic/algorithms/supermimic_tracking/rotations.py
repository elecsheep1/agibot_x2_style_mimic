"""跟踪移植层使用 WXYZ；只在 PhysX/FK 边界转换 XYZW，避免混用。"""
import torch
from ...utils import rotation as r


def to_xyzw(q):
    return torch.cat((q[..., 1:], q[..., :1]), -1)


def to_wxyz(q):
    return torch.cat((q[..., -1:], q[..., :3]), -1)


def quat_normalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_mul(a, b):
    return to_wxyz(r.quaternion_multiply(to_xyzw(a), to_xyzw(b)))


def quat_inv(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


quat_conjugate = quat_inv


def quat_apply(q, v):
    return r.rotate_vector(to_xyzw(q), v)


def quat_apply_inverse(q, v):
    return r.inverse_rotate_vector(to_xyzw(q), v)


def quat_canonical(q):
    return torch.where(q[..., :1] < 0, -q, q)


def quat_from_rotvec(v):
    return to_wxyz(r.rotation_vector_to_quaternion(v))


def rotvec_from_quat(q):
    q = quat_canonical(quat_normalize(q))
    xyz = q[..., 1:]
    squared = xyz.square().sum(-1, keepdim=True)
    length = squared.clamp_min(1e-30).sqrt()
    small = squared < 1e-12
    angle = 2 * torch.atan2(length, q[..., :1])
    factor = torch.where(small, 2 + squared / 3, angle / length)
    return xyz * factor


def quat_box_minus(a, b):
    return rotvec_from_quat(quat_mul(a, quat_inv(b)))


def matrix_from_quat(q):
    return r.quaternion_to_rotation_matrix(to_xyzw(q))


def rot6d_from_quat(q):
    return matrix_from_quat(q)[..., :, :2].flatten(-2)


def heading_yaw(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def heading_quat(q):
    half = heading_yaw(q) / 2
    zero = torch.zeros_like(half)
    return torch.stack((half.cos(), zero, zero, half.sin()), -1)


yaw_quat = heading_quat


def projected_gravity(q):
    g = torch.zeros_like(q[..., 1:])
    g[..., 2] = -1
    return quat_apply_inverse(q, g)


def wrap_to_pi(a):
    return (a + torch.pi) % (2 * torch.pi) - torch.pi
