import json

import numpy as np
import torch

AMP_KEY_BODY_NAMES = (
    "torso_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

# 根状态 10 + 关节状态 58 + 11 个刚体 × 15。
AMP_STATE_DIM = 68 + 15 * len(AMP_KEY_BODY_NAMES)
AMP_FEATURE_VERSION = "x2_amp_relative_body_v1_233"

def normalize_quaternion(q):
    norm = q.norm(dim=-1, keepdim=True)
    if bool((norm < 1e-8).any()):
        raise ValueError("四元数包含零向量")
    return q / norm

def quaternion_conjugate(q):
    """计算四元数共轭。"""
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)

def quaternion_multiply(a, b):
    """xyzw 四元数乘法，支持广播。"""
    av = a[..., :3]
    bv = b[..., :3]
    av, bv = torch.broadcast_tensors(av, bv)
    xyz = (a[..., 3:4] * bv
        + b[..., 3:4] * av
        + torch.cross(av, bv, dim=-1)
    )
    w = (
        a[..., 3:4] * b[..., 3:4]
        - (av * bv).sum(dim=-1, keepdim=True)
    )
    return torch.cat((xyz, w), dim=-1)

def inverse_rotate(quaternion, vectors):
    """使用 xyzw 四元数，将世界向量转换到根坐标系。"""
    xyz, vectors = torch.broadcast_tensors(quaternion[..., :3], vectors)
    w = quaternion[..., 3:4]
    t = 2.0 * torch.cross(xyz, vectors, dim=-1)     # 计算叉积
    return vectors - w * t + torch.cross(xyz, t, dim=-1)

def relative_body_features(
    root_pos_w,
    root_quat_xyzw,
    root_lin_vel_w,
    root_ang_vel_w,
    body_pose_w,
    body_vel_w,
):
    """在完整骨盆坐标系中计算相对位姿和运动。

    body_pose_w: [N, B, 7]，位置 + xyzw 四元数。
    body_vel_w:  [N, B, 6]，刚体原点线速度 + 世界角速度。

    relative_linear 是相对位置在骨盆坐标系中的时间导数，
    因而包含骨盆旋转引起的 omega × offset 修正。
    """
    root_q = normalize_quaternion(root_quat_xyzw)
    body_q = normalize_quaternion(body_pose_w[..., 3:7])

    offset_w = body_pose_w[..., :3] - root_pos_w[..., None, :]  # [N, B, 3]

    relative_position = inverse_rotate(root_q[:, None, :], offset_w)  # [N, B, 3]
    relative_q = quaternion_multiply(
        quaternion_conjugate(root_q)[:, None, :],
        body_q,
    )  # [N, B, 4]

    # 相对旋转矩阵的前两列。
    # 使用旋转矩阵表达，避免 q 与 -q 的符号不连续。
    axes = torch.eye(
        3,
        device=body_pose_w.device,
        dtype=body_pose_w.dtype,
    )[:2]

    relative_rotation_6d = inverse_rotate(
        quaternion_conjugate(relative_q)[..., None, :],  # [N, B, 1, 4]
        axes[None, None, :, :],  # [1, 1, 2, 3]
    ).flatten(start_dim=-2)  # [N, B, 6]

    root_angular = root_ang_vel_w[:, None, :].expand_as(offset_w)  # [N, 1, 3]

    relative_linear = inverse_rotate(
        root_q[:, None, :],
        body_vel_w[..., :3] - root_lin_vel_w[:, None, :]
        - torch.cross(root_angular, offset_w, dim=-1),
    )  # [N, B, 3]

    relative_angular = inverse_rotate(
        root_q[:, None, :],
        body_vel_w[..., 3:6] - root_ang_vel_w[:, None, :],
    )  # [N, B, 3]

    return (
        relative_position,
        relative_rotation_6d,
        relative_linear,
        relative_angular,
    )

def build_amp_state(
    root_pos_w,         # 根位置
    root_quat_xyzw,     # 根四元数
    root_lin_vel_w,     # 根线速度
    root_ang_vel_w,     # 根角速度
    joint_pos,          # 关节位置
    joint_vel,          # 关节速度
    key_pose_w,          # 关键刚体位置
    key_vel_w,          # 关键刚体速度
    ground_height=0.0,  # 地面高度基准
):
    """参考数据和仿真观测共用的特征构造函数。"""
    root_q = normalize_quaternion(root_quat_xyzw)

    # 世界系中的单位重力方向
    gravity_w = torch.zeros_like(root_pos_w)
    gravity_w[:, 2] = -1.0

    position, rotation, linear, angular = relative_body_features(
        root_pos_w,
        root_q,
        root_lin_vel_w,
        root_ang_vel_w,
        key_pose_w,
        key_vel_w,
    )

    state = torch.cat(
        (
            root_pos_w[:, 2:3] - ground_height,  # 根高度
            inverse_rotate(root_q, gravity_w),  # 根坐标系中的重力方向
            inverse_rotate(root_q, root_lin_vel_w),  # 根坐标系中的线速度
            inverse_rotate(root_q, root_ang_vel_w) * 0.25,  # 根坐标系中的角速度，缩放
            joint_pos,  # 关节位置
            joint_vel * 0.05,  # 关节速度
            position.flatten(start_dim=1),  # 关键刚体相对位置
            rotation.flatten(start_dim=1),  # 关键刚体相对旋转
            linear.flatten(start_dim=1),  # 关键刚体相对线速度
            angular.flatten(start_dim=1) * 0.25,  # 关键刚体相对角速度
        ),
        dim=-1,
    )

    if state.shape[-1] != AMP_STATE_DIM:
        raise ValueError(f"AMP 状态应为 {AMP_STATE_DIM} 维，实际为 {state.shape}")

    return state

def amp_observation(env, asset_cfg):
    """读取仿真状态，生成无噪声的 AMP 观测。"""
    robot = env.scene[asset_cfg.name]

    # 使用根刚体原点的姿态和速度，避免混用质心速度。
    root_pose = robot.data.root_link_pose_w.torch
    root_velocity = robot.data.root_link_vel_w.torch

    return build_amp_state(
        root_pos_w=root_pose[:, :3],
        root_quat_xyzw=root_pose[:, 3:7],
        root_lin_vel_w=root_velocity[:, :3],
        root_ang_vel_w=root_velocity[:, 3:6],
        joint_pos=robot.data.joint_pos.torch[:, asset_cfg.joint_ids],
        joint_vel=robot.data.joint_vel.torch[:, asset_cfg.joint_ids],
        key_pose_w=robot.data.body_link_pose_w.torch[:, asset_cfg.body_ids],
        key_vel_w=robot.data.body_link_vel_w.torch[:, asset_cfg.body_ids],
        ground_height=env.scene.env_origins[:, 2:3],
    )

def body_observation(env, asset_cfg, wrist_only=False):
    """策略侧相对观测；脚和躯干使用完整特征，手腕仅位置和线速度。"""
    robot = env.scene[asset_cfg.name]

    root_pose = robot.data.root_link_pose_w.torch
    root_velocity = robot.data.root_link_vel_w.torch

    parts = relative_body_features(
        root_pos_w=root_pose[:, :3],
        root_quat_xyzw=root_pose[:, 3:7],
        root_lin_vel_w=root_velocity[:, :3],
        root_ang_vel_w=root_velocity[:, 3:6],
        body_pose_w=robot.data.body_link_pose_w.torch[:, asset_cfg.body_ids],
        body_vel_w=robot.data.body_link_vel_w.torch[:, asset_cfg.body_ids],
    )

    if wrist_only:
        selected = (parts[0], parts[2])  # 仅位置和线速度
    else:
        selected = parts  # 全部特征

    return torch.cat(
        [part.flatten(start_dim=1) for part in selected],
        dim=-1,
    )

def reference_body_velocity(position, quaternion, dt):
    """由相邻参考帧估算刚体原点世界线速度和世界角速度。"""
    if len(position) < 2:
        raise ValueError("至少需要两帧参考数据")

    q = normalize_quaternion(quaternion)

    # 中间帧用跨两帧差分，首尾使用单边差分。
    previous_q = torch.cat((q[:1], q[:-1]), dim=0)
    following_q = torch.cat((q[1:], q[-1:]), dim=0)

    delta_q = quaternion_multiply(
        following_q,
        quaternion_conjugate(previous_q),
    )
    delta_q = normalize_quaternion(delta_q)

    # 选择最短旋转，同时消除 q/-q 等价带来的符号跳变。
    delta_q = torch.where(
        delta_q[..., 3:4] < 0.0,
        -delta_q,
        delta_q,
    )

    vector_norm = delta_q[..., :3].norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, delta_q[..., 3:4])

    angle_scale = torch.where(
        vector_norm > 1e-7,
        angle / vector_norm.clamp_min(1e-8),
        torch.full_like(vector_norm, 2.0),
    )

    duration = torch.full(
        (len(q), 1, 1),
        2.0 * dt,
        device=q.device,
        dtype=q.dtype,
    )
    duration[0] = dt
    duration[-1] = dt
    angular_velocity = delta_q[..., :3] * angle_scale / duration

    linear_velocity = torch.gradient(
        position,
        spacing=(dt,),
        dim=(0,),
        edge_order=1,
    )[0]

    return torch.cat((linear_velocity, angular_velocity), dim=-1)

def name_indices(source_names, wanted_names):
    """按名称映射索引，不依赖模型内部排列。"""
    if (
        len(set(source_names)) != len(source_names)         # set() 把名称变成集合，会去掉重复项。
        or len(set(wanted_names)) != len(wanted_names)
    ):
        raise ValueError("关节或刚体名称重复")

    missing = set(wanted_names) - set(source_names)

    if missing:
        raise ValueError(f"参考数据缺少名称：{sorted(missing)}")

    return [
        source_names.index(name)
        for name in wanted_names
    ]


class AMPReferenceMotion:
    """加载参考动作，采样片段内部的相邻状态。"""
    def __init__(
        self,
        motion_file,
        joint_names,
        device,
        step_dt=0.02,
    ):
        if len(joint_names) != 29:
            raise ValueError("AMP 要求使用策略的 29 个关节")

        with np.load(motion_file, allow_pickle=False) as data:          # 加载 .npz 文件。
            metadata = json.loads(str(data["metadata_json"].item()))    # 解析成字典

            if (
                metadata["schema_version"] != 1
                or metadata["quat_order"] != "xyzw"
                or metadata["root_frame"] != "pelvis"
                or metadata["up_axis"] != "Z"
            ):
                raise ValueError("参考数据格式或坐标系不匹配")

            times = data["time_s"]      # 读取每帧时间戳

            if len(times) < 2 or not np.allclose(
                np.diff(times),         # 计算相邻时间之差。
                step_dt,
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError("参考采样间隔必须等于策略控制间隔")

            joint_ids = name_indices(
                data["joint_names"].tolist(),
                list(joint_names),
            )
            body_ids = name_indices(
                data["body_names"].tolist(),
                list(AMP_KEY_BODY_NAMES),
            )

            def tensor(key, indices=None):
                values = data[key]
                if indices is not None:
                    values = values[:, indices]

                if not np.isfinite(values).all():
                    raise ValueError(f"参考字段 {key} 包含非有限值")

                return torch.tensor(
                    values,
                    dtype=torch.float32,
                    device=device,
                )

            body_position = tensor("body_pos_w", body_ids)
            body_quaternion = tensor("body_quat_xyzw", body_ids)
            body_pose = torch.cat((body_position, body_quaternion), dim=-1)
            body_velocity = reference_body_velocity(
                body_position,
                body_quaternion,
                step_dt,
            )

            self.states = build_amp_state(
                root_pos_w=tensor("root_pos_w"),
                root_quat_xyzw=tensor("root_quat_xyzw"),
                root_lin_vel_w=tensor("root_lin_vel_w"),
                root_ang_vel_w=tensor("root_ang_vel_w"),
                joint_pos=tensor("joint_pos", joint_ids),
                joint_vel=tensor("joint_vel", joint_ids),
                key_pose_w=body_pose,
                key_vel_w=body_velocity,
                ground_height=float(metadata["ground_z"]),
            )
            if not bool(torch.isfinite(self.states).all()):
                raise ValueError("生成的 AMP 特征包含非有限值")

        self.device = device
        self.step_dt = step_dt
        self.transition_count = len(self.states) - 1        # 计算有效相邻转移数量。
        self.feature_version = AMP_FEATURE_VERSION          # 保存特征版本，供检查点保存和恢复校验使用。

    def sample(self, batch_size):
        """采样 i 和 i+1，禁止将末帧连接到首帧。"""
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正数")

        indices = torch.randint(
            self.transition_count,
            (batch_size,),
            device=self.device,
        )

        return (
            self.states[indices],
            self.states[indices + 1],
        )