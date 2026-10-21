import json

import numpy as np
import torch

AMP_KEY_BODY_NAMES = (
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

AMP_STATE_DIM = 80
AMP_FEATURE_VERSION ="x2_amp_v0_80"

def inverse_rotate(quaternion, vectors):
    """使用 xyzw 四元数，将世界向量转换到根坐标系。"""
    xyz = quaternion[..., :3].expand_as(vectors)                # ...表示前面的维度全部保留。量通过广播扩展到与 vectors 相同的形状。
    w = quaternion[..., 3:4].expand(vectors.shape[:-1] + (1,))  # vectors.shape[:-1]取得向量形状中除了最后一维之外的部分。在形状后面追加一个长度为 1 的维度：

    t = 2.0 * torch.cross(xyz, vectors, dim=-1)     # 计算叉积

    return vectors - w * t + torch.cross(xyz, t, dim=-1)

def build_amp_state(
    root_pos_w,         # 根原点世界位置
    root_quat_xyzw,     # 根姿态
    root_lin_vel_w,     # 根原点世界系线速度
    root_ang_vel_w,     # 世界系角速度
    joint_pos,          # 29 个关节位置
    joint_vel,          # 29 个关节速度
    key_pos_w,          # 四个关键刚体世界位置
    ground_height=0.0,  # 地面高度基准
):
    """参考数据和仿真观测共用的特征构造函数。"""
    # 归一化四元数
    quaternion = (
        root_quat_xyzw
        / root_quat_xyzw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    )

    # 世界系中的单位重力方向
    gravity_w = torch.zeros_like(root_pos_w)
    gravity_w[:, 2] = -1.0

    height = root_pos_w[:, 2:3] - ground_height

    # 根坐标系中的重力方向
    gravity_b = inverse_rotate(
        quaternion,
        gravity_w,
    )

    linear_velocity_b = inverse_rotate(
        quaternion,
        root_lin_vel_w,
    )

    angular_velocity_b = inverse_rotate(
        quaternion,
        root_ang_vel_w,
    )

    key_position_b = inverse_rotate(
        quaternion[:, None, :],                 # None 在张量索引中表示增加一个长度为 1 的维度。
        key_pos_w - root_pos_w[:, None, :]
    )

    # 固定缩放，两侧必须一致
    state = torch.cat(
        (
            height,                                     # 1dim
            gravity_b,                                  # 3dim
            linear_velocity_b,                          # 3dim
            angular_velocity_b * 0.25,                  # 3dim  （缩放）
            joint_pos,                                  # 29dim
            joint_vel * 0.05,                           # 29dim （缩放）
            key_position_b.flatten(start_dim=1),        # 12dim
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
        joint_pos=robot.data.joint_pos.torch[
            :, asset_cfg.joint_ids
        ],
        joint_vel=robot.data.joint_vel.torch[
            :, asset_cfg.joint_ids
        ],
        key_pos_w=robot.data.body_link_pose_w.torch[
            :, asset_cfg.body_ids, :3
        ],
        ground_height=env.scene.env_origins[:, 2:3],
    )

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
    """加载单条参考动作，采样片段内部的相邻状态。"""
    def __init__(
        self,
        motion_file,
        joint_names,
        device,
        step_dt=0.02,
    ):
        if len(joint_names) != 29:
            raise ValueError("AMP v0 要求使用策略的 29 个关节")

        with np.load(motion_file, allow_pickle=False) as data:      # 加载 .npz 文件。
            metadata = json.loads(                                  # 解析成字典
                str(data["metadata_json"].item())
            )

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
                values = (
                    data[key]
                    if indices is None
                    else data[key][:, indices]      # 保留全部帧，只选指定关节或刚体，并按索引重排。
                )

                if not np.isfinite(values).all():
                    raise ValueError(f"参考字段 {key} 包含非有限值")

                return torch.tensor(
                    values,
                    dtype=torch.float32,
                    device=device,
                )

            self.states = build_amp_state(
                root_pos_w=tensor("root_pos_w"),
                root_quat_xyzw=tensor("root_quat_xyzw"),
                root_lin_vel_w=tensor("root_lin_vel_w"),
                root_ang_vel_w=tensor("root_ang_vel_w"),
                joint_pos=tensor("joint_pos", joint_ids),
                joint_vel=tensor("joint_vel", joint_ids),
                key_pos_w=tensor("body_pos_w", body_ids),
                ground_height=float(metadata["ground_z"]),
            )

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