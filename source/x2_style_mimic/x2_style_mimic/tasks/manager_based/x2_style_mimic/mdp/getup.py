from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .motion_dataset import MotionDataset
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def _write_reference_state(
        env,
        env_ids,
        robot,
        root_pose,
        root_velocity,
        joint_pos,
        joint_vel,
    ):
    """把参考状态写入物理仿真。"""
    # 把根位姿、根速度和关节状态写入物理仿真。 [x, y, z, qx, qy, qz, qw]
    robot.write_root_link_pose_to_sim_index(
        root_pose=root_pose,
        env_ids=env_ids,
    )
    # [vx, vy, vz, wx, wy, wz]
    robot.write_root_link_velocity_to_sim_index(
        root_velocity=root_velocity,
        env_ids=env_ids,
    )
    # 兼容此前发现的根速度写入问题：
    # 保留根原点速度缓存，同时修正 PhysX 接收的质心速度。
    # 质心在刚体坐标系中的位姿。
    # 选择指定环境；
    # 选择第 0 个刚体，通常是根部刚体；
    # 取位置的前三个分量。
    offset_b = robot.data.body_com_pose_b.torch[
        env_ids, 0, :3
    ]
    quaternion = root_pose[:, 3:7]
    # 四元数旋转公式：v_rotated = v + w * t + q_vector × t    t = 2 * (q_vector × v)
    t = 2.0 * torch.cross(
        quaternion[:, :3],
        offset_b,
        dim=-1,
    )

    # 也就是把质心相对根部原点的局部偏移，转换成世界坐标系中的偏移。
    offset_w = (
        offset_b
        + quaternion[:, 3:4] * t
        + torch.cross(
            quaternion[:, :3],
            t,
            dim=-1,
        )
    )

    # 质心速度 v_com = v_origin + ω × r
    com_velocity = root_velocity.clone()
    com_velocity[:, :3] += torch.cross(
        root_velocity[:, 3:6],
        offset_w,
        dim=-1,
    )

    robot.write_root_com_velocity_to_sim_index(
        root_velocity=com_velocity,
        env_ids=env_ids,
    )
    robot.write_joint_state_to_sim_index(
        position=joint_pos,
        velocity=joint_vel,
        env_ids=env_ids,
    )

    # 防止重置后、首个动作到来前使用上个 episode 的目标。
    robot.set_joint_position_target_index(
        target=joint_pos,
        env_ids=env_ids,
    )

    # 当前 PhysX 后端写入根状态后，部分机身坐标系派生缓存
    # 可能仍保留本控制步重置前的值。
    # 使缓存失效，下次读取时根据新姿态和速度重新计算。
    # 遍历一组机器人数据缓存名称。
    for buffer_name in (
        "_projected_gravity_b",
        "_root_link_lin_vel_b",
        "_root_link_ang_vel_b",
        "_root_com_lin_vel_b",
        "_root_com_ang_vel_b",
        "_heading_w",
    ):
        buffer = getattr(robot.data, buffer_name, None)         # 动态获取属性，如果属性不存在，就返回 None
        if buffer is not None:
            buffer.timestamp = -1.0         # 把缓存时间戳设置为 -1.0，表示缓存已经失效。

def reset_from_reference(
    env,
    env_ids,
    motion_file,
    random_frame_probability=0.0,
    start_phase=0.0,
    end_phase=0.95,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """混合首帧初始化与随机参考帧初始化。"""

    if not 0.0 <= random_frame_probability <= 1.0:
        raise ValueError("random_frame_probability 必须在 [0, 1] 内")

    if not 0.0 <= start_phase <= end_phase <= 1.0:
        raise ValueError("参考采样范围必须满足0 <= start_phase <= end_phase <= 1")

    # 根据配置名称从场景中取得机器人对象，默认名称为 "robot"。
    robot = env.scene[asset_cfg.name]

    # env_ids=None 表示重置全部并行环境。
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # 没有待重置环境时直接返回，避免后续创建空张量。
    if len(env_ids) == 0:
        return

    # 参考动作只需要读取一次，之后直接复用缓存在环境对象上的张量。
    cache = getattr(env, "_getup_reference_cache", None)

    # 如果缓存不存在或 motion_file 不同，则重新加载参考动作。
    if (cache is None or cache.get("motion_file") != str(motion_file)):
        with np.load(motion_file, allow_pickle=False) as data:
            # metadata_json 在 npz 中以字符串标量保存。
            metadata = json.loads(str(data["metadata_json"].item()))

            # Isaac Lab 此处写入的根姿态四元数要求顺序为 (x, y, z, w)。
            if (metadata["quat_order"] != "xyzw" or metadata["root_frame"] != "pelvis"):
                raise ValueError("参考坐标系不匹配")

            source_names = data["joint_names"].tolist()

            # 名称集合必须完全一致，否则无法安全地逐关节映射参考姿态。
            if (len(source_names) != len(robot.joint_names) or set(source_names) != set(robot.joint_names)):
                raise ValueError("The reference joints and the robot’s joint set are inconsistent.")

            # 按实际 Isaac 关节顺序重排，包含头部两个关节。
            indices = [
                source_names.index(name)
                for name in robot.joint_names
            ]

            # 读取完整片段，后面按帧编号选择初始化状态。
            # 拼接根部位姿
            pose = np.concatenate(
                (
                    data["root_pos_w"],
                    data["root_quat_xyzw"],
                ),
                axis=-1,
            )

            # 拼接根部线速度和角速度
            root_velocity_data = np.concatenate(
                (
                    data["root_lin_vel_w"],
                    data["root_ang_vel_w"],
                ),
                axis=-1,
            )

            # 按机器人关节顺序重排关节位置和速度
            joint_pos = data["joint_pos"][:, indices].copy()
            joint_vel_data = data["joint_vel"][:, indices].copy()

        # 将 NumPy 数据转到仿真设备上，供所有并行环境重复使用。
        # 把动作缓存到 GPU 或 CPU
        cache = {
            "root_pose": torch.tensor(
                pose,
                dtype=torch.float32,
                device=env.device,
            ),
            "joint_pos": torch.tensor(
                joint_pos,
                dtype=torch.float32,
                device=env.device,
            ),
            "root_velocity": torch.tensor(
                root_velocity_data,
                dtype=torch.float32,
                device=env.device,
            ),
            "joint_vel": torch.tensor(
                joint_vel_data,
                dtype=torch.float32,
                device=env.device,
            ),
            "motion_file": str(motion_file),

        }

        env._getup_reference_cache = cache

    # 得到这次要重置的环境数量。s
    count = len(env_ids)

    last = len(cache["root_pose"]) - 1
    lower = int(start_phase * last)
    upper = int(end_phase * last)

    # 每个环境独立决定是否从随机参考帧出生。
    random_start = (
        torch.rand(count, device=env.device)
        < random_frame_probability
    )

    # 随机选择参考帧编号，范围在 [lower, upper] 之间。
    frames = torch.randint(
        lower,
        upper + 1,
        (count,),
        device=env.device,
    )

    # 未选中随机初始化的环境使用第 0 帧。
    frames = torch.where(
        random_start,
        frames,
        torch.zeros_like(frames),
    )

    root_pose = cache["root_pose"][frames].clone()
    # 动作文件中的根部位置一般是局部坐标，因此需要加上对应环境的世界坐标原点。
    root_pose[:, :3] += env.scene.env_origins[env_ids]      # 把环境原点加到根位置上，得到世界坐标系下的根位置
    root_pose[:, 2] += 0.01                                 # 在 z 方向上加一个小的偏移，避免根部穿透地面

    joint_pos = cache["joint_pos"][frames].clone()
    root_velocity = cache["root_velocity"][frames].clone()
    joint_vel = cache["joint_vel"][frames].clone()

    # 固定首帧保持零速度，随机帧恢复对应参考速度。
    # ~random_start 表示逻辑取反
    root_velocity[~random_start] = 0.0
    joint_vel[~random_start] = 0.0

    # 保存出生帧，方便诊断。
    if not hasattr(env, "_getup_reference_frame_ids"):
        env._getup_reference_frame_ids = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )

    # 保存出生帧编号，方便诊断。
    env._getup_reference_frame_ids[env_ids] = frames

    _write_reference_state(
        env, env_ids, robot,
        root_pose, root_velocity, joint_pos, joint_vel,
    )

def reset_from_motion_dataset(
    env,
    env_ids,
    manifest_file,
    policy_joint_names,
    start_phase=0.0,
    end_phase=0.0,
    use_reference_velocity=False,           # 是否使用参考速度
    zero_initial_frame_velocity=False,      # 是否将初始帧的速度置零
    height_offset=0.01,                     # 高度偏移
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    initial_frame_probability=0.5,
    transition_frame_ranges=None,
):
    """按权重选择动作；默认从首帧、零速度出生。"""
    if not math.isfinite(height_offset) or height_offset < 0.0:
        raise ValueError("height_offset 必须为非负有限数")

    robot = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)

    if env_ids.numel() == 0:
        return

    # 加载一次，后续重置复用。
    # 构造数据集缓存的键。
    key = (
        str(Path(manifest_file).resolve()),
        tuple(robot.joint_names),
        tuple(policy_joint_names),
        str(env.device),
        float(env.step_dt),
    )
    cache = getattr(env, "_getup_motion_dataset_cache", None)
    '''
    cache[0]  # key
    cache[1]  # dataset
    '''

    if cache is None or cache[0] != key:
        dataset = MotionDataset(
            manifest_file=manifest_file,
            robot_joint_names=robot.joint_names,
            policy_joint_names=policy_joint_names,
            device=env.device,
            step_dt=env.step_dt,
        )
        env._getup_motion_dataset_cache = (key, dataset)
    else:
        dataset = cache[1]          # 如果缓存存在且键匹配，则直接使用缓存中的数据集对象

    state = dataset.sample_reset_states(
        len(env_ids),
        start_phase,
        end_phase,
        initial_frame_probability=initial_frame_probability,
        transition_frame_ranges=transition_frame_ranges,
    )

    root_pose = state["root_pose"].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    root_pose[:, 2] += height_offset

    root_velocity = state["root_velocity"].clone()
    joint_pos = state["joint_pos"].clone()
    joint_vel = state["joint_vel"].clone()

    if not use_reference_velocity:
        root_velocity.zero_()
        joint_vel.zero_()
    elif zero_initial_frame_velocity:
        # 混合出生：首帧清零，其他参考帧保留参考速度。
        initial_mask = state["frame_ids"] == 0
        root_velocity[initial_mask] = 0.0
        joint_vel[initial_mask] = 0.0

    # 与当前仿真模型的硬限位对齐，并记录裁剪数量。
    limits = robot.data.joint_pos_limits.torch[env_ids]
    clamped = joint_pos.clamp(
        min=limits[:, :, 0],
        max=limits[:, :, 1],
    )
    # 计算每个环境中被裁剪的关节数量，判断裁剪前后关节位置的绝对差值是否大于 1e-6，然后对每个环境求和得到裁剪数量。
    clamp_count = ((clamped - joint_pos).abs() > 1e-6).sum(dim=-1)
    joint_pos = clamped

    velocity_limits = robot.data.joint_vel_limits.torch[env_ids]
    joint_vel = joint_vel.clamp(
        min=-velocity_limits,
        max=velocity_limits,
    )

    # 保存每个环境的出生动作、帧号和裁剪数量。
    for name in (
        "_getup_reference_frame_ids",
        "_getup_reference_motion_ids",
        "_getup_reference_joint_clamp_counts",
    ):
        if not hasattr(env, name):
            setattr(
                env,
                name,
                torch.zeros(
                    env.num_envs,
                    dtype=torch.long,
                    device=env.device,
                ),
            )

    # 保存每个环境的出生动作、帧号和裁剪数量。
    env._getup_reference_frame_ids[env_ids] = state["frame_ids"]
    env._getup_reference_motion_ids[env_ids] = state["motion_ids"]
    env._getup_reference_joint_clamp_counts[env_ids] = clamp_count
    env._getup_reference_motion_paths = [
        str(path) for path in dataset.paths
    ]

    _write_reference_state(
        env,
        env_ids,
        robot,
        root_pose,
        root_velocity,
        joint_pos,
        joint_vel,
    )

def smoothstep(x):
    x = x.clamp(0.0, 1.0)
    return x.square() * (3.0 - 2.0 * x)

def _torso_rise_quality(robot, torso_cfg):
    """连续奖励抬胸，保留少量低姿态探索收益。"""
    quat = robot.data.body_link_pose_w.torch[
        :, torso_cfg.body_ids, 3:7
    ]
    torso_up = (1.0 - 2.0 * quat[..., :2].square().sum(dim=-1)).amin(dim=-1)

    return 0.10 + 0.90 * torso_up.clamp(0.0, 1.0).square()

def height_reward(
    env,
    minimum_height: float,
    target_height: float,
    inverted_cutoff: float = -0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """鼓励根高度接近站立高度，单位 m。"""
    robot = env.scene[asset_cfg.name]

    # 世界系根高度减去环境原点高度，得到每个环境中的相对高度 [num_envs]。
    height = (
        robot.data.root_pos_w.torch[:, 2]
        - env.scene.env_origins[:, 2]
    )

    # ==================== 高度进度奖励 ====================
    height_progress = (
        (height - minimum_height)
        / (target_height - minimum_height)
    ).clamp(0.0, 1.0)

    # ==================== 姿态门控 ====================
    # 这是与世界竖直方向的点积，不是角度：
    # 直立约为 +1，水平约为 0，倒置约为 -1。
    upright = -robot.data.projected_gravity_b.torch[:, 2]

    # 明显倒置时不给高度奖励。
    # 水平附近仍保留少量反馈，帮助俯卧阶段探索。
    orientation_gate = (
        (upright - inverted_cutoff)
        / (1.0 - inverted_cutoff)
    ).clamp(0.0, 1.0)

    # ==================== 最终奖励 ====================
    return height_progress * orientation_gate

def upright_reward(
    env,
    minimum_height: float,
    full_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """根坐标系竖直向上时为 1，水平或倒置时为 0。"""
    robot = env.scene[asset_cfg.name]

    # ==================== 状态量计算 ====================
    height = (
        robot.data.root_pos_w.torch[:, 2]
        - env.scene.env_origins[:, 2]
    )

    # ==================== 高度门控 ====================
    height_gate = (
        (height - minimum_height)
        / (full_height - minimum_height)
    ).clamp(0.0, 1.0)

    # ==================== 姿态奖励 ====================
    upright = (
        -robot.data.projected_gravity_b.torch[:, 2]
    ).clamp(0.0, 1.0)

    # ==================== 最终奖励 ====================
    return height_gate * upright

def rise_speed_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """身体抬离地面时，惩罚过大的线速度和角速度。"""
    robot = env.scene[asset_cfg.name]

    # ==================== 状态量计算 ====================
    height = (
        robot.data.root_pos_w.torch[:, 2]
        - env.scene.env_origins[:, 2]
    )

    # ==================== 高度与姿态门控 ====================
    # 低于 0.25 m 不启用，到 0.40 m 完全启用。
    height_gate = ((height - 0.25) / 0.15).clamp(0.0, 1.0)

    upright_gate = (
        -robot.data.projected_gravity_b.torch[:, 2]
    ).clamp(0.0, 1.0)

    # ==================== 超限速度惩罚 ====================
    linear_speed = robot.data.root_lin_vel_w.torch.norm(dim=-1)
    angular_speed = robot.data.root_ang_vel_w.torch.norm(dim=-1)

    # 允许正常起身运动，只惩罚超出范围的部分。
    linear_excess = (linear_speed - 0.60).clamp_min(0.0)
    angular_excess = (angular_speed - 1.50).clamp_min(0.0)

    # ==================== 最终惩罚 ====================
    return height_gate * upright_gate * (
        linear_excess.square()
        + 0.10 * angular_excess.square()
    )

def ground_contact_force(
    env,
    sensor_cfg: SceneEntityCfg,
):
    """返回每个环境历史帧中的最大地面接触力，单位 N。"""
    sensor = env.scene.sensors[sensor_cfg.name]
    history = sensor.data.force_matrix_w_history

    if history is None:
        raise RuntimeError(f"{sensor_cfg.name} 缺少地面过滤接触力历史")

    # [环境, 历史帧, 传感刚体, 过滤对象, xyz]
    forces = history.torch

    # 每个传感器只匹配一个刚体和一个地面。
    if forces.shape[2:] != (1, 1, 3):
        raise RuntimeError(f"{sensor_cfg.name} 匹配数量不正确：{forces.shape}")

    return forces.norm(dim=-1).amax(dim=(1, 2, 3))

def current_ground_contact_force(
    env,
    sensor_cfg: SceneEntityCfg,
):
    """返回当前采样的地面接触力大小，形状为 [num_envs]。"""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.force_matrix_w.torch

    if forces.shape[1:] != (1, 1, 3):
        raise RuntimeError(f"{sensor_cfg.name} 必须只匹配一个刚体和一个地面")

    return forces[:, 0, 0, :].norm(dim=-1)

def getup_transition_reward(
    env,
    asset_cfg: SceneEntityCfg,
    torso_cfg: SceneEntityCfg,
    foot_sensor_cfgs: tuple[SceneEntityCfg, ...],
    hand_sensor_cfgs: tuple[SceneEntityCfg, ...],
):
    """为抬髋、改善躯干姿态、伸膝和逐渐离手提供连续反馈。防止卡在蹲伏阶段。"""
    robot = env.scene[asset_cfg.name]

    height = robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]

    # ==================== 高度门控和奖励 ====================
    # 0.25 m 开始介入，0.45 m 完全介入。
    # 避免直接要求俯卧状态伸膝
    gate = ((height - 0.25) / 0.20).clamp(0.0, 1.0)
    gate = gate.square() * (3.0 - 2.0 * gate)

    # 从当前约 0.32 m 的停滞高度，到半蹲高度，始终有反馈。
    height_score = ((height - 0.25) / 0.28).clamp(0.0, 1.0)

    # ==================== 姿态与关节奖励 ====================
    torso_q = robot.data.body_link_pose_w.torch[
        :, torso_cfg.body_ids, 3:7
    ]
    # 1 - 2 * (x² + y²)算出的是躯干自身向上方向与世界竖直向上方向夹角的余弦
    torso_up = (
        1.0
        - 2.0 * torso_q[..., :2].square().sum(dim=-1)
    ).amin(dim=-1).clamp(-1.0, 1.0)
    # 使用连续余弦分数，避免在大倾角时奖励几乎为零。
    torso_score = ((torso_up + 1.0) * 0.5).clamp(0.0, 1.0)

    knees = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    # 过度屈膝时不再奖励，避免卡在蹲伏阶段。
    knee_excess = (knees.abs() - 1.50).clamp_min(0.0)

    # 深屈膝约 2.4 rad 时接近零；逐渐伸到 1.5 rad 时达到一。
    # 不继续奖励把膝盖伸直。
    knee_score = (1.0 - knee_excess / 0.90).clamp(0.0, 1.0).mean(dim=-1)

    # ==================== 支撑与离手门控 ====================
    feet_force = torch.stack(
        [
            current_ground_contact_force(env, cfg)
            for cfg in foot_sensor_cfgs
        ],
        dim=-1
    )
    # 要求每一只脚都承受一定的地面接触力，避免卡在蹲伏阶段。
    # 5 N 以下不算支撑，30 N 以上门控为1。
    feet_gate = ((feet_force - 5.0) / 25.0).clamp(0.0, 1.0).amin(dim=-1)

    hand_force = torch.stack(
        [
            current_ground_contact_force(env, cfg)
            for cfg in hand_sensor_cfgs
        ],
        dim=-1
    ).sum(dim=-1)
    # 0.34 m 以下不要求离手，0.50 m 以上完全离手。
    release_gate = ((height - 0.34) / 0.16).clamp(0.0, 1.0)
    release_gate = release_gate.square() * (3.0 - 2.0 * release_gate)
    hand_release_score = torch.exp(-hand_force / 80.0)

    # ==================== 运动质量 ====================
    root_up = (-robot.data.projected_gravity_b.torch[:, 2]).clamp(-1.0, 1.0)

    velocity = robot.data.root_link_vel_w.torch
    horizontal_speed_sq = velocity[:, :2].square().sum(dim=-1)
    angular_speed_sq = velocity[:, 3:].square().sum(dim=-1)

    # 不压制正常向上运动，只评价水平漂移和身体转动。
    motion_score = torch.exp(
        -horizontal_speed_sq / 0.60**2
        -angular_speed_sq / 1.50**2
    )

    # ==================== 基础奖励组合 ====================
    base_score = (
        0.40 * height_score
        + gate * (
            0.30 * torso_score
            + 0.20 * feet_gate * knee_score
        )
        + 0.10 * release_gate * feet_gate * hand_release_score
    )

    # ==================== 转换质量门控 ====================
    # 身体较低时保留原转换引导。
    # 抬高后逐渐要求运动平稳、双脚有接触。
    # 保留非零底值，不禁止正常换支撑。
    support_quality = 0.30 + 0.70 * feet_gate
    motion_quality = 0.25 + 0.75 * motion_score
    transition_quality = (
        1.0 - gate
        + gate * support_quality * motion_quality
    )

    # ==================== 联合奖励 ====================
    # 联合奖励：抬高、姿态改善、双脚接触和低速同时出现。
    # 高度低时几乎没有这项，避免奖励静止趴着。
    root_pose_score = torch.exp(
        -4.0 * (1.0 - root_up)
    )
    torso_pose_score = torch.exp(
        -4.0 * (1.0 - torso_up)
    )
    balanced_score = (
        height_score
        * feet_gate
        * motion_score
        * root_pose_score
        * torso_pose_score
        * hand_release_score
    )

    # ==================== 最终奖励 ====================
    # 分项相加：不要求所有转换条件同时成立才获得奖励。
    return (
        base_score * transition_quality
        + 0.50 * balanced_score
    )

def _getup_completion_scores(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    torso_cfg: SceneEntityCfg,
    foot_sensor_cfgs: tuple[SceneEntityCfg, ...],
    hand_sensor_cfgs: tuple[SceneEntityCfg, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene[asset_cfg.name]

    height = robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]

    root_up = (-robot.data.projected_gravity_b.torch[:, 2]).clamp(-1.0, 1.0)

    torso_quat = robot.data.body_link_pose_w.torch[
        :, torso_cfg.body_ids, 3:7
    ]
    torso_up = (
        1.0 - 2.0 * torso_quat[..., :2].square().sum(dim=-1)
    ).amin(dim=-1).clamp(-1.0, 1.0)

    knees = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]

    feet_force = torch.stack(
        [
            current_ground_contact_force(env, cfg)
            for cfg in foot_sensor_cfgs
        ],
        dim=-1
    )
    both_feet_score = (
        (feet_force - 5.0) / 25.0
    ).clamp(0.0, 1.0).amin(dim=-1)

    hand_force = torch.stack(
        [
            current_ground_contact_force(env, cfg)
            for cfg in hand_sensor_cfgs
        ],
        dim=-1
    ).sum(dim=-1)
    hand_release_score = torch.exp(-hand_force / 80.0)

    transfer_gate = smoothstep((height - 0.30) / 0.15)
    height_score = smoothstep((height - 0.40) / 0.15)
    # 躯干从倾角 80° 到 20° 平滑增长
    torso_start = math.cos(math.radians(80.0))  # 约为 0.1736
    torso_target = math.cos(math.radians(20.0))    # 约为 0.9397
    torso_score = smoothstep((torso_up - torso_start) / (torso_target - torso_start))

    # 骨盆从倾角 70° 到 20° 平滑增长
    root_start = math.cos(math.radians(70.0))   # 约为 0.3420
    root_target = math.cos(math.radians(20.0))      # 约为 0.9397
    root_score = smoothstep((root_up - root_start) / (root_target - root_start))

    # 抬胸到约 75° 时开始加强要求，35° 时完全加强。
    poseture_start = math.cos(math.radians(75.0))  # 约为 0.2588
    poseture_target = math.cos(math.radians(35.0))    # 约为 0.8192
    posture_progress = smoothstep((torso_up - poseture_start) / (poseture_target - poseture_start))

    # 弯曲更多的一条腿决定伸膝得分。
    # 第一版采用 0.8 rad 满分，避免一开始要求过度伸直。
    max_knee = knees.abs().amax(dim=-1)
    knee_score = smoothstep((1.80 - max_knee) / (1.80 - 0.80))

    # 当前姿态也保留少量伸膝反馈，随抬胸逐步加强。
    knee_strength = 0.20 + 0.80 * posture_progress

    # 离手要求随姿态进度增强，并由双脚接触调制。
    # 没有抬胸时，避免鼓励贸然松手。
    release_strength = posture_progress * both_feet_score

    # 过程奖励采用分别相加，避免多个小分数相乘后消失。
    # 双脚接触不再单独贡献一大块固定收益。
    transfer_reward = transfer_gate * (
        0.15 * height_score
        + 0.35 * torso_score
        + 0.15 * root_score
        + 0.20 * knee_strength * knee_score
        + 0.15 * release_strength * hand_release_score
    )

    root_vel = robot.data.root_link_vel_w.torch
    motion_score = torch.exp(
        -root_vel[:, :3].square().sum(dim=-1) / 0.25**2
        -root_vel[:, 3:6].square().sum(dim=-1) / 0.60**2
    )

    # 联合完成奖励仍要求各条件同时改善。
    # 高度从 0.42 m 开启，到 0.55 m 完全开启。
    completion_gate = smoothstep((height - 0.42) / 0.13)

    completion_reward = (
        completion_gate
        * root_score
        * torso_score
        * both_feet_score
        * hand_release_score
        * knee_score
        * motion_score
    )

    return transfer_reward, completion_reward


def getup_support_transfer_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    torso_cfg: SceneEntityCfg,
    foot_sensor_cfgs: tuple[SceneEntityCfg, ...],
    hand_sensor_cfgs: tuple[SceneEntityCfg, ...],
) -> torch.Tensor:
    transfer_reward, _ = _getup_completion_scores(
        env,
        asset_cfg,
        torso_cfg,
        foot_sensor_cfgs,
        hand_sensor_cfgs,
    )
    return transfer_reward

def getup_completion_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    torso_cfg: SceneEntityCfg,
    foot_sensor_cfgs: tuple[SceneEntityCfg, ...],
    hand_sensor_cfgs: tuple[SceneEntityCfg, ...],
) -> torch.Tensor:
    _, completion_reward = _getup_completion_scores(
        env,
        asset_cfg,
        torso_cfg,
        foot_sensor_cfgs,
        hand_sensor_cfgs,
    )
    return completion_reward

def transition_motion_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """抬起后惩罚过大的滚转/俯仰角速度和水平漂移。"""
    robot = env.scene[asset_cfg.name]

    # ==================== 状态量计算 ====================
    pose = robot.data.root_link_pose_w.torch
    velocity = robot.data.root_link_vel_w.torch

    height = pose[:, 2] - env.scene.env_origins[:, 2]

    # ==================== 高度门控 ====================
    height_gate = ((height - 0.25) / 0.15).clamp(0.0, 1.0)
    height_gate = height_gate.square() * (3.0 - 2.0 * height_gate)

    # ==================== 速度超限惩罚 ====================
    # 世界角速度转换到骨盆坐标系。
    q = pose[:, 3:7]
    xyz = q[:, :3]
    w = q[:, 3:4]
    omega_w = velocity[:, 3:]
    cross = 2.0 * torch.cross(xyz, omega_w, dim=-1)
    omega_b = omega_w - w * cross + torch.cross(xyz, cross, dim=-1)

    # 允许一定的正常转换速度，只惩罚超出部分。
    roll_pitch_speed = omega_b[:, :2].norm(dim=-1)
    horizontal_speed = velocity[:, :2].norm(dim=-1)

    angular_excess = (roll_pitch_speed - 1.50).clamp(0.0, 3.0)
    linear_excess = (horizontal_speed - 0.60).clamp(0.0, 3.0)

    # ==================== 最终惩罚 ====================
    # 不乘直立门控：大倾角时也需要抑制快速翻转
    return height_gate * (
        angular_excess.square()
        + 0.50 * linear_excess.square()
    )

def crouch_stability_reward(
    env,
    asset_cfg: SceneEntityCfg,
    torso_cfg: SceneEntityCfg,
    ground_sensor_cfgs: tuple[SceneEntityCfg, ...],
    minimum_height: float = 0.42,
    full_height: float = 0.50,
    target_height: float = 0.53,
    linear_speed_scale: float = 0.25,
    angular_speed_scale: float = 0.60,
    force_threshold: float = 5.0,
):
    """身体抬起后，连续奖励稳定的半蹲高度、姿态、速度和支撑。"""
    robot = env.scene[asset_cfg.name]

    # ==================== 状态量计算 ====================
    height = (robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2])

    # ==================== 高度与姿态门控 ====================
    height_gate = ((height - minimum_height) / (full_height - minimum_height)).clamp(0.0, 1.0)
    # smoothstep 函数：平滑后的值 = x² × (3 - 2x)
    height_gate = height_gate.square() * (3.0 - 2.0 * height_gate)

    root_up = -robot.data.projected_gravity_b.torch[:, 2]
    root_gate = ((root_up - 0.5) / 0.44).clamp(0.0, 1.0)
    root_gate = root_gate.square() * (3.0 - 2.0 * root_gate)

    # ==================== 姿态质量 ====================
    torso_quat = robot.data.body_link_pose_w.torch[
        :, torso_cfg.body_ids, 3:7
    ]
    # 1 - 2 * (x² + y²)算出的是躯干自身向上方向与世界竖直向上方向夹角的余弦
    torso_up = (
        1.0
        - 2.0 * (
            torso_quat[..., 0].square()
            + torso_quat[..., 1].square()
        )
    ).amin(dim=-1)
    # 骨盆和躯干姿态保持连续反馈。
    # 用 1-cos(倾角) 衡量姿态误差，避免 acos 在接近直立时的数值问题。
    root_pose_score = torch.exp(-4.0 * (1.0 - root_up.clamp(-1.0, 1.0)))
    torso_pose_score = torch.exp(-8.0 * (1.0 - torso_up.clamp(-1.0, 1.0)))
    posture_score = 0.5 * (
        root_pose_score + torso_pose_score
    )

    # ==================== 支撑与速度质量 ====================
    # 只在双脚都有地面接触时鼓励稳定。
    # 不要求左右脚承重相等。
    feet_score = torch.stack(
        [
            ground_contact_force(env, cfg) > force_threshold
            for cfg in ground_sensor_cfgs
        ],
        dim=-1
    ).float().mean(dim=-1)

    # 先平方再求和防止正负抵消
    linear_speed_sq = robot.data.root_lin_vel_w.torch.square().sum(dim=-1)
    angular_speed_sq = robot.data.root_ang_vel_w.torch.square().sum(dim=-1)

    # 速度越小，奖励越高；这里不直接采用成功判定的硬阈值。
    speed_score = torch.exp(
        -linear_speed_sq / linear_speed_scale**2
        -angular_speed_sq / angular_speed_scale**2
    )

    # ==================== 高度与膝关节质量 ====================
    height_score = torch.exp(-((height - target_height) / 0.10).square())

    knee_angles = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    knee_excess = (knee_angles.abs() - 1.50).clamp_min(0.0)
    knee_score = torch.exp(
        -knee_excess.square().sum(dim=-1) / 0.25**2
    )

    # ==================== 稳定性奖励组合 ====================
    quality = (
        0.15 * height_score
        + 0.25 * posture_score
        + 0.40 * speed_score
        + 0.10 * feet_score
        + 0.10 * knee_score
    )

    # ==================== 最终奖励 ====================
    return height_gate * root_gate * quality


def head_contact_penalty(
    env,
    force_threshold: float,
    ground_sensor_cfgs: tuple[SceneEntityCfg, ...],
):
    """任一头部刚体发生明显接触时返回 1。"""
    # ==================== 接触检测 ====================
    contact = torch.stack(
        [
            ground_contact_force(env, cfg) > force_threshold
            for cfg in ground_sensor_cfgs
        ],
        dim=-1,
    )

    # ==================== 最终惩罚 ====================
    # 两个头部刚体同时接触也只计一次。
    return contact.any(dim=-1).float()

def success_reward(env):
    """读取本步成功终止标志，不重复更新成功计时。"""
    # ==================== 成功奖励 ====================
    # 将布尔终止标志转换为浮点奖励：成功为 1，否则为 0。
    return env.termination_manager.get_term("success").float()

def out_of_bounds_failure(env):
    """越界时触发一次；补偿奖励管理器的控制步长缩放。"""
    # ==================== 越界失败惩罚 ====================
    return (
        env.termination_manager.get_term("out_of_bounds").float()
        / env.step_dt
    )

def out_of_bounds(
    env,
    max_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """机器人离开自己的实验区域时结束 episode。"""
    robot = env.scene[asset_cfg.name]

    # 根位置减去所属环境原点，得到机器人相对于自己出生区域的位置。
    relative_position = (
        robot.data.root_pos_w.torch
        - env.scene.env_origins
    )

    # 只计算水平面 xy 距离；超过允许半径时返回 True。
    return (
        torch.linalg.vector_norm(
            relative_position[:, :2],
            dim=-1,
        )
        > max_distance
    )


class GetUpSuccess(ManagerTermBase):
    """站立条件连续满足指定时长后，才判定成功。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # 记录每个并行环境已连续满足站立条件的仿真步数。
        self._hold_steps = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
        # 记录每个环境上次执行判定时的 episode 步数，防止同一步重复计数。
        self._last_step = torch.full_like(
            self._hold_steps,
            -1,
        )

    def reset(self, env_ids=None):
        # env_ids=None 时重置全部环境，否则只重置指定环境。
        ids = slice(None) if env_ids is None else env_ids

        self._hold_steps[ids] = 0
        self._last_step[ids] = -1

    def __call__(
        self,
        env,
        minimum_height: float,
        maximum_tilt: float,
        maximum_linear_speed: float,
        maximum_angular_speed: float,
        maximum_knee_angle: float,
        hold_time: float,
        force_threshold: float,
        asset_cfg: SceneEntityCfg,
        ground_sensor_cfgs: tuple[SceneEntityCfg, ...],
        maximum_torso_tilt: float = 0.55,
        torso_cfg: SceneEntityCfg | None = None,
    ):
        robot = env.scene[asset_cfg.name]

        # 每个机器人根节点相对于所属环境原点的高度，形状为 [num_envs]。
        height = (
            robot.data.root_pos_w.torch[:, 2]
            - env.scene.env_origins[:, 2]
        )

        # 机身竖直方向与世界竖直方向夹角小于 maximum_tilt 时视为直立。
        # 两个单位方向向量的点积等于夹角余弦；这里该点积为 -projected_gravity_z。
        upright = (
            -robot.data.projected_gravity_b.torch[:, 2]
            > math.cos(maximum_tilt)
        )

        # 根节点直立不代表胸部也回正，因此单独检查躯干。
        torso_upright = torch.ones_like(upright)

        if torso_cfg is not None:
            torso_quat = robot.data.body_link_pose_w.torch[
                :, torso_cfg.body_ids, 3:7
            ]
            torso_up_cosine = (
                1.0
                - 2.0 * (
                    torso_quat[..., 0].square()
                    + torso_quat[..., 1].square()
                )
            )
            torso_upright = (
                torso_up_cosine > math.cos(maximum_torso_tilt)
            ).all(dim=-1)

        # 根节点三维线速度和三维角速度的 L2 范数，形状均为 [num_envs]。
        linear_speed = torch.linalg.vector_norm(
            robot.data.root_lin_vel_w.torch,
            dim=-1,
        )

        angular_speed = torch.linalg.vector_norm(
            robot.data.root_ang_vel_w.torch,
            dim=-1,
        )

        # asset_cfg 在配置中只选择左右膝关节。
        knee_angles = robot.data.joint_pos.torch[
            :, asset_cfg.joint_ids
        ]

        # 对每个环境取左右膝绝对角度的最大值；两膝都小于阈值才算伸直。
        knees_straight = (
            knee_angles.abs().amax(dim=-1)
            < maximum_knee_angle
        )

        feet_contact = torch.stack(
            [
                ground_contact_force(env, cfg) > force_threshold
                for cfg in ground_sensor_cfgs
            ],
            dim=-1,
        ).all(dim=-1)

        # 六项条件必须同时成立，才认为当前仿真步处于有效站立状态。
        valid = (
            (height > minimum_height)
            & upright
            & torso_upright
            & (linear_speed < maximum_linear_speed)
            & (angular_speed < maximum_angular_speed)
            & knees_straight
            & feet_contact
        )

        # 同一环境步被重复查询时，不重复累计。
        new_step = (
            env.episode_length_buf != self._last_step
        )

        # 有效时计数加一，无效时清零，从而要求站立状态必须连续保持。
        updated = torch.where(
            valid,
            self._hold_steps + 1,
            torch.zeros_like(self._hold_steps),
        )

        # 只在进入新的环境步时写回计数结果：
        # - new_step=True：采用 updated，使有效环境加一、无效环境归零；
        # - new_step=False：保留原计数，避免同一仿真步被多次查询时重复累加。
        # [:] 表示原地更新已有张量，保持管理器内部状态张量本身不变。
        self._hold_steps[:] = torch.where(
            new_step,
            updated,
            self._hold_steps,
        )

        # 保存本次查询对应的 episode 步数。
        self._last_step.copy_(env.episode_length_buf)

        # 将要求保持的秒数换算成仿真步数，向上取整保证时长不少于 hold_time。
        required_steps = math.ceil(
            hold_time / env.step_dt
        )

        # 每个环境独立返回是否已经连续站稳足够长时间。
        return self._hold_steps >= required_steps
 