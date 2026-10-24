from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .motion_dataset import MotionDataset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def smoothstep(x):
    """平滑阶跃函数，输出范围为 [0, 1]。"""
    return x.square() * (3.0 - 2.0 * x)

# 重置
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

# 接触力
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

# 任务奖励与惩罚
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

def out_of_bounds_failure(env):
    """越界时触发一次；补偿奖励管理器的控制步长缩放。"""
    # ==================== 越界失败惩罚 ====================
    return (
        env.termination_manager.get_term("out_of_bounds").float()
        / env.step_dt
    )

def standing_quality(
    height,
    root_up,
    torso_up,
    linear_velocity,
    angular_velocity,
    feet_support,
    wrist_load,
):
    """有界站立质量，输出范围约为 [0, 1]。"""

    # ------------------------------- 高度指标 ------------------------------
    # 从低位到站立高度提供连续反馈。
    height_progress = (
        (height - 0.08) / (0.63 - 0.08)
    ).clamp(0.0, 1.0)

    # ------------------------------- 姿态指标 ------------------------------
    root_quality = (root_up.clamp(-1.0, 1.0) + 1.0) * 0.5
    torso_quality = (torso_up.clamp(-1.0, 1.0) + 1.0) * 0.5
    posture_quality = 0.5 * (root_quality + torso_quality)

    geometry_quality = height_progress * (0.25 + 0.75 * posture_quality)

    # ----------------------------- 接近站立门控 -----------------------------
    # 站立质量逐渐接管，前期不要求手部卸载或完全静止。
    near_standing = (
        (height - 0.48) / (0.60 - 0.48)
    ).clamp(0.0, 1.0)
    near_standing = smoothstep(near_standing)
    # ----------------------------- 直立门控 -----------------------------
    upright_gate = (
        (torch.minimum(root_up, torso_up) - 0.50) / 0.44
    ).clamp(0.0, 1.0)
    # ---------------------------- 速度稳定性指标 ----------------------------
    stability_quality = torch.exp(
        -linear_velocity.square().sum(dim=-1) / 0.25**2
        -angular_velocity.square().sum(dim=-1) / 0.60**2
    )
    # ---------------------------- 支撑质量指标 ----------------------------
    # 双脚接触不要求左右承重完全相等。
    # 接近站立时，手部持续承重会降低支撑质量。
    support_quality = feet_support * torch.exp(
        -wrist_load / 20.0
    )

    return (
        0.35 * geometry_quality
        + 0.65
        * near_standing
        * upright_gate
        * (
            0.65 * stability_quality
            + 0.35 * support_quality
        )
    )

def standing_quality_reward(
    env,
    torso_cfg: SceneEntityCfg,
    ground_sensor_cfgs: tuple[SceneEntityCfg, ...],
    wrist_sensor_cfgs: tuple[SceneEntityCfg, ...],
):
    robot = env.scene["robot"]

    height = robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]

    root_up = -robot.data.projected_gravity_b.torch[:, 2]

    torso_quaternion = robot.data.body_link_pose_w.torch[:, torso_cfg.body_ids, 3:7]
    torso_quaternion = torso_quaternion / (
        torso_quaternion.norm(dim=-1, keepdim=True)
        .clamp_min(1e-8)
    )

    torso_up = (
        1.0
        - 2.0
        * (torso_quaternion[..., 0].square() + torso_quaternion[..., 1].square())
    ).amin(dim=-1)

    feet_support = torch.stack(
        [
            ground_contact_force(env, cfg) > 5.0
            for cfg in ground_sensor_cfgs
        ],
        dim=-1,
    ).float().mean(dim=-1)

    wrist_load = torch.stack(
        [
            ground_contact_force(env, cfg)
            for cfg in wrist_sensor_cfgs
        ],
        dim=-1,
    ).sum(dim=-1)

    root_velocity = robot.data.root_link_vel_w.torch

    return standing_quality(
        height=height,
        root_up=root_up,
        torso_up=torso_up,
        linear_velocity=root_velocity[:, :3],
        angular_velocity=root_velocity[:, 3:6],
        feet_support=feet_support,
        wrist_load=wrist_load,
    )

# 终止与成功条件
def out_of_bounds(
    env,
    max_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """机器人离开自己的实验区域时结束 episode。"""
    robot = env.scene[asset_cfg.name]

    # 根位置减去所属环境原点，得到机器人相对于自己出生区域的位置。
    relative_position = robot.data.root_pos_w.torch - env.scene.env_origins

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
