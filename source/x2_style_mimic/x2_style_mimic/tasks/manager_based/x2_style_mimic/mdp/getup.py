from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .motion_dataset import MotionDataset
from .getup_births import FallenStateBank
from x2_style_mimic.assets.x2_ultra_cfg import X2_ULTRA_USD

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
    manifest_file,                          # 动作数据集清单文件路径
    policy_joint_names,                     # 策略控制的关节名称列表
    start_phase=0.0,
    end_phase=0.0,
    use_reference_velocity=False,           # 是否使用参考速度
    zero_initial_frame_velocity=False,      # 是否将初始帧的速度置零
    height_offset=0.01,                     # 高度偏移
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    initial_frame_probability=0.5,
    transition_frame_ranges=None,
    perturb_initial_probability=0.0,        # 初始状态扰动概率
    perturb_state_bank=None,                # 初始状态扰动库
):
    """按权重选择动作；默认从首帧、零速度出生。"""
    if not math.isfinite(height_offset) or height_offset < 0.0:
        raise ValueError("height_offset 必须为非负有限数")
    if not (math.isfinite(perturb_initial_probability) and 0.0 <= perturb_initial_probability <= 1.0):
        raise ValueError("perturb_initial_probability 必须是 [0, 1] 内的有限数")
    if perturb_initial_probability > 0.0 and perturb_state_bank is None:
        raise ValueError("启用扰动出生前必须提供已验证的扰动状态库")

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
    initial = state["initial_branch"]
    perturbed = torch.zeros_like(initial)
    if perturb_initial_probability > 0.0:
        perturbed = initial & (
            torch.rand(len(env_ids), device=env.device)
            < perturb_initial_probability
        )
    # 0：原始首帧。
    # 1：扰动首帧。
    # 2：参考过渡帧。
    birth_types = torch.full(
        (len(env_ids),),
        2,
        dtype=torch.long,
        device=env.device,
    )
    birth_types[initial] = 0
    birth_types[perturbed] = 1

    root_pose = state["root_pose"].clone()
    root_velocity = state["root_velocity"].clone()
    joint_pos = state["joint_pos"].clone()
    joint_vel = state["joint_vel"].clone()

    motion_ids = state["motion_ids"].clone()
    frame_ids = state["frame_ids"].clone()
    bank_row_ids = torch.full_like(motion_ids, -1)

    # 先处理参考状态速度。
    if not use_reference_velocity:
        root_velocity.zero_()
        joint_vel.zero_()
    elif zero_initial_frame_velocity:
        initial_mask = frame_ids == 0
        root_velocity[initial_mask] = 0.0
        joint_vel[initial_mask] = 0.0

    if bool(perturbed.any()):
        # 加载、抽样并替换已验证的扰动状态。
        bank_key = (
            str(Path(perturb_state_bank).resolve()),
            tuple(robot.joint_names),
            tuple(str(path) for path in dataset.paths),
            str(env.device),
        )
        cache = getattr(env, "_getup_fallen_bank_cache", None)
        if cache is None or cache[0] != bank_key:
            bank = FallenStateBank(
                path=perturb_state_bank,
                robot_joint_names=robot.joint_names,
                device=env.device,
                motion_paths=dataset.paths,
                asset_root=X2_ULTRA_USD.parent,
            )
            env._getup_fallen_bank_cache = (bank_key, bank)
        else:
            bank = cache[1]

        # 库分支内部：近邻和仿真摔倒各 50%。
        source_types = torch.randint(
            0, 2,
            (int(perturbed.sum().item()),),
            device=env.device,
        )
        sampled = bank.sample(
            motion_ids[perturbed], source_types
        )

        root_pose[perturbed] = sampled["root_pose"]
        root_velocity[perturbed] = sampled["root_velocity"]
        joint_pos[perturbed] = sampled["joint_pos"]
        joint_vel[perturbed] = sampled["joint_vel"]
        # 采集时的水平位移不作为出生位置。
        # 保留根高度、朝向、关节姿态和全部速度。
        root_pose[perturbed, :2] = 0.0
        birth_types[perturbed] = torch.where(
            source_types == 0, 1, 3
        )
        motion_ids[perturbed] = sampled["motion_id"]
        frame_ids[perturbed] = -1
        bank_row_ids[perturbed] = sampled["bank_row"]

    # 库中姿态已经落定，不再次加高。
    root_pose[~perturbed, 2] += height_offset
    root_pose[:, :3] += env.scene.env_origins[env_ids]

    # 与当前仿真模型的硬限位对齐，并记录裁剪数量。
    limits = robot.data.joint_pos_limits.torch[env_ids]
    clamped = joint_pos.clamp(
        min=limits[:, :, 0],
        max=limits[:, :, 1],
    )
    clamp_error = (clamped - joint_pos).abs().amax(dim=-1)
    # 允许小幅求解器限位偏差，明显越界仍拒绝。
    bank_limit_tolerance = 0.003
    if bool((perturbed & (clamp_error > bank_limit_tolerance)).any()):
        raise ValueError("状态库关节位置明显超限")
    # 计算每个环境中被裁剪的关节数量，判断裁剪前后关节位置的绝对差值是否大于 1e-6，然后对每个环境求和得到裁剪数量。
    clamp_count = ((clamped - joint_pos).abs() > 1e-6).sum(dim=-1)
    joint_pos = torch.where(
        perturbed[:, None],
        joint_pos,
        clamped,
    )

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
        "_getup_birth_types",
        "_getup_bank_row_ids",
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
    env._getup_reference_frame_ids[env_ids] = frame_ids
    env._getup_reference_motion_ids[env_ids] = motion_ids
    env._getup_reference_joint_clamp_counts[env_ids] = clamp_count
    env._getup_reference_motion_paths = [str(path) for path in dataset.paths]
    env._getup_birth_types[env_ids] = birth_types
    env._getup_bank_row_ids[env_ids] = bank_row_ids

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
    limb_quality=None,
    limb_fraction=0.10,
):
    """有界站立质量，输出范围约为 [0, 1]。"""

    # ------------------------------- 高度指标 ------------------------------
    # 从低位到站立高度提供连续反馈。
    height_progress = (
        (height - 0.08) / (0.63 - 0.08)     # 高度范围从 0.08 到 0.63 米
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
        (torch.minimum(root_up, torso_up) - 0.50) / (math.cos(math.radians(10.0)) - 0.50)
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
    # ---------------------------- 站立分数 ----------------------------
    standing_score = (
        0.65 * stability_quality
        + 0.35 * support_quality
    )
    # ---------------------------- 手脚质量因子 ----------------------------
    if not 0.0 <= limb_fraction <= 1.0:
        raise ValueError("limb_fraction 必须在 [0, 1] 内")
    # 未配置手脚评分时，保持原有奖励行为。
    if limb_quality is None:
        limb_quality = torch.ones_like(height)
    # 手脚姿态只影响现有站立分支，不重复奖励高度或稳定性。
    limb_factor = (
        1.0 - limb_fraction
        + limb_fraction * limb_quality.clamp(0.0, 1.0)
    )

    return (
        0.35 * geometry_quality
        + 0.65
        * near_standing
        * upright_gate
        * standing_score
        * limb_factor
    )

def standing_quality_reward(
    env,
    torso_cfg: SceneEntityCfg,
    ground_sensor_cfgs: tuple[SceneEntityCfg, ...],
    wrist_sensor_cfgs: tuple[SceneEntityCfg, ...],
    limb_joint_cfg: SceneEntityCfg | None = None,
    feet_cfg: SceneEntityCfg | None = None,
    limb_joint_targets: tuple[float, ...] | None = None,
    stance_width: float = 0.34,
    toe_out_rad: float = 0.08726646,
    limb_fraction: float = 0.10,
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

    limb_quality = None

    if limb_joint_cfg is not None:
        limb_quality = standing_limb_quality(
            env,
            joint_cfg=limb_joint_cfg,
            feet_cfg=feet_cfg,
            joint_targets=limb_joint_targets,
            stance_width=stance_width,
            toe_out_rad=toe_out_rad,
        )


    standing_reward = standing_quality(
        height=height,
        root_up=root_up,
        torso_up=torso_up,
        linear_velocity=root_velocity[:, :3],
        angular_velocity=root_velocity[:, 3:6],
        feet_support=feet_support,
        wrist_load=wrist_load,
        limb_quality=limb_quality,
        limb_fraction=limb_fraction
    )
    return standing_reward


def standing_limb_quality(
    env: ManagerBasedRLEnv,
    joint_cfg: SceneEntityCfg,
    feet_cfg: SceneEntityCfg,
    joint_targets: tuple[float, ...],
    stance_width: float = 0.34,
    toe_out_rad: float = 0.08726646,     # 左右各外展 5°
) -> torch.Tensor:
    robot = env.scene["robot"]

    # ---------- 肩、肘、腕 ----------
    q = robot.data.joint_pos.torch[:, joint_cfg.joint_ids]
    target = q.new_tensor(joint_targets)

    if target.numel() != q.shape[1]:
        raise ValueError("站姿目标角数量与选中的关节数量不一致")

    # 允许约 5.7° 偏差，超出后再降低评分。
    joint_error = (q - target).abs().sub(0.10).clamp_min(0.0)
    arm_quality = torch.exp(-(joint_error / 0.30).square().mean(dim=-1))

    # ---------- 骨盆朝向 ----------
    # 当前项目四元数顺序为 XYZW。
    root_q = robot.data.root_link_pose_w.torch[:, 3:7]
    root_q = root_q / root_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    x, y, z, w = root_q.unbind(dim=-1)
    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )
    c, s = yaw.cos(), yaw.sin()

    # feet_cfg 必须保持左脚、右脚的顺序。
    feet = robot.data.body_link_pose_w.torch[:, feet_cfg.body_ids]
    if feet.shape[1] != 2:
        raise ValueError("feet_cfg 必须按左、右顺序匹配两个踝刚体")

    # ---------- 双脚站宽和前后错位 ----------
    delta = feet[:, 0, :3] - feet[:, 1, :3]
    delta_x = c * delta[:, 0] + s * delta[:, 1]
    delta_y = -s * delta[:, 0] + c * delta[:, 1]

    # 目标站宽允许 ±4 cm；前后错位允许 ±4 cm。
    width_error = ((delta_y - stance_width).abs() - 0.04).clamp_min(0.0)
    stagger_error = (delta_x.abs() - 0.04).clamp_min(0.0)

    placement_quality = torch.exp(
        -(width_error / 0.08).square()
        -(stagger_error / 0.08).square()
    )

    # ---------- 脚尖方向 ----------
    foot_q = feet[:, :, 3:7]
    foot_q = foot_q / foot_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    fx, fy, fz, fw = foot_q.unbind(dim=-1)

    # 踝刚体局部 +X 轴在世界水平面上的方向。
    forward_x = 1.0 - 2.0 * (fy.square() + fz.square())
    forward_y = 2.0 * (fx * fy + fw * fz)

    foot_yaw = torch.atan2(
        -s[:, None] * forward_x + c[:, None] * forward_y,
        c[:, None] * forward_x + s[:, None] * forward_y,
    )

    target_foot_yaw = foot_yaw.new_tensor(
        [toe_out_rad, -toe_out_rad]
    )
    foot_yaw_delta = foot_yaw - target_foot_yaw
    foot_yaw_error = torch.atan2(
        foot_yaw_delta.sin(), foot_yaw_delta.cos()
    ).abs()

    # 允许目标方向附近 ±5°。
    foot_yaw_error = (foot_yaw_error - 0.08726646).clamp_min(0.0)
    direction_quality = torch.exp(
        -(foot_yaw_error / 0.17453293).square().mean(dim=-1)
    )

    return (
        0.50 * arm_quality
        + 0.30 * placement_quality
        + 0.20 * direction_quality
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
    """站立条件连续满足指定时长后，才判定起身成功。

    该管理器项返回一个形状为 ``[num_envs]`` 的布尔张量：每个元素对应
    一个并行环境是否已经成功起身。成功不是某一帧瞬间满足条件，而是要求
    机器人在连续多个仿真步内同时满足高度、姿态、速度、膝关节和足部接触
    等条件。这样可以避免机器人短暂弹起、倾倒过程中经过直立姿态，或尚未
    稳定时被误判为起身成功。
    """

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

        # 七项条件必须同时成立，当前仿真步才算“有效站立”：
        # 1. 根节点高度超过 minimum_height；
        # 2. 机器人整体倾斜角小于 maximum_tilt；
        # 3. 躯干倾斜角小于 maximum_torso_tilt（未配置 torso_cfg 时默认通过）；
        # 4. 根节点线速度小于 maximum_linear_speed；
        # 5. 根节点角速度小于 maximum_angular_speed；
        # 6. 左右膝关节角度均小于 maximum_knee_angle；
        # 7. 所有配置的足部接触传感器都检测到足够大的接触力。
        #
        # 只有 valid 连续保持 hold_time 秒，最终才会判定为起身成功。
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
