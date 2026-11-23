"""真实与想象起身共用的纯张量奖励；不读取仿真，不关闭梯度。"""
import math
import torch


def smoothstep(x):
    return x.square() * (3.0 - 2.0 * x)


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
    limb_gate = (stability_quality * feet_support).clamp(0.0, 1.0)
    limb_factor = 1.0 - limb_fraction * limb_gate * (
        1.0 - limb_quality.clamp(0.0, 1.0)
    )

    return (
        0.35 * geometry_quality
        + 0.65
        * near_standing
        * upright_gate
        * standing_score
        * limb_factor
    )


def standing_limb_quality_tensor(
    q, root_q, feet, joint_targets,
    stance_width=0.34, toe_out_rad=0.08726646,
):
    """feet 为按左、右排列的 [B, 2, 7]，四元数采用 XYZW。"""
    # ---------- 肩、肘、腕 ----------
    target = q.new_tensor(joint_targets)

    if target.numel() != q.shape[1]:
        raise ValueError("站姿目标角数量与选中的关节数量不一致")

    # 允许约 5.7° 偏差，超出后再降低评分。
    joint_error = (q - target).abs().sub(0.10).clamp_min(0.0)
    arm_quality = torch.exp(-(joint_error / 0.30).square().mean(dim=-1))

    # ---------- 骨盆朝向 ----------
    # 当前项目四元数顺序为 XYZW。
    root_q = root_q / root_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    x, y, z, w = root_q.unbind(dim=-1)
    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )
    c, s = yaw.cos(), yaw.sin()

    # feet_cfg 必须保持左脚、右脚的顺序。
    if feet.shape[1] != 2:
        raise ValueError("feet_cfg 必须按左、右顺序匹配两个踝刚体")

    # ---------- 双脚站宽和前后错位 ----------
    delta = feet[:, 0, :3] - feet[:, 1, :3]
    delta_x = c * delta[:, 0] + s * delta[:, 1]
    delta_y = -s * delta[:, 0] + c * delta[:, 1]

    # 目标站宽允许 ±4 cm；前后错位允许 ±4 cm。
    width_error = ((delta_y - stance_width).abs() - 0.04).clamp_min(0.0)
    stagger_error = (delta_x.abs() - 0.04).clamp_min(0.0)

    placement_quality = 1.0 / (
        1.0
        + (width_error / 0.12).square()
        + (stagger_error / 0.12).square()
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
        0.40 * arm_quality
        + 0.45 * placement_quality
        + 0.15 * direction_quality
    )
