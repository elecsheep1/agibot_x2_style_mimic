"""把冻结 GMR 参考转换为固定频率 NPZ；不启动 Isaac Lab、不运行物理。"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from x2_motion_utils import joint_indices, load_motion, policy_joint_names


def interpolate_columns(times, source_times, values):
    return np.column_stack([np.interp(times, source_times, column) for column in values.T])


def angular_velocity_world(rotations, dt):
    """世界系相邻旋转增量；内部帧取相邻两个区间速度的均值。"""
    intervals = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / dt
    return np.vstack((intervals[0], (intervals[:-1] + intervals[1:]) / 2, intervals[-1]))


def export_mujoco_fk(path, names, root_pos, root_quat, joints):
    """提供跨模拟器骨架对照；不是把 MuJoCo 接触结果当作训练标签。"""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
        raise ValueError("当前转换要求模型的首个关节是根自由关节")
    if model.body(int(model.jnt_bodyid[0])).name != "pelvis" or model.nq != 7 + len(names):
        raise ValueError("模型根或关节数量与当前 X2 参考约定不一致")
    addresses, body_ids = [], [int(model.jnt_bodyid[0])]
    for name in names:
        joint = model.joint(name)
        if model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"不支持的参考关节：{name}")
        addresses.append(int(model.jnt_qposadr[joint.id]))
        body_ids.append(int(model.jnt_bodyid[joint.id]))
    if len(set(body_ids)) != len(body_ids):
        raise ValueError("当前 X2 FK 要求每个驱动关节对应独立刚体")
    limits = np.array([model.jnt_range[model.joint(name).id] for name in names])
    if np.any(joints < limits[:, 0] - 1e-6) or np.any(joints > limits[:, 1] + 1e-6):
        raise ValueError("插值后的参考角度超出模型硬限位")
    positions, quaternions = [], []
    for pos, quat, joint_pos in zip(root_pos, root_quat, joints):
        data.qpos[:3], data.qpos[3:7], data.qpos[addresses] = pos, quat[[3, 0, 1, 2]], joint_pos
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[body_ids].copy())
        quaternions.append(data.xquat[body_ids][:, [1, 2, 3, 0]].copy())
    return {
        "body_names": np.array([model.body(i).name for i in body_ids]),
        "body_pos_w": np.asarray(positions, dtype=np.float32),
        "body_quat_xyzw": np.asarray(quaternions, dtype=np.float32),
        "joint_hard_limits": limits.astype(np.float32),
    }


def convert_reference(source, target_fps, policy_names, model_xml):
    """线性插值位置/关节角，SLERP 插值根旋转，再在目标时间轴计算速度。"""
    source = Path(source).resolve()
    with source.open("rb") as file:
        raw = pickle.load(file)
    if raw.get("quat_order") != "xyzw" or int(raw.get("start_frame_30hz", 0)) != 0:
        raise ValueError("本阶段只接收从第零帧开始、明确标注 xyzw 的冻结参考")
    fps = float(raw["fps"])
    if not np.isfinite([fps, target_fps]).all() or fps <= 0 or target_fps <= 0:
        raise ValueError("源帧率和目标帧率必须为有限正数")
    names = list(raw["joint_names"])
    policy_ids = joint_indices(names, policy_names)
    if set(names) - set(policy_names) != {"head_yaw_joint", "head_pitch_joint"}:
        raise ValueError("策略外关节应恰好为头部 yaw/pitch")
    joints, root_pos, root_quat = [np.asarray(raw[key], dtype=float) for key in ("dof_pos", "root_pos", "root_rot")]
    count = len(joints)
    if (count < 3 or joints.shape != (count, len(names)) or root_pos.shape != (count, 3)
            or root_quat.shape != (count, 4) or not all(np.isfinite(x).all() for x in (joints, root_pos, root_quat))):
        raise ValueError("源数据形状或数值不合法")
    norms = np.linalg.norm(root_quat, axis=1)
    if not np.allclose(norms, 1, atol=1e-5):
        raise ValueError("源根四元数未归一化")
    source_times = np.arange(count) / fps
    dt = 1 / target_fps
    # 不在末尾补静止帧；末帧不足一个固定步长时舍去尾部不足 dt 的区间。
    times = np.arange(int(np.floor(source_times[-1] * target_fps + 1e-9)) + 1) * dt
    if len(times) < 3:
        raise ValueError("目标时间轴至少需要三帧")
    # 固定步长时间轴的末点可能因浮点舍入略超源终点；只钳制插值查询，
    # 保留输出的均匀时间轴，不外推或额外补帧。
    query_times = np.minimum(times, source_times[-1])
    rotations = Slerp(source_times, Rotation.from_quat(root_quat / norms[:, None]))(query_times)
    quats = rotations.as_quat()
    for i in range(1, len(quats)):
        if np.dot(quats[i-1], quats[i]) < 0:
            quats[i] *= -1
    positions = interpolate_columns(times, source_times, root_pos)
    joint_pos = interpolate_columns(times, source_times, joints)
    joint_vel = np.gradient(joint_pos, dt, axis=0, edge_order=1)
    linear = np.gradient(positions, dt, axis=0, edge_order=1)
    angular = angular_velocity_world(rotations, dt)
    gravity = rotations.inv().apply(np.tile([0, 0, -1.], (len(times), 1)))
    metadata = {
        "schema_version": 1, "status": "amp_reference_candidate_not_final_discriminator_schema",
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_fps": fps, "source_frames": count, "source_duration_s": float(source_times[-1]),
        "fps": target_fps, "dt_s": dt, "duration_s": float(times[-1]), "quat_order": "xyzw",
        "root_frame": "pelvis", "isaac_root_frame": "base_link", "up_axis": "Z",
        "root_transform": "identity; verify with cross-simulator FK",
        "position_units": "m", "joint_units": "rad", "angular_velocity_units": "rad/s",
        "velocity_method": "finite differences on resampled positions; world relative rotation increments",
        "interpolation": "linear joint/root positions; shortest-path SLERP root rotation",
        "ground_z": float(raw.get("ground_z", 0)), "loop": False,
        "mujoco_xml": str(model_xml) if model_xml else None,
    }
    result = {
        "time_s": times, "joint_names": np.array(names), "policy_joint_names": np.array(policy_names),
        "policy_joint_indices": policy_ids, "root_pos_w": positions, "root_quat_xyzw": quats,
        "root_lin_vel_w": linear, "root_ang_vel_w": angular,
        "root_lin_vel_b": rotations.inv().apply(linear), "root_ang_vel_b": rotations.inv().apply(angular),
        "projected_gravity_b": gravity, "joint_pos": joint_pos, "joint_vel": joint_vel,
        "policy_joint_pos": joint_pos[:, policy_ids], "policy_joint_vel": joint_vel[:, policy_ids],
    }
    if model_xml:
        result.update(export_mujoco_fk(model_xml, names, positions, quats, joint_pos))
    for key, value in result.items():
        if value.dtype.kind == "f" and key != "time_s":
            result[key] = value.astype(np.float32)
    result["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=50.)
    parser.add_argument("--mujoco_xml", type=Path, help="冻结标定 XML，用于导出跨模拟器 FK 对照。")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("输出已存在；若确实要替换，请显式使用 --overwrite")
    if args.output.suffix != ".npz":
        parser.error("输出扩展名应为 .npz")
    model_xml = args.mujoco_xml
    if model_xml is None:
        candidate = args.input.resolve().parent / "calibration/x2_reference_v3_segments/x2_reference.xml"
        if candidate.is_file():
            model_xml = candidate
    if model_xml is None:
        parser.error("未找到冻结标定 XML，请用 --mujoco_xml 明确指定")
    result = convert_reference(args.input, args.fps, policy_joint_names(), model_xml.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    motion = load_motion(args.output)
    report = {
        "metadata": motion["metadata"], "frames": len(motion["time_s"]),
        "joint_names": motion["joint_names"].tolist(), "policy_joint_names": motion["policy_joint_names"].tolist(),
        "policy_to_reference_indices": motion["policy_joint_indices"].tolist(),
        "joint_speed_max_rad_s": dict(zip(motion["joint_names"].tolist(), np.max(abs(motion["joint_vel"]), axis=0).tolist())),
        "body_names": motion["body_names"].tolist(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"[REFERENCE] {len(motion['time_s'])} frames, {args.fps:g} Hz, duration {motion['time_s'][-1]:.3f}s")
    print("[REFERENCE] all joints=31, policy joints=29; mapping by name; root quaternion=xyzw")
    print(f"[SAVED] {args.output.resolve()}")


if __name__ == "__main__":
    main()
