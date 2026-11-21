"""Slow an exported X2 reference while preserving its geometric path.

The output remains at the task control frequency.  Positions are interpolated
on a longer time axis, while all velocity and FK fields are regenerated rather
than scaled in-place.  The source is never overwritten.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from common.x2_motion_utils import load_motion
from retarget.convert_x2_reference import angular_velocity_world, export_mujoco_fk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-scale", type=float, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        parser.error("output must be a new file")
    if not np.isfinite(args.time_scale) or args.time_scale <= 1.0:
        parser.error("time-scale must be finite and greater than one")

    source = load_motion(args.input)
    metadata = dict(source.pop("metadata"))
    dt = float(metadata["dt_s"])
    old_time = np.asarray(source["time_s"], dtype=float)
    duration = float(old_time[-1] * args.time_scale)
    # Preserve the fixed 50 Hz contract and include the retimed endpoint.
    new_time = np.arange(round(duration / dt) + 1, dtype=float) * dt
    query = np.minimum(new_time / args.time_scale, old_time[-1])

    # Linear interpolation cannot overshoot the source joint coordinates.  A
    # higher-order spline produced new inter-frame self-penetration on tightly
    # folded recovery poses even though every source frame was valid.
    joint_pos = np.column_stack([
        np.interp(query, old_time, column) for column in source["joint_pos"].T
    ])
    root_pos = np.column_stack([
        np.interp(query, old_time, column) for column in source["root_pos_w"].T
    ])
    rotations = Slerp(old_time, Rotation.from_quat(source["root_quat_xyzw"]))(query)
    root_quat = rotations.as_quat()
    for index in range(1, len(root_quat)):
        if root_quat[index - 1] @ root_quat[index] < 0:
            root_quat[index] *= -1

    joint_vel = np.gradient(joint_pos, dt, axis=0, edge_order=1)
    root_lin_vel_w = np.gradient(root_pos, dt, axis=0, edge_order=1)
    root_ang_vel_w = angular_velocity_world(rotations, dt)
    policy_ids = source["policy_joint_indices"]
    result = {
        "time_s": new_time,
        "joint_names": source["joint_names"],
        "policy_joint_names": source["policy_joint_names"],
        "policy_joint_indices": policy_ids,
        "root_pos_w": root_pos,
        "root_quat_xyzw": root_quat,
        "root_lin_vel_w": root_lin_vel_w,
        "root_ang_vel_w": root_ang_vel_w,
        "root_lin_vel_b": rotations.inv().apply(root_lin_vel_w),
        "root_ang_vel_b": rotations.inv().apply(root_ang_vel_w),
        "projected_gravity_b": rotations.inv().apply(
            np.tile([0.0, 0.0, -1.0], (len(new_time), 1))
        ),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "policy_joint_pos": joint_pos[:, policy_ids],
        "policy_joint_vel": joint_vel[:, policy_ids],
    }
    result.update(export_mujoco_fk(
        Path(metadata["mujoco_xml"]), source["joint_names"].tolist(),
        root_pos, root_quat, joint_pos,
    ))
    metadata.update(
        status="time_retimed_reference_candidate_not_dynamic_validated",
        retime_source=str(args.input.resolve()),
        retime_source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        retime_scale=float(args.time_scale),
        source_duration_before_retime_s=float(old_time[-1]),
        duration_s=float(new_time[-1]),
        interpolation="non-overshooting linear positions/joints; shortest-path SLERP root; regenerated velocities and FK",
    )
    for key, value in result.items():
        if isinstance(value, np.ndarray) and value.dtype.kind == "f" and key != "time_s":
            result[key] = value.astype(np.float32)
    result["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    load_motion(args.output)
    print(json.dumps({
        "input_frames": len(old_time), "output_frames": len(new_time),
        "input_duration_s": float(old_time[-1]), "output_duration_s": float(new_time[-1]),
        "time_scale": args.time_scale,
    }, indent=2))


if __name__ == "__main__":
    main()
