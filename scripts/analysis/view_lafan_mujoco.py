#!/usr/bin/env python3
"""在 MuJoCo 里回放 LAFAN getup 的 X2 重定向动作。

用法:
    python scripts/analysis/view_lafan_mujoco.py \
        --mjcf /home/panda/robotics_ws/retarget/GMR/assets/agibot_x2/lafan_subject2_prone_getup_umr.floating_mjcf.xml \
        --motion data/motions/lafan1_getup_umr/x2_lafan_subject2_prone_getup_umr_50hz.npz

说明:
- MJCF 是 floating base 的 X2 模型, qpos = [root_pos(3), root_quat_wxyz(4), joints(31)]
- npz 的 root_quat 是 xyzw, 这里转成 wxyz
- 关节按名字映射, 不依赖排列
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


DEFAULT_MJCF = (
    "/home/panda/robotics_ws/retarget/GMR/assets/agibot_x2/"
    "lafan_subject2_prone_getup_umr.floating_mjcf.xml"
)
DEFAULT_MOTION = (
    "data/motions/lafan1_getup_umr/"
    "x2_lafan_subject2_prone_getup_umr_50hz.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", default=DEFAULT_MJCF, help="X2 floating-base MJCF")
    parser.add_argument("--motion", default=DEFAULT_MOTION, help="retargeted .npz")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed")
    parser.add_argument("--loop", action="store_true", default=True, help="loop the motion")
    parser.add_argument("--no-loop", dest="loop", action="store_false", help="stop at the end")
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="start time in seconds",
    )
    return parser.parse_args()


def load_motion(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    required = ("time_s", "joint_names", "root_pos_w", "root_quat_xyzw", "joint_pos")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"motion missing keys: {missing}")
    return {key: data[key] for key in required}


def resolve_joint_qpos_adr(model: mujoco.MjModel, joint_names) -> np.ndarray:
    addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, str(name)
        )
        if joint_id < 0:
            raise KeyError(f"MJCF has no joint named {name!r}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=int)


def main() -> None:
    args = parse_args()
    mjcf_path = Path(args.mjcf)
    motion_path = Path(args.motion)
    if not mjcf_path.is_file():
        raise FileNotFoundError(mjcf_path)
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    motion = load_motion(str(motion_path))

    times = np.asarray(motion["time_s"], dtype=float)
    joint_names = [str(name) for name in motion["joint_names"]]
    joint_adr = resolve_joint_qpos_adr(model, joint_names)

    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError("expected the first joint to be a free joint")
    if model.nq != 7 + len(joint_names):
        raise RuntimeError(
            f"qpos size mismatch: model nq={model.nq}, "
            f"expected 7+{len(joint_names)}"
        )

    root_pos = np.asarray(motion["root_pos_w"], dtype=float)
    root_quat_xyzw = np.asarray(motion["root_quat_xyzw"], dtype=float)
    joint_pos = np.asarray(motion["joint_pos"], dtype=float)

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.02
    print(
        f"[viewer] model={mjcf_path.name} motion={motion_path.name} "
        f"frames={len(times)} duration={times[-1] - times[0]:.2f}s "
        f"dt={dt:.4f}s"
    )
    print(f"[viewer] joints={len(joint_names)} speed={args.speed} loop={args.loop}")

    start = float(np.clip(args.start, times[0], times[-1]))
    frame = int(np.searchsorted(times, start, side="right") - 1)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = data.qpos[:3]
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 90.0
        wall_start = time.perf_counter()
        while viewer.is_running():
            elapsed = start + (time.perf_counter() - wall_start) * args.speed
            if elapsed > times[-1]:
                if not args.loop:
                    break
                start = times[0]
                wall_start = time.perf_counter()
                elapsed = start
            frame = int(np.searchsorted(times, elapsed, side="right") - 1)
            frame = min(max(frame, 0), len(times) - 1)

            data.qpos[0:3] = root_pos[frame]
            qx, qy, qz, qw = root_quat_xyzw[frame]
            data.qpos[3:7] = (qw, qx, qy, qz)
            data.qpos[joint_adr] = joint_pos[frame]
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = data.qpos[:3]

            viewer.sync()
            time.sleep(min(dt / max(args.speed, 1e-6), 0.02))

    print("[viewer] closed")


if __name__ == "__main__":
    main()
