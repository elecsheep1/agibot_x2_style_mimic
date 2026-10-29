"""Isaac Lab 参考姿态回放与跨模拟器 FK 检查，不运行策略或动力学跟踪。"""

# Support direct execution: python scripts/<category>/<tool>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from types import SimpleNamespace
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

from common.x2_motion_utils import (
    joint_indices,
    load_motion,
    policy_joint_names,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--task", default="Template-X2-Style-Mimic-v0", help="只复用资产与物理配置，不创建 RL 环境。")
    parser.add_argument("--slowdown", type=float, default=1.)
    parser.add_argument("--loop", action="store_true", help="人工查看时重复片段；循环边界不作为训练转移。")
    parser.add_argument("--max_frames", type=int, default=None, help="非可视化检查时限制帧数。")
    parser.add_argument("--report", type=Path, default=Path("logs/reference_replay/x2_getup_v0.json"))
    parser.add_argument("--fk_tolerance", type=float, default=.005, help="跨模拟器最大刚体位置误差阈值，m。")
    parser.add_argument("--orientation_tolerance_deg", type=float, default=.5, help="跨模拟器刚体方向误差阈值，度。")
    parser.add_argument("--amp_tolerance", type=float, default=1e-3,
                        help="Maximum absolute error across all scaled AMP features; body velocities use reference finite differences.")
    # AppLauncher inspects the parser before the final CLI parse.
    # Defer required custom arguments only during launcher registration.
    required_args = [action for action in parser._actions if action.required]
    for action in required_args:
        action.required = False
    try:
        add_launcher_args(parser)
    finally:
        for action in required_args:
            action.required = True
    parser.set_defaults(visualizer=["kit"])
    args, remaining = setup_preset_cli(parser)
    if not np.isfinite(args.slowdown) or args.slowdown <= 0 or not np.isfinite(args.fk_tolerance) or args.fk_tolerance <= 0:
        parser.error("slowdown 和 fk_tolerance 必须为有限正数")
    if not np.isfinite(args.amp_tolerance) or args.amp_tolerance <= 0:
        parser.error("amp_tolerance must be finite and positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("max_frames 必须为正整数")
    if not np.isfinite(args.orientation_tolerance_deg) or args.orientation_tolerance_deg <= 0:
        parser.error("orientation_tolerance_deg 必须为有限正数")
    if args.loop and args.max_frames is not None:
        parser.error("循环查看与有限帧检查请分别运行")
    sys.argv = [sys.argv[0]] + remaining
    return args


def main():
    args = arguments()
    motion = load_motion(args.motion)
    import x2_style_mimic.tasks  # noqa: F401

    env_cfg, _ = resolve_task_config(args.task, "")
    if args.device is not None:
        env_cfg.sim.device = args.device
    # 零物理步时 PhysX/Fabric 的渲染缓存可能停留在出生姿态。
    # 此回放使用显式 USD 同步；只改变本进程，不修改资产或训练配置。
    env_cfg.sim.use_fabric = False
    # 这里只用 scene 和 sim，不构造 reward/termination/action manager。
    scene_cfg = env_cfg.scene.copy()
    scene_cfg.num_envs = 1
    scene_cfg.contact_forces = None
    scene_cfg.robot = scene_cfg.robot.copy()
    scene_cfg.robot.spawn.activate_contact_sensors = False
    scene_cfg.robot.spawn.rigid_props.disable_gravity = True
    scene_cfg.robot.init_state.pos = tuple(float(v) for v in motion["root_pos_w"][0])
    scene_cfg.robot.init_state.rot = tuple(float(v) for v in motion["root_quat_xyzw"][0])
    scene_cfg.robot.init_state.joint_pos = dict(zip(motion["joint_names"].tolist(), motion["joint_pos"][0].tolist()))
    env_cfg.scene = scene_cfg
    with launch_simulation(env_cfg, args):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from common.x2_reference_visuals import ReferenceVisualSync

        sim = sim_utils.SimulationContext(env_cfg.sim)
        sim.set_setting("/rtx/hydra/readTransformsFromFabricInRenderDelegate", False)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        # 显示由下方同步器负责，避免 PhysX 的自动 USD 回写覆盖显示通道。
        sim.set_setting("/physics/updateToUsd", False)
        sim.set_setting("/physics/updateVelocitiesToUsd", False)
        if args.loop and not sim.visualizers:
            sim.clear_instance()
            raise ValueError("--loop 仅用于有可视化窗口的人工检查")
        robot = scene["robot"]
        visual_sync = ReferenceVisualSync(sim.stage, scene.env_prim_paths[0] + "/Robot", robot.body_names)
        source_names = motion["joint_names"].tolist()
        # 把参考列重排成实际 PhysX 关节顺序，包含两个头部关节。
        order = joint_indices(source_names, robot.joint_names)
        if set(robot.joint_names) != set(source_names):
            raise ValueError("Isaac 资产关节集合与冻结参考不一致")
        states = torch.tensor(motion["joint_pos"][:, order], device=sim.device)
        speeds = torch.tensor(motion["joint_vel"][:, order], device=sim.device)
        poses = torch.tensor(np.c_[motion["root_pos_w"], motion["root_quat_xyzw"]], device=sim.device)
        velocities = torch.tensor(np.c_[motion["root_lin_vel_w"], motion["root_ang_vel_w"]], device=sim.device)
        poses[:, :3] += scene.env_origins[0]
        limits = robot.data.joint_pos_limits.torch[0]
        violation = torch.maximum(limits[:, 0] - states, states - limits[:, 1]).clamp_min(0)
        max_violation = float(violation.max().item())
        if max_violation > 1e-5:
            raise ValueError(f"参考违反 Isaac 资产硬限位：{max_violation:.6g} rad")
        body_names = motion["body_names"].tolist()
        isaac_names = ["base_link" if name == "pelvis" else name for name in body_names]
        body_ids = joint_indices(robot.body_names, isaac_names)
        body_errors = np.zeros(len(body_names))
        orientation_errors = np.zeros(len(body_names))
        joint_error, root_error = 0., 0.
        first_root = motion["root_pos_w"][0]
        sim.set_camera_view(tuple(first_root + [2.1, -2.1, 1.2]), tuple(first_root + [0, 0, .2]))
        print("[REFERENCE] KINEMATIC STATE REPLAY: no physics stepping, no contact-force validation")
        print("[REFERENCE] Render synchronization: explicit body poses -> USD; Fabric disabled for this replay")
        print(f"[REFERENCE] pelvis -> base_link, xyzw; actual joint mapping: {order.tolist()}")
        print(f"[REFERENCE] {len(states)} frames, {motion['metadata']['fps']:g} Hz; compare {len(body_names)} bodies")
        frame, count = 0, 0
        dt = float(motion["metadata"]["dt_s"])
        from isaaclab.managers import SceneEntityCfg

        from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.amp import (
            AMP_KEY_BODY_NAMES,
            AMP_STATE_DIM,
            AMPReferenceMotion,
            amp_observation,
        )

        # 按当前策略的关节顺序生成参考特征。
        amp_reference = AMPReferenceMotion(
            motion_file=args.motion,
            joint_names=policy_joint_names(),
            device=sim.device,
            step_dt=0.02,
        )

        amp_asset_cfg = SceneEntityCfg(
            "robot",
            joint_names=policy_joint_names(),
            body_names=list(AMP_KEY_BODY_NAMES),
            preserve_order=True,
        )
        amp_asset_cfg.resolve(scene)

        # amp_observation 目前只访问 env.scene。
        amp_env = SimpleNamespace(scene=scene)

        amp_feature_blocks = {
            "height": slice(0, 1),
            "gravity": slice(1, 4),
            "linear_velocity": slice(4, 7),
            "angular_velocity_scaled": slice(7, 10),
            "joint_position": slice(10, 39),
            "joint_velocity_scaled": slice(39, 68),
        }
        offset = 68
        for name, width in (
            ("body_relative_position", 3),
            ("body_relative_rotation_6d", 6),
            ("body_relative_linear_velocity", 3),
            ("body_relative_angular_scaled", 3),
        ):
            end = offset + width * len(AMP_KEY_BODY_NAMES)
            amp_feature_blocks[name] = slice(offset, end)
            offset = end
        if offset != AMP_STATE_DIM:
            raise ValueError("AMP feature blocks do not cover the current feature layout")


        amp_max_errors = {
            name: 0.0
            for name in amp_feature_blocks
        }
        amp_worst_frame = {
            name: -1
            for name in amp_feature_blocks
        }

        # 阈值针对经过固定缩放后的特征值。
        amp_tolerance = args.amp_tolerance

        print(
            "[AMP] Compare simulation and reference: "
            f"{AMP_STATE_DIM} features per frame"
        )

        try:
            while True:
                if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break
                start = time.perf_counter()
                with torch.inference_mode():
                    robot.write_root_link_pose_to_sim_index(root_pose=poses[frame:frame+1])
                    # 参考保存的是根原点速度。
                    # 显式转换为根刚体质心速度，再写入 PhysX。
                    root_velocity = velocities[frame:frame + 1].clone()
                    quaternion = poses[frame:frame + 1, 3:7]

                    # 根刚体质心相对根原点的位置，在根坐标系中表达。
                    offset_b = robot.data.body_com_pose_b.torch[
                        :, 0, :3
                    ]

                    # xyzw 四元数正旋转：将质心偏移转换到世界坐标系。
                    t = 2.0 * torch.cross(
                        quaternion[:, :3],
                        offset_b,
                        dim=-1,
                    )
                    offset_w = (
                        offset_b
                        + quaternion[:, 3:4] * t
                        + torch.cross(
                            quaternion[:, :3],
                            t,
                            dim=-1,
                        )
                    )

                    # v_com = v_root + omega × offset
                    root_velocity[:, :3] += torch.cross(
                        root_velocity[:, 3:6],
                        offset_w,
                        dim=-1,
                    )

                    robot.write_root_com_velocity_to_sim_index(
                        root_velocity=root_velocity,
                    )
                    robot.write_joint_state_to_sim_index(position=states[frame:frame+1], velocity=speeds[frame:frame+1])
                    sim.forward()
                    scene.update(dt)
                    actual_amp = amp_observation(
                        amp_env,
                        amp_asset_cfg,
                    )

                    expected_amp = amp_reference.states[
                        frame:frame + 1
                    ]

                    if actual_amp.shape != expected_amp.shape:
                        raise ValueError(
                            "AMP 形状不一致："
                            f"simulation={actual_amp.shape}, "
                            f"reference={expected_amp.shape}"
                        )

                    if not torch.isfinite(actual_amp).all().item():
                        raise ValueError(
                            f"第 {frame} 帧仿真 AMP 特征包含 NaN/Inf"
                        )

                    difference = (
                        actual_amp - expected_amp
                    ).abs()

                    for name, columns in amp_feature_blocks.items():
                        error = float(
                            difference[:, columns].max().item()
                        )

                        if error > amp_max_errors[name]:
                            amp_max_errors[name] = error
                            amp_worst_frame[name] = frame
                    all_body_poses = robot.data.body_link_pose_w.torch[0].cpu().numpy()
                    actual = all_body_poses[body_ids]
                    actual_pos = actual[:, :3] - scene.env_origins[0].cpu().numpy()
                    body_errors = np.maximum(body_errors, np.linalg.norm(actual_pos - motion["body_pos_w"][frame], axis=1))
                    # 使用 float64 计算小角度，避免 float32 点积误差被 arccos 放大。
                    current_quat = actual[:, 3:7].astype(np.float64)
                    reference_quat = motion["body_quat_xyzw"][frame].astype(np.float64)
                    dot = np.abs(np.sum(current_quat * reference_quat, axis=1))
                    dot /= np.linalg.norm(current_quat, axis=1) * np.linalg.norm(reference_quat, axis=1)
                    orientation_errors = np.maximum(orientation_errors, 2*np.arccos(np.clip(dot, -1, 1)))
                    joint_error = max(joint_error, float(torch.max(abs(robot.data.joint_pos.torch[0] - states[frame])).item()))
                    root_error = max(root_error, float(np.linalg.norm(actual_pos[0] - motion["root_pos_w"][frame])))
                    # FK 数据刷新不等于渲染网格刷新，需要显式把当前变换写入 USD。
                    visual_sync.update(all_body_poses)
                    sim.render()
                count += 1
                if count == 1 or count % 50 == 0:
                    print(f"[REFERENCE] frame={frame}/{len(states)-1}, t={motion['time_s'][frame]:.2f}s", flush=True)
                if sim.visualizers:
                    delay = dt * args.slowdown - (time.perf_counter() - start)
                    if delay > 0:
                        time.sleep(delay)
                if args.max_frames is not None and count >= args.max_frames:
                    break
                frame += 1
                if frame >= len(states):
                    if not args.loop:
                        break
                    frame = 0
        finally:
            amp_passed = bool(
                count >= len(states)
                and max(amp_max_errors.values()) <= amp_tolerance
            )

            print("\n[AMP] Maximum absolute feature errors:")

            for name, error in amp_max_errors.items():
                print(
                    f"  {name:26s}: {error:.8f} "
                    f"(frame {amp_worst_frame[name]})"
                )

            print(
                f"[AMP] Full-clip consistency passed: {amp_passed}"
            )

            report = {
                "status": "kinematic_mapping_check_not_dynamic_validation", "motion": str(args.motion.resolve()),
                "frames_checked": count, "reference_frames": len(states),
                "full_clip_checked": count >= len(states), "quat_order": "xyzw",
                "root_mapping": "pelvis -> base_link; identity transform", "physics_steps": 0,
                "render_sync": "explicit body poses to USD; use_fabric=False",
                "isaac_joint_names": robot.joint_names, "isaac_to_reference_indices": order.tolist(),
                "max_hard_limit_violation_rad": max_violation,
                "max_joint_write_error_rad": joint_error, "max_root_position_error_m": root_error,
                "max_body_position_error_m": float(body_errors.max()),
                "body_position_errors_m": dict(zip(body_names, body_errors.tolist())),
                "body_orientation_errors_deg": dict(zip(body_names, np.degrees(orientation_errors).tolist())),
                "fk_tolerance_m": args.fk_tolerance,
                "orientation_tolerance_deg": args.orientation_tolerance_deg,
                "position_mapping_passed": bool(count and body_errors.max() <= args.fk_tolerance),
                "orientation_mapping_passed": bool(count and np.degrees(orientation_errors.max()) <= args.orientation_tolerance_deg),
                "amp_feature_version": amp_reference.feature_version,
                "amp_state_dim": AMP_STATE_DIM,
                "body_velocity_reference": "finite differences of reference body poses",
                "amp_feature_tolerance": amp_tolerance,
                "amp_max_errors": amp_max_errors,
                "amp_worst_frame": amp_worst_frame,
                "amp_consistency_passed": amp_passed,
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(f"[REPORT] {args.report.resolve()}")
            print(f"[FK] max body error {body_errors.max():.6f}m; checked frames {count}")
            sim.clear_instance()
        if not report["position_mapping_passed"] or not report["orientation_mapping_passed"]:
            raise RuntimeError("跨模拟器骨架位置或方向误差超过阈值，请先检查映射再使用参考数据")
        if not report["amp_consistency_passed"]:
            raise RuntimeError(
                "AMP 全片段特征一致性检查未通过，"
                "请查看报告中的分项误差"
            )

if __name__ == "__main__":
    main()
