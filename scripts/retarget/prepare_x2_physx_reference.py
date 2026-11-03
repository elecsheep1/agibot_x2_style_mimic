"""Export X2 reference kinematics with IsaacLab/PhysX, without physics stepping.

Original NPZ files are never overwritten. This tool does not import AMP code,
create an RL environment, load a policy, or modify the training configuration.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.x2_motion_utils import joint_indices, load_motion


def differentiated_state(motion):
    """Return world root velocities and joint velocities from the full trajectory."""
    times = np.asarray(motion["time_s"], dtype=np.float64)
    dt = np.diff(times)
    if len(times) < 3 or not np.isfinite(times).all() or (dt <= 0).any():
        raise ValueError("Reference needs at least three finite, increasing timestamps")
    if not np.allclose(dt, dt[0], rtol=0, atol=1e-7):
        raise ValueError("Reference timestamps must be uniformly spaced")
    q = np.asarray(motion["root_quat_xyzw"], dtype=np.float64).copy()
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or (norm < 1e-8).any():
        raise ValueError("Invalid root quaternion")
    q /= norm
    previous = np.r_[0, np.arange(len(q) - 1)]
    following = np.r_[np.arange(1, len(q)), len(q) - 1]
    a, b = q[following], q[previous]
    # q_following * conjugate(q_previous): world-frame rotation increment.
    xyz = (-a[:, 3:] * b[:, :3] + b[:, 3:] * a[:, :3]
           - np.cross(a[:, :3], b[:, :3]))
    w = a[:, 3:] * b[:, 3:] + (a[:, :3] * b[:, :3]).sum(-1, keepdims=True)
    sign = np.where(w < 0, -1., 1.)
    xyz, w = xyz * sign, w * sign
    magnitude = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(magnitude, w)
    scale = np.where(magnitude > 1e-8, angle / np.maximum(magnitude, 1e-12), 2.)
    omega = xyz * scale / (times[following] - times[previous])[:, None]
    linear = np.gradient(np.asarray(motion["root_pos_w"], dtype=np.float64),
                         times, axis=0, edge_order=1)
    joints = np.gradient(np.asarray(motion["joint_pos"], dtype=np.float64),
                         times, axis=0, edge_order=1)
    return q.astype(np.float32), np.c_[linear, omega].astype(np.float32), joints.astype(np.float32)


def rotate(q, vector):
    """Rotate vectors using xyzw unit quaternions (PyTorch)."""
    t = 2 * torch.cross(q[..., :3], vector, dim=-1)
    return vector + q[..., 3:] * t + torch.cross(q[..., :3], t, dim=-1)


def inverse_rotate(q, vector):
    conjugate = torch.cat((-q[..., :3], q[..., 3:]), dim=-1)
    return rotate(conjugate, vector)


def arguments():
    from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion", type=Path)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--input-dir", type=Path, help="Convert every NPZ; preserve motions.json if present")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="Template-X2-Style-Mimic-v0")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-frames", type=int, help="Preview only; output is explicitly marked partial")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--velocity-tolerance", type=float, default=1e-3)
    # The launcher inspects the parser before final argument parsing.
    required = [a for a in parser._actions if a.required]
    source.required = False
    for action in required:
        action.required = False
    try:
        add_launcher_args(parser)
    finally:
        source.required = True
        for action in required:
            action.required = True
    parser.set_defaults(visualizer=[])
    args, remaining = setup_preset_cli(parser)
    if args.num_envs < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("num-envs/max-frames must be positive")
    if not np.isfinite(args.velocity_tolerance) or args.velocity_tolerance <= 0:
        parser.error("velocity-tolerance must be finite and positive")
    sys.argv = [sys.argv[0]] + remaining
    return args


def sources(args):
    manifest = None
    if args.motion:
        paths = [args.motion.resolve()]
    elif args.manifest:
        path = args.manifest.resolve()
        manifest = json.loads(path.read_text())
        paths = [(path.parent / e["file"]).resolve() for e in manifest["motions"]]
    else:
        directory = args.input_dir.resolve()
        paths = sorted(directory.glob("*.npz"))
        if (directory / "motions.json").exists():
            manifest = json.loads((directory / "motions.json").read_text())
    if not paths or len({p.name for p in paths}) != len(paths):
        raise ValueError("No motions found, or duplicate output filenames")
    out = args.output_dir.resolve()
    for path in paths:
        target = out / path.name
        if target in paths:
            raise ValueError("Output directory must be different from source directory")
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"{target}: use --overwrite to replace a generated copy")
    for name in ("report.json", "motions.json"):
        if (out / name).exists() and not args.overwrite:
            raise FileExistsError(out / name)
    return paths, manifest, out


def export_motion(path, motion, robot, scene, sim, args, out, asset):
    root_q, root_vel, joint_vel = differentiated_state(motion)
    count_full = len(root_q)
    count = min(count_full, args.max_frames or count_full)
    source_joints = motion["joint_names"].tolist()
    if set(source_joints) != set(robot.joint_names):
        raise ValueError("Reference and PhysX joint sets differ")
    order = joint_indices(source_joints, robot.joint_names)
    body_names = motion["body_names"].tolist()
    body_ids = joint_indices(robot.body_names,
                             ["base_link" if n == "pelvis" else n for n in body_names])
    if robot.is_fixed_base or robot.num_base_dofs != 6:
        raise ValueError("X2 preprocessing requires a floating-base articulation")
    device = sim.device
    positions = torch.as_tensor(motion["root_pos_w"][:count], device=device, dtype=torch.float32)
    quats = torch.as_tensor(root_q[:count], device=device)
    roots = torch.as_tensor(root_vel[:count], device=device)
    q = torch.as_tensor(motion["joint_pos"][:count, order], device=device, dtype=torch.float32)
    qd = torch.as_tensor(joint_vel[:count, order], device=device)
    result = {k: np.empty((count, len(body_ids), width), dtype=np.float32)
              for k, width in (("body_pos_w", 3), ("body_quat_xyzw", 4),
                               ("body_lin_vel_w", 3), ("body_ang_vel_w", 3))}
    max_velocity_error = 0.
    max_fk_position_difference = 0.
    for start in range(0, count, args.num_envs):
        stop = min(count, start + args.num_envs)
        n = stop - start
        ids = torch.arange(n, dtype=torch.long, device=device)
        pose = torch.cat((positions[start:stop] + scene.env_origins[:n], quats[start:stop]), -1)
        robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
        # PhysX consumes COM velocity. Explicitly shift from the root origin;
        # this also avoids the installed link-velocity writer's COM ambiguity.
        offset = rotate(quats[start:stop], robot.data.body_com_pose_b.torch[:n, 0, :3])
        com_velocity = roots[start:stop].clone()
        com_velocity[:, :3] += torch.cross(com_velocity[:, 3:], offset, dim=-1)
        robot.write_root_com_velocity_to_sim_index(root_velocity=com_velocity, env_ids=ids)
        robot.write_joint_state_to_sim_index(position=q[start:stop], velocity=qd[start:stop], env_ids=ids)
        sim.forward()
        scene.update(float(motion["metadata"]["dt_s"]))
        poses = robot.data.body_link_pose_w.torch[:n, body_ids].clone()
        root_position = robot.data.root_link_pose_w.torch[:n, :3]
        jac = robot.data.body_link_jacobian_w.torch[:n, body_ids, :, robot.num_base_dofs:]
        relative_motion = torch.einsum("nbij,nj->nbi", jac, qd[start:stop])
        omega = roots[start:stop, None, 3:].expand_as(poses[..., :3])
        linear = (roots[start:stop, None, :3]
                  + torch.cross(omega, poses[..., :3] - root_position[:, None], dim=-1)
                  + relative_motion[..., :3])
        angular = omega + relative_motion[..., 3:]
        predicted = torch.cat((linear, angular), -1)
        actual = robot.data.body_link_vel_w.torch[:n, body_ids]
        error = float((predicted - actual).abs().max())
        max_velocity_error = max(max_velocity_error, error)
        if not torch.isfinite(predicted).all() or not torch.isfinite(actual).all() or error > args.velocity_tolerance:
            raise RuntimeError(f"{path.name} frames {start}:{stop}: J*qdot vs PhysX velocity error={error:g}")
        world_pos = poses[..., :3] - scene.env_origins[:n, None]
        for key, value in (("body_pos_w", world_pos), ("body_quat_xyzw", poses[..., 3:]),
                           ("body_lin_vel_w", linear), ("body_ang_vel_w", angular)):
            result[key][start:stop] = value.cpu().numpy()
        original = torch.as_tensor(motion["body_pos_w"][start:stop], device=device)
        max_fk_position_difference = max(max_fk_position_difference, float((world_pos - original).abs().max()))
        print(f"[PhysX FK] {path.name}: {stop}/{count}", flush=True)
    payload = {}
    for key, value in motion.items():
        if key == "metadata":
            continue
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == count_full and value.dtype.kind not in "US":
            value = value[:count].copy()
        payload[key] = value
    payload.update(result)
    payload["root_quat_xyzw"] = root_q[:count]
    payload["root_lin_vel_w"] = root_vel[:count, :3]
    payload["root_ang_vel_w"] = root_vel[:count, 3:]
    payload["joint_vel"] = joint_vel[:count]
    policy_ids = joint_indices(source_joints, motion["policy_joint_names"].tolist())
    payload["policy_joint_vel"] = joint_vel[:count, policy_ids]
    qr = torch.from_numpy(root_q[:count])
    payload["root_lin_vel_b"] = inverse_rotate(qr, torch.from_numpy(root_vel[:count, :3])).numpy()
    payload["root_ang_vel_b"] = inverse_rotate(qr, torch.from_numpy(root_vel[:count, 3:])).numpy()
    gravity = torch.zeros((count, 3)); gravity[:, 2] = -1
    payload["projected_gravity_b"] = inverse_rotate(qr, gravity).numpy()
    metadata = dict(motion["metadata"])
    info = dict(backend="isaaclab_physx", method="root_joint_differences_then_link_jacobian",
                velocity_frame="world", velocity_point="link_origin", physics_steps=0,
                source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                usd_path=str(asset), usd_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                exported_frames=count, source_frames=count_full, partial=count != count_full,
                max_velocity_error=max_velocity_error, max_fk_position_difference_m=max_fk_position_difference)
    metadata["physx_kinematics"] = info
    metadata["num_frames"] = count
    payload["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    for key, value in payload.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Generated field {key} contains NaN/Inf")
    target = out / path.name
    temporary = target.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(target)
    load_motion(target)  # Recheck original schema and policy-joint consistency.
    return dict(file=path.name, **info)


def main():
    args = arguments()
    paths, manifest, out = sources(args)
    motions = [load_motion(p) for p in paths]
    import x2_style_mimic.tasks  # noqa: F401
    from isaaclab_tasks.utils import launch_simulation, resolve_task_config
    cfg, _ = resolve_task_config(args.task, "")
    if args.device is not None:
        cfg.sim.device = args.device
    cfg.sim.use_fabric = True  # sim.forward() must refresh PhysX articulation kinematics.
    scene_cfg = cfg.scene.copy()
    scene_cfg.num_envs = args.num_envs
    scene_cfg.contact_forces = None
    scene_cfg.robot = scene_cfg.robot.copy()
    scene_cfg.robot.spawn.activate_contact_sensors = False
    scene_cfg.robot.spawn.rigid_props.disable_gravity = True
    scene_cfg.robot.init_state.pos = tuple(float(v) for v in motions[0]["root_pos_w"][0])
    scene_cfg.robot.init_state.rot = tuple(float(v) for v in motions[0]["root_quat_xyzw"][0])
    scene_cfg.robot.init_state.joint_pos = dict(zip(motions[0]["joint_names"].tolist(), motions[0]["joint_pos"][0].tolist()))
    cfg.scene = scene_cfg
    asset = Path(scene_cfg.robot.spawn.usd_path)
    out.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args):
        from isaaclab.sim import SimulationContext
        from isaaclab.scene import InteractiveScene
        sim = SimulationContext(cfg.sim)
        try:
            scene = InteractiveScene(scene_cfg)
            sim.reset()
            robot = scene["robot"]
            reports = [export_motion(p, m, robot, scene, sim, args, out, asset) for p, m in zip(paths, motions)]
            # Save before SimulationApp.close(), which may terminate the process.
            if manifest is not None and all(not r["partial"] for r in reports):
                copied = json.loads(json.dumps(manifest))
                for entry in copied["motions"]:
                    entry["file"] = Path(entry["file"]).name
                    if not (out / entry["file"]).is_file():
                        raise ValueError("Manifest references a file that was not converted")
                (out / "motions.json").write_text(json.dumps(copied, ensure_ascii=False, indent=2) + "\n")
            (out / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
            print(f"[PhysX FK] Done: {out}", flush=True)
        finally:
            sim.clear_instance()


if __name__ == "__main__":
    main()
