"""Free-root PD endpoint support audit. No policy, assistance, or automatic resets."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

BODY_NAMES = (
    'left_ankle_roll_link', 'right_ankle_roll_link',
    'left_wrist_roll_link', 'right_wrist_roll_link',
    'left_knee_link', 'right_knee_link',
    'left_elbow_link', 'right_elbow_link',
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('data/motions/prone_getup/motions.json'))
    p.add_argument('--duration', type=float, default=5.)
    p.add_argument('--spawn-lift', type=float, default=.01)
    p.add_argument('--initial-velocity', choices=('zero', 'reference'), default='zero')
    p.add_argument('--report', type=Path, default=Path('logs/reference_physics/multimotion_endpoint_support.json'))
    add_launcher_args(p)
    p.set_defaults(visualizer=['kit'])
    a, remaining = setup_preset_cli(p)
    sys.argv = [sys.argv[0]] + remaining
    if not np.isfinite(a.duration) or a.duration <= 0: p.error('duration must be positive')
    if not np.isfinite(a.spawn_lift) or a.spawn_lift < 0: p.error('spawn-lift must be nonnegative')
    from x2_motion_utils import load_motion
    manifest = a.manifest.resolve()
    entries = json.loads(manifest.read_text())['motions']
    paths = [(manifest.parent/e['file']).resolve() for e in entries if e.get('weight', 1)>0]
    if not paths: p.error('No enabled motions')
    motions = [load_motion(path) for path in paths]
    import x2_style_mimic.tasks
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
    cfg, _ = resolve_task_config('Template-X2-Style-Mimic-v0', '')
    if a.device is not None: cfg.sim.device = a.device
    cfg.sim.dt = .002
    cfg.sim.render_interval = 10
    if cfg.sim.physics is None: cfg.sim.physics = PhysxCfg(solver_type=0)
    else: cfg.sim.physics.solver_type = 0
    scene_cfg = cfg.scene.copy()
    scene_cfg.num_envs = len(paths)
    scene_cfg.env_spacing = 2.5
    scene_cfg.contact_forces = None
    scene_cfg.robot = scene_cfg.robot.copy()
    scene_cfg.robot.spawn.activate_contact_sensors = True
    scene_cfg.robot.spawn.rigid_props.disable_gravity = False
    for i, name in enumerate(BODY_NAMES):
        setattr(scene_cfg, f'support_probe_{i}', ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/Robot/Geometry/base_link/.*'+name,
            update_period=0., history_length=3,
            filter_prim_paths_expr=['/World/ground/GroundPlane/CollisionPlane']))
    cfg.scene = scene_cfg
    with launch_simulation(cfg, a):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        sim = sim_utils.SimulationContext(cfg.sim)
        records = []
        try:
            scene = InteractiveScene(scene_cfg)
            sim.reset()
            robot = scene['robot']
            sensors = [scene.sensors[f'support_probe_{i}'] for i in range(len(BODY_NAMES))]
            for sensor, name in zip(sensors, BODY_NAMES):
                if list(sensor.body_names) != [name] or sensor.contact_view.filter_count != 1:
                    raise RuntimeError(f'Probe mapping invalid: {name}')
            targets, joint_speeds, poses, velocities = [], [], [], []
            for motion in motions:
                names = motion['joint_names'].tolist()
                order = [names.index(name) for name in robot.joint_names]
                targets.append(motion['joint_pos'][-1, order])
                joint_speeds.append(motion['joint_vel'][-1, order])
                pose = np.r_[motion['root_pos_w'][-1], motion['root_quat_xyzw'][-1]].copy()
                pose[2] -= float(motion['metadata']['ground_z'])
                poses.append(pose)
                velocities.append(np.r_[motion['root_lin_vel_w'][-1], motion['root_ang_vel_w'][-1]])
            def tensor(values): return torch.tensor(np.asarray(values), dtype=torch.float32, device=sim.device)
            targets = tensor(targets); poses = tensor(poses)
            limits = robot.data.joint_pos_limits.torch
            violation = torch.maximum(limits[..., 0]-targets, targets-limits[..., 1]).clamp_min(0)
            if float(violation.max()) > 1e-5:
                raise ValueError(f'Endpoint exceeds Isaac joint limits by {float(violation.max())} rad; not silently clamped')
            reference_height = poses[:, 2].clone()
            poses[:, :3] += scene.env_origins
            poses[:, 2] += a.spawn_lift
            velocity = tensor(velocities) if a.initial_velocity == 'reference' else torch.zeros(len(paths), 6, device=sim.device)
            qvelocity = tensor(joint_speeds) if a.initial_velocity == 'reference' else torch.zeros_like(targets)
            # Root/joint state initialization only once. No further pose writes.
            robot.write_root_link_pose_to_sim_index(root_pose=poses)
            robot.write_root_link_velocity_to_sim_index(root_velocity=velocity)
            offset = robot.data.body_com_pose_b.torch[:, 0, :3]
            quat = poses[:, 3:7]
            cross = 2*torch.cross(quat[:, :3], offset, dim=-1)
            offset_w = offset + quat[:, 3:4]*cross + torch.cross(quat[:, :3], cross, dim=-1)
            com_velocity = velocity.clone()
            com_velocity[:, :3] += torch.cross(velocity[:, 3:], offset_w, dim=-1)
            robot.write_root_com_velocity_to_sim_index(root_velocity=com_velocity)
            robot.write_joint_state_to_sim_index(position=targets, velocity=qvelocity)
            robot.set_joint_position_target_index(target=targets)
            sim.forward(); scene.update(cfg.sim.dt)
            knee_ids = [robot.joint_names.index(s+'_knee_joint') for s in ('left', 'right')]
            torso_id = robot.body_names.index('torso_link')
            initial_torso_q = robot.data.body_link_pose_w.torch[:, torso_id, 3:7]
            initial_torso_tilt = (1-2*initial_torso_q[:, :2].square().sum(-1)).clamp(-1, 1).acos()
            center = scene.env_origins.mean(dim=0).cpu().numpy()+[0,0,.4]
            sim.set_camera_view(tuple(center+[5,-5,3.5]), tuple(center))
            total_steps = max(1, round(a.duration/.02))
            interrupted = False
            with torch.inference_mode():
                for step in range(total_steps):
                    if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                        interrupted = True; break
                    started = time.monotonic()
                    robot.set_joint_position_target_index(target=targets)
                    force_norm_sum = torch.zeros(len(paths), len(sensors), device=sim.device)
                    force_z_sum = torch.zeros_like(force_norm_sum)
                    for _ in range(10):
                        scene.write_data_to_sim()
                        sim.step(render=bool(sim.visualizers))
                        scene.update(cfg.sim.dt)
                        forces = torch.stack([s.data.force_matrix_w.torch[:, 0, 0, :] for s in sensors], dim=1)
                        force_norm_sum += forces.norm(dim=-1)
                        force_z_sum += forces[..., 2]
                    force_mean = force_norm_sum/10
                    force_z_mean = force_z_sum/10
                    height = robot.data.root_link_pose_w.torch[:, 2]-scene.env_origins[:, 2]
                    root_velocity = robot.data.root_link_vel_w.torch
                    tq = robot.data.body_link_pose_w.torch[:, torso_id, 3:7]
                    torso_tilt = (1-2*tq[:, :2].square().sum(-1)).clamp(-1, 1).acos()
                    for i, path in enumerate(paths):
                        both_feet = bool((force_mean[i, :2] > 5).all())
                        monitored_other = float(force_mean[i, 2:].sum())
                        records.append(dict(env_id=i, file=path.name, time_s=(step+1)*.02,
                            height_m=float(height[i]), torso_tilt_deg=float(torch.rad2deg(torso_tilt[i])),
                            root_speed=float(root_velocity[i, :3].norm()), angular_speed=float(root_velocity[i, 3:].norm()),
                            knee_angles=robot.data.joint_pos.torch[i, knee_ids].tolist(),
                            joint_tracking_rms=float((robot.data.joint_pos.torch[i]-targets[i]).square().mean().sqrt()),
                            ground_force_norm_mean_n=force_mean[i].tolist(), ground_force_z_mean_n=force_z_mean[i].tolist(),
                            both_feet_contact=both_feet,
                            monitored_hand_knee_elbow_force_n=monitored_other,
                            endpoint_pose_retained=bool(abs(float(height[i]-reference_height[i]))<.05
                                and abs(float(torso_tilt[i]-initial_torso_tilt[i]))<np.deg2rad(10)),
                        ))
                    if sim.visualizers:
                        delay=.02-(time.monotonic()-started)
                        if delay>0: time.sleep(delay)
            report = dict(status='free_root_fixed_joint_PD_support_diagnostic', root_written_only_at_initialization=True,
                initial_velocity=a.initial_velocity, spawn_lift_m=a.spawn_lift, physics_dt=.002, control_dt=.02,
                duration_requested_s=a.duration, interrupted=interrupted, force_body_order=list(BODY_NAMES),
                force_definition='Mean over ten physics substeps; ground-filtered force norm and signed world Z component.',
                limits='Only listed feet/wrists/knees/elbows monitored; other body contacts may still exist. Fixed PD is not a balance controller.',
                samples=records)
            a.report.parent.mkdir(parents=True, exist_ok=True)
            a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print('\n[末帧 PD 支撑对照] 无策略、无辅助；根自由；关节保持原末帧目标。', flush=True)
            print('力为最近 0.02 秒物理子步均值，单位 N；仅列出的刚体，不是全身承重总和。', flush=True)
            for i, path in enumerate(paths):
                rows = [r for r in records if r['env_id']==i]
                print(f'\n环境 {i}: {path.name}', flush=True)
                for second in [.02,.1,.5,1,2,3,5]:
                    if not rows or second>rows[-1]['time_s']+1e-6: continue
                    row=min(rows,key=lambda r:abs(r['time_s']-second));f=row['ground_force_norm_mean_n']
                    print(f"  t={row['time_s']:.2f} | 高度 {row['height_m']:.3f} | 躯干 {row['torso_tilt_deg']:.1f}°"
                          f" | 速度 {row['root_speed']:.3f} | 脚 {f[0]:.1f}/{f[1]:.1f}"
                          f" | 腕 {f[2]:.1f}/{f[3]:.1f} | 膝 {f[4]:.1f}/{f[5]:.1f}"
                          f" | 肘 {f[6]:.1f}/{f[7]:.1f} | 关节误差 {row['joint_tracking_rms']:.3f} rad", flush=True)
            print('\n倒下不能单独证明参考不可行；保持也不能代替策略平衡验收。', flush=True)
            print(f'报告：{a.report.resolve()}', flush=True)
        finally:
            sim.clear_instance()


if __name__ == '__main__':
    main()
