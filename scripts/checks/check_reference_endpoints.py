"""Compare four reference endpoints with Isaac FK; no policy or physics stepping."""

# Support direct execution: python scripts/<category>/<tool>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

# Existing reference display helper; never changes files on disk.
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli


def tilt(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    return float(np.degrees(np.arccos(np.clip(1-2*(q[0]**2+q[1]**2), -1, 1))))


def inspect(actual, joint_position, body_names, joint_names, origin, motion):
    source = motion['body_names'].tolist()
    source_map = ['base_link' if name == 'pelvis' else name for name in source]
    ids = [body_names.index(name) for name in source_map]
    position = actual[ids, :3] - origin
    ref_pos = motion['body_pos_w'][-1]
    qa = actual[ids, 3:7].astype(np.float64)
    qr = motion['body_quat_xyzw'][-1].astype(np.float64)
    dot = np.abs((qa*qr).sum(-1)) / (np.linalg.norm(qa, axis=-1)*np.linalg.norm(qr, axis=-1))
    error_deg = np.degrees(2*np.arccos(np.clip(dot, -1, 1)))
    root = actual[body_names.index('base_link')]
    torso = actual[body_names.index('torso_link')]
    ground_z = float(motion['metadata']['ground_z'])
    knees = [float(joint_position[joint_names.index(side+'_knee_joint')]) for side in ('left', 'right')]
    height = float(root[2]-origin[2]-ground_z)
    root_tilt, torso_tilt = tilt(root[3:7]), tilt(torso[3:7])
    gates = dict(height=bool(height > .5), pelvis_tilt=bool(root_tilt < np.degrees(.35)),
                 torso_tilt=bool(torso_tilt < np.degrees(.55)), knees=bool(max(map(abs, knees)) < 1.5))
    feet = {}
    for side in ('left', 'right'):
        pose = actual[body_names.index(side+'_ankle_roll_link')]
        feet[side] = dict(ankle_height_m=float(pose[2]-origin[2]-ground_z),
                          local_z_tilt_deg=tilt(pose[3:7]))
    return dict(height_m=height, pelvis_tilt_deg=root_tilt, torso_tilt_deg=torso_tilt,
                reference_torso_tilt_deg=tilt(motion['body_quat_xyzw'][-1, source.index('torso_link')]),
                knee_angles_rad=knees, pose_gates=gates, ankle_geometry=feet,
                max_body_position_error_m=float(np.linalg.norm(position-ref_pos, axis=-1).max()),
                max_body_orientation_error_deg=float(error_deg.max()),
                mapping_passed=bool(np.linalg.norm(position-ref_pos, axis=-1).max()<.005 and error_deg.max()<.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/motions/prone_getup/motions.json'))
    parser.add_argument('--duration', type=float, default=30., help='Seconds to display the fixed endpoints.')
    parser.add_argument('--report', type=Path, default=Path('logs/reference_replay/multimotion_endpoints.json'))
    add_launcher_args(parser)
    parser.set_defaults(visualizer=['kit'])
    args, remaining = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + remaining
    if not np.isfinite(args.duration) or args.duration <= 0: parser.error('duration must be positive')
    manifest = args.manifest.resolve()
    entries = json.loads(manifest.read_text())['motions']
    paths = [(manifest.parent/e['file']).resolve() for e in entries if e.get('weight', 1)>0]
    if not paths: parser.error('No enabled reference motions')
    from common.x2_motion_utils import load_motion
    motions = [load_motion(path) for path in paths]
    import x2_style_mimic.tasks
    cfg, _ = resolve_task_config('Template-X2-Style-Mimic-v0', '')
    if args.device is not None: cfg.sim.device = args.device
    cfg.sim.use_fabric = False
    scene_cfg = cfg.scene.copy()
    scene_cfg.num_envs = len(paths)
    scene_cfg.env_spacing = 2.5
    scene_cfg.contact_forces = None
    scene_cfg.robot = scene_cfg.robot.copy()
    scene_cfg.robot.spawn.activate_contact_sensors = False
    scene_cfg.robot.spawn.rigid_props.disable_gravity = True
    cfg.scene = scene_cfg
    with launch_simulation(cfg, args):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from common.x2_reference_visuals import ReferenceVisualSync
        sim = sim_utils.SimulationContext(cfg.sim)
        try:
            sim.set_setting('/rtx/hydra/readTransformsFromFabricInRenderDelegate', False)
            scene = InteractiveScene(scene_cfg)
            sim.reset()
            sim.set_setting('/physics/updateToUsd', False)
            sim.set_setting('/physics/updateVelocitiesToUsd', False)
            robot = scene['robot']
            targets, poses = [], []
            for motion in motions:
                names = motion['joint_names'].tolist()
                if set(names) != set(robot.joint_names): raise ValueError('Joint sets differ')
                order = [names.index(name) for name in robot.joint_names]
                targets.append(motion['joint_pos'][-1, order])
                poses.append(np.r_[motion['root_pos_w'][-1], motion['root_quat_xyzw'][-1]])
            targets = torch.tensor(np.asarray(targets), dtype=torch.float32, device=sim.device)
            poses = torch.tensor(np.asarray(poses), dtype=torch.float32, device=sim.device)
            poses[:, :3] += scene.env_origins
            limits = robot.data.joint_pos_limits.torch
            violation = torch.maximum(limits[..., 0]-targets, targets-limits[..., 1]).clamp_min(0)
            robot.write_root_link_pose_to_sim_index(root_pose=poses)
            robot.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(len(paths), 6, device=sim.device))
            robot.write_joint_state_to_sim_index(position=targets, velocity=torch.zeros_like(targets))
            sim.forward()
            scene.update(.02)
            actual = robot.data.body_link_pose_w.torch.cpu().numpy().copy()
            joints = robot.data.joint_pos.torch.cpu().numpy().copy()
            origins = scene.env_origins.cpu().numpy()
            rows = []
            synchronizers = []
            for i, (path, motion) in enumerate(zip(paths, motions)):
                row = dict(env_id=i, file=path.name, reference_frame=len(motion['time_s'])-1,
                           max_joint_limit_violation_rad=float(violation[i].max()),
                           **inspect(actual[i], joints[i], list(robot.body_names), list(robot.joint_names), origins[i], motion))
                rows.append(row)
                synchronizers.append(ReferenceVisualSync(sim.stage, scene.env_prim_paths[i]+'/Robot', robot.body_names))
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(dict(status='kinematic_endpoint_check_only', physics_steps=0,
                thresholds=dict(minimum_height=.5, maximum_pelvis_tilt_deg=float(np.degrees(.35)),
                                maximum_torso_tilt_deg=float(np.degrees(.55)), maximum_knee_angle=1.5),
                endpoints=rows, note='No contact forces or balance are validated; feet positions are ankle-body origins.'),
                ensure_ascii=False, indent=2), encoding='utf-8')
            print('\n[参考末帧姿态] 无策略、无物理步；每个环境展示一条参考末帧。', flush=True)
            for row in rows:
                print(f"环境 {row['env_id']}: {row['file']}\n"
                      f"  高度 {row['height_m']:.3f} m | 骨盆 {row['pelvis_tilt_deg']:.1f}°"
                      f" | 躯干 {row['torso_tilt_deg']:.1f}°（NPZ {row['reference_torso_tilt_deg']:.1f}°）"
                      f" | 膝 {[round(v,3) for v in row['knee_angles_rad']]}", flush=True)
                failed = [name for name, passed in row['pose_gates'].items() if not passed]
                print(f"  现有姿态阈值未满足项：{failed or '无'}"
                      f" | FK位置误差 {row['max_body_position_error_m']:.6f} m"
                      f" | FK方向误差 {row['max_body_orientation_error_deg']:.3f}°", flush=True)
            print('画面仅核对半蹲形态；不代表双脚承重或能自主保持平衡。', flush=True)
            print(f'报告：{args.report.resolve()}', flush=True)
            center = origins.mean(axis=0) + [0,0,.4]
            sim.set_camera_view(tuple(center+[5,-5,3.5]), tuple(center))
            start = time.monotonic()
            while True:
                if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers): break
                for sync, body_poses in zip(synchronizers, actual): sync.update(body_poses)
                sim.render()
                if not sim.visualizers or time.monotonic()-start >= args.duration: break
                time.sleep(.02)
        finally:
            sim.clear_instance()


if __name__ == '__main__':
    main()
