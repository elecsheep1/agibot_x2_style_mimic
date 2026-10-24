"""Free-root diagnostic; does not turn an open-loop failure into a reference rejection."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
import x2_style_mimic.tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--mode',choices=['hold','follow'],default='hold')
    p.add_argument('--duration',type=float,default=3.)
    p.add_argument('--frame', type=int, default=None, help='Hold a specific reference frame instead of the last frame.')
    p.add_argument('--initial-velocity', choices=['zero', 'reference'], default='zero')
    p.add_argument('--spawn-lift', type=float, default=.002, help='Additional spawn height in meters.')
    p.add_argument('--report',type=Path,required=True)
    # AppLauncher inspects the parser before the final CLI parse.
    # Defer required custom arguments only during launcher registration.
    required_args = [action for action in p._actions if action.required]
    for action in required_args:
        action.required = False
    try:
        add_launcher_args(p)
    finally:
        for action in required_args:
            action.required = True
    a,remaining = setup_preset_cli(p)
    sys.argv = [sys.argv[0]]+remaining
    if not np.isfinite(a.duration) or a.duration <= 0: p.error('duration must be positive')
    if a.report.exists(): p.error('Choose a new report path; existing report protected.')
    from x2_motion_utils import load_motion
    motion = load_motion(a.motion)
    if a.frame is not None and (a.mode != 'hold' or not 0 <= a.frame < len(motion['time_s'])):
        p.error('--frame requires hold mode and a valid frame index')
    if not np.isfinite(a.spawn_lift) or a.spawn_lift < 0:
        p.error('--spawn-lift must be finite and nonnegative')
    cfg,_ = resolve_task_config('X2-Prone-MultiMotion-v0','')
    cfg.scene.num_envs = 1
    if a.device is not None:
        cfg.sim.device = a.device
    standing = cfg.terminations.progress.params
    if not np.isclose(cfg.sim.dt*cfg.decimation,motion['metadata']['dt_s']):
        raise ValueError('Physics control dt and motion dt must match.')
    with launch_simulation(cfg,a):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        import warp as wp
        sim = sim_utils.SimulationContext(cfg.sim)
        try:
            scene = InteractiveScene(cfg.scene); sim.reset()
            robot = scene['robot']; sensor = scene.sensors['contact_forces']
            names = list(robot.body_names)
            paths_by_name = {s.rsplit('/',1)[-1]:s for s in sensor.body_physx_view.prim_paths}
            paths = [paths_by_name[n] for n in names]
            contact = sensor._physics_sim_view.create_rigid_contact_view(paths,filter_patterns=[paths]*len(paths))
            if contact.sensor_count != len(names) or contact.filter_count != len(names):
                raise RuntimeError('Unexpected exact-body contact matrix layout.')
            order = [motion['joint_names'].tolist().index(n) for n in robot.joint_names]
            targets = torch.as_tensor(motion['joint_pos'][:,order],dtype=torch.float32,device=sim.device)
            start = len(targets)-1 if a.mode=='hold' else 0
            if a.frame is not None:
                start = a.frame
            pose = torch.as_tensor(np.r_[motion['root_pos_w'][start],motion['root_quat_xyzw'][start]][None],
                                   dtype=torch.float32,device=sim.device)
            pose[:,:3] += scene.env_origins
            pose[:,2] += a.spawn_lift
            # Root initialized once only, then fully free under gravity.
            robot.write_root_link_pose_to_sim_index(root_pose=pose)
            robot.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(1,6,device=sim.device))
            robot.write_joint_state_to_sim_index(position=targets[[start]],velocity=torch.zeros_like(targets[[start]]))
            if a.initial_velocity == 'reference':
                link_velocity = torch.as_tensor(np.r_[motion['root_lin_vel_w'][start], motion['root_ang_vel_w'][start]][None],
                                                dtype=torch.float32, device=sim.device)
                joint_velocity = torch.as_tensor(motion['joint_vel'][start, order][None], dtype=torch.float32, device=sim.device)
                robot.write_root_link_velocity_to_sim_index(root_velocity=link_velocity)
                offset = robot.data.body_com_pose_b.torch[:, 0, :3]
                quat = pose[:, 3:7]
                cross = 2.0 * torch.cross(quat[:, :3], offset, dim=-1)
                offset_w = offset + quat[:, 3:4] * cross + torch.cross(quat[:, :3], cross, dim=-1)
                com_velocity = link_velocity.clone()
                com_velocity[:, :3] += torch.cross(link_velocity[:, 3:6], offset_w, dim=-1)
                robot.write_root_com_velocity_to_sim_index(root_velocity=com_velocity)
                robot.write_joint_state_to_sim_index(position=targets[[start]], velocity=joint_velocity)
            robot.set_joint_position_target_index(target=targets[[start]])
            sim.forward(); scene.update(cfg.sim.dt)
            peak = torch.zeros(len(names),len(names),device=sim.device)
            early_peak = torch.zeros_like(peak)
            records = []; longest = streak = 0
            knees = [robot.joint_names.index(s+'_knee_joint') for s in ('left','right')]
            dt = cfg.sim.dt*cfg.decimation
            count = max(1, round(a.duration/dt)) if a.mode=='hold' else len(targets)
            with torch.inference_mode():
                for step in range(count):
                    index = start if a.mode=='hold' else step
                    robot.set_joint_position_target_index(target=targets[[index]])
                    for _ in range(cfg.decimation):
                        scene.write_data_to_sim();sim.step(render=bool(sim.visualizers));scene.update(cfg.sim.dt)
                        f = wp.to_torch(contact.get_contact_force_matrix(dt=cfg.sim.dt)).norm(dim=-1)
                        peak = torch.maximum(peak,f)
                        if step * dt < .10:
                            early_peak = torch.maximum(early_peak, f)
                    height = float(robot.data.root_link_pose_w.torch[0,2]-scene.env_origins[0,2])
                    cosine = float(-robot.data.projected_gravity_b.torch[0,2])
                    velocity = robot.data.root_link_vel_w.torch[0]
                    speed = float(velocity[:3].norm()); spin = float(velocity[3:].norm())
                    knee = float(robot.data.joint_pos.torch[0,knees].abs().max())
                    foot = [float(scene.sensors[s+'_foot_ground_contact'].data.force_matrix_w_history.torch.norm(dim=-1).max())
                            for s in ('left','right')]
                    head = max(float(scene.sensors[s].data.force_matrix_w_history.torch.norm(dim=-1).max())
                               for s in ('head_yaw_ground_contact','head_pitch_ground_contact'))
                    torso_id = names.index("torso_link")
                    torso_q = robot.data.body_link_pose_w.torch[0, torso_id, 3:7]
                    torso_q = torso_q / torso_q.norm().clamp_min(1e-8)
                    torso_cos = float(1 - 2 * torso_q[:2].square().sum())
                    valid = bool(
                        height > standing["minimum_height"]
                        and cosine > np.cos(standing["maximum_tilt"])
                        and torso_cos > np.cos(standing["maximum_torso_tilt"])
                        and speed < standing["maximum_linear_speed"]
                        and spin < standing["maximum_angular_speed"]
                        and knee < standing["maximum_knee_angle"]
                        and min(foot) > standing["force_threshold"]
                    )
                    streak = streak+1 if valid else 0;longest = max(longest,streak)
                    records.append(dict(t=(step+1)*dt,height=height,upright_cos=cosine,linear_speed=speed,
                        angular_speed=spin,knee_max=knee,foot_ground_forces=foot,head_ground_force=head,
                        valid_standing=valid,joint_tracking_rms=float((robot.data.joint_pos.torch[0]-targets[index]).square().mean().sqrt())))
            matrix = peak.cpu().numpy();pairs=[]
            for i in range(len(names)):
                for j in range(i+1,len(names)):
                    force = float(max(matrix[i,j],matrix[j,i]))
                    if force>10.:pairs.append(dict(pair=[names[i],names[j]],peak_force_N=force))
            pairs.sort(key=lambda item:item['peak_force_N'],reverse=True)
            tail = records[-max(1,round(1/dt)):]
            report = dict(mode=a.mode,motion=str(a.motion.resolve()),physics_dt=cfg.sim.dt,control_dt=dt,
                initial_frame=start, initial_velocity=a.initial_velocity, spawn_lift_m=a.spawn_lift,
                solver_type=cfg.sim.physics.solver_type,gravity=list(cfg.sim.gravity),
                root_written_only_at_initialization=True,duration_s=count*dt,
                standing_thresholds={key: value for key, value in standing.items()
                                     if isinstance(value, (int, float))},
                stable_last_second=bool(len(tail)*dt>=1. and all(r['valid_standing'] for r in tail)),
                longest_valid_standing_s=longest*dt,final=records[-1],self_contact_pairs=pairs,samples=records,
                interpretation='Open-loop PD test, no balance controller. Failure alone does not establish AMP infeasibility; success does not certify full dynamic feasibility.')
            a.report.parent.mkdir(parents=True,exist_ok=True)
            a.report.write_text(json.dumps(report,indent=2)+'\n')
            print('[PHYSICS]',json.dumps({k:v for k,v in report.items() if k not in ('samples','self_contact_pairs')}),flush=True)
            print('[SELF CONTACT TOP]',json.dumps(pairs[:8]),flush=True)
            print('[REPORT]',a.report.resolve(),flush=True)
            early = [r for r in records if r['t'] <= .10 + 1e-8]
            print('\n' + '─' * 72, flush=True)
            print('[参考帧 PD 对照] 无策略；根节点自由运动，仅保持关节目标', flush=True)
            print(f'参考帧：{start} | 初始速度：{a.initial_velocity} | 抬高：{a.spawn_lift:.3f} m', flush=True)
            print(f'全程根高度：{min(r["height"] for r in records):.3f}～{max(r["height"] for r in records):.3f} m', flush=True)
            if early:
                print(f'前 0.10 s：最大线速度 {max(r["linear_speed"] for r in early):.3f} m/s'
                      f' | 最大角速度 {max(r["angular_speed"] for r in early):.3f} rad/s'
                      f' | 自碰撞力峰值 {float(early_peak.max()):.1f} N', flush=True)
            for row in records[:5]:
                print(f'  t={row["t"]:.2f} s | 高度 {row["height"]:.3f} m'
                      f' | 线速度 {row["linear_speed"]:.3f} m/s'
                      f' | 角速度 {row["angular_speed"]:.3f} rad/s'
                      f' | 关节跟踪误差 {row["joint_tracking_rms"]:.3f} rad', flush=True)
            print(f'末态：高度 {records[-1]["height"]:.3f} m | 线速度 {records[-1]["linear_speed"]:.3f} m/s', flush=True)
            print('注意：固定 PD 不是平衡控制器；倒下本身不能证明参考动作不可行。', flush=True)
            print(f'报告：{a.report.resolve()}\n' + '─' * 72, flush=True)
        finally:
            sim.clear_instance()


if __name__=='__main__': main()
