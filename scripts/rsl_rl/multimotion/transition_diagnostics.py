"""Capture the first 10 s of each environment before automatic resets."""
import json
import math
from pathlib import Path

import torch
from diagnostics import MultiMotionDiagnostics
from analyze_references import analyze


def inverse_rotate_xyzw(q, vector):
    """Express a world vector in the quaternion's local frame."""
    xyz, w = q[..., :3], q[..., 3:4]
    cross = 2 * torch.cross(xyz, vector, dim=-1)
    return vector - w * cross + torch.cross(xyz, cross, dim=-1)


def roll_pitch_yaw_xyzw(q):
    """ZYX Euler angles; positive pitch points the local +X axis down."""
    x, y, z, w = q.unbind(-1)
    roll = torch.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = torch.asin((2 * (w*y - z*x)).clamp(-1, 1))
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return torch.stack((roll, pitch, yaw), dim=-1)


class TransitionDiagnostics(MultiMotionDiagnostics):
    def __init__(self, env, episodes_per_env, output_path):
        assisted = any(force > 0 for force in env.cfg.assist_force_levels)
        super().__init__(env, episodes_per_env, output_path, require_zero_assist=not assisted)
        self.trace = []
        self.birth_frames = env._getup_reference_frame_ids.clone()
        self.trace_printed = False
        self.first_episode_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        robot = env.scene['robot']
        self.robot = robot
        self.torso_id = robot.body_names.index('torso_link')
        self.knee_ids = [robot.joint_names.index(n) for n in ('left_knee_joint', 'right_knee_joint')]
        self.foot_ids = [robot.body_names.index(n) for n in ('left_ankle_roll_link', 'right_ankle_roll_link')]
        self.joint_names = [side + '_' + joint + '_joint'
                            for joint in ('hip_pitch', 'knee', 'ankle_pitch')
                            for side in ('left', 'right')] + ['waist_pitch_joint']
        self.joint_ids = [robot.joint_names.index(n) for n in self.joint_names]
        self.support_sensors = {}
        for name in ('left_wrist_roll_link', 'right_wrist_roll_link',
                     'left_knee_link', 'right_knee_link',
                     'left_elbow_link', 'right_elbow_link'):
            sensor = env.scene.sensors['transition_' + name]
            if list(sensor.body_names) != [name] or sensor.contact_view.filter_count != 1:
                raise RuntimeError('Invalid ground probe mapping: ' + name)
            self.support_sensors[name] = sensor
        self.reference_audit = [analyze(Path(path)) for path in self.motion_paths]
        self.original_compute = env.reward_manager.compute
        def capture_compute(dt):
            reward = self.original_compute(dt)
            self.capture()
            return reward
        env.reward_manager.compute = capture_compute

    def capture(self):
        e, r = self.env, self.robot
        poses = r.data.root_link_pose_w.torch
        velocity = r.data.root_link_vel_w.torch
        body = r.data.body_link_pose_w.torch
        q = poses[:, 3:7]
        xyz, w = q[:, :3], q[:, 3:4]
        v = velocity[:, :3]
        t = 2 * torch.cross(xyz, v, dim=-1)
        velocity_b = v - w * t + torch.cross(xyz, t, dim=-1)
        torso_q = body[:, self.torso_id, 3:7]
        torso_tilt = (1-2*torso_q[:, :2].square().sum(-1)).clamp(-1, 1).acos() * (180/math.pi)
        root_tilt = (-r.data.projected_gravity_b.torch[:, 2]).clamp(-1, 1).acos() * (180/math.pi)
        root_rpy = roll_pitch_yaw_xyzw(q)
        torso_rpy = roll_pitch_yaw_xyzw(torso_q)
        angular_velocity_b = inverse_rotate_xyzw(q, velocity[:, 3:])
        # Horizontal velocity rotated by pelvis yaw only: it excludes vertical motion.
        yaw = root_rpy[:, 2]
        horizontal_velocity_heading = torch.stack((
            yaw.cos()*velocity[:, 0] + yaw.sin()*velocity[:, 1],
            -yaw.sin()*velocity[:, 0] + yaw.cos()*velocity[:, 1],
        ), dim=-1)
        foot_relative_b = inverse_rotate_xyzw(
            q[:, None, :].expand(-1, 2, -1),
            body[:, self.foot_ids, :3] - poses[:, None, :3],
        )
        foot_pose = body[:, self.foot_ids]
        foot_velocity = r.data.body_link_vel_w.torch[:, self.foot_ids]
        foot_rpy = roll_pitch_yaw_xyzw(foot_pose[..., 3:7]) * (180/math.pi)
        feet = []
        feet_current = []
        for name in ('left_foot_ground_contact', 'right_foot_ground_contact'):
            sensor = e.scene.sensors[name]
            if sensor.data.force_matrix_w.torch.shape[1:] != (1, 1, 3):
                raise RuntimeError('Invalid foot probe mapping: ' + name)
            feet.append(sensor.data.force_matrix_w_history.torch.norm(dim=-1).amax(dim=(1, 2, 3)))
            feet_current.append(sensor.data.force_matrix_w.torch[:, 0, 0, :])
        feet = torch.stack(feet, dim=-1)
        feet_current = torch.stack(feet_current, dim=1)
        term = e.action_manager.get_term('joint_pos')
        requested = term.raw_actions * term._scale + term._offset
        limits = r.data.joint_pos_limits.torch[:, term._joint_ids]
        saturation = ((requested < limits[..., 0]) | (requested > limits[..., 1])).float().mean(-1)
        progress = e._getup_progress_snapshot
        action_ids = list(range(r.num_joints)) if isinstance(term._joint_ids, slice) else list(term._joint_ids)
        action_columns = [action_ids.index(j) for j in self.joint_ids]
        support = {name: sensor.data.force_matrix_w.torch[:, 0, 0, :].norm(dim=-1)
                   for name, sensor in self.support_sensors.items()}
        reward_names = e.reward_manager.active_terms
        for env_id in range(e.num_envs):
            step = int(e.episode_length_buf[env_id])
            elapsed = step * e.step_dt
            terminal = bool(e.reset_buf[env_id])
            if not bool(self.first_episode_done[env_id]) and elapsed <= 10.000001 and (step == 1 or step % 5 == 0 or terminal):
                self.trace.append(dict(
                    env_id=env_id, motion_id=int(e._getup_reference_motion_ids[env_id]), time_s=elapsed,
                    birth_frame=int(self.birth_frames[env_id]),
                    leg_target_hold_active=bool(getattr(e, '_diagnostic_leg_target_hold', None).active[env_id])
                        if getattr(e, '_diagnostic_leg_target_hold', None) is not None else False,
                    force_n=float(progress['assist_applied_force_n'][env_id]),
                    episode_force_n=float(progress['assist_episode_force_n'][env_id]),
                    height=float(poses[env_id, 2]-e.scene.env_origins[env_id, 2]),
                    torso_tilt_deg=float(torso_tilt[env_id]), root_tilt_deg=float(root_tilt[env_id]),
                    velocity_b=velocity_b[env_id].tolist(), root_speed=float(velocity[env_id, :3].norm()),
                    angular_speed=float(velocity[env_id, 3:].norm()),
                    root_rpy_deg=(root_rpy[env_id] * (180/math.pi)).tolist(),
                    torso_rpy_deg=(torso_rpy[env_id] * (180/math.pi)).tolist(),
                    angular_velocity_b_rad_s=angular_velocity_b[env_id].tolist(),
                    horizontal_velocity_heading_m_s=horizontal_velocity_heading[env_id].tolist(),
                    horizontal_speed_m_s=float(velocity[env_id, :2].norm()),
                    feet_force_current_w_n=feet_current[env_id].tolist(),
                    feet_force_current_n=feet_current[env_id].norm(dim=-1).tolist(),
                    ankle_position_relative_pelvis_b_m=foot_relative_b[env_id].tolist(),
                    ankle_position_w_m=(foot_pose[env_id, :, :3] - e.scene.env_origins[env_id]).tolist(),
                    ankle_velocity_w_m_s=foot_velocity[env_id, :, :3].tolist(),
                    ankle_angular_velocity_w_rad_s=foot_velocity[env_id, :, 3:].tolist(),
                    ankle_rpy_w_deg=foot_rpy[env_id].tolist(),
                    ankle_total_tilt_deg=(1 - 2 * foot_pose[env_id, :, 3:5].square().sum(-1)).clamp(-1, 1).acos().mul(180/math.pi).tolist(),
                    knee_angles=r.data.joint_pos.torch[env_id, self.knee_ids].tolist(),
                    feet_force=feet[env_id].tolist(),
                    ankle_height=(body[env_id, self.foot_ids, 2]-e.scene.env_origins[env_id, 2]).tolist(),
                    raw_target_saturation_ratio=float(saturation[env_id]),
                    reset_joint_clamp_count=int(e._getup_reference_joint_clamp_counts[env_id]),
                    joints={name: dict(
                        position_rad=float(r.data.joint_pos.torch[env_id, j]),
                        velocity_rad_s=float(r.data.joint_vel.torch[env_id, j]),
                        requested_target_rad=float(requested[env_id, col]),
                        position_clamped_target_rad=float(requested[env_id, col].clamp(
                            limits[env_id, col, 0], limits[env_id, col, 1])),
                        final_target_rad=float(term.processed_actions[env_id, col]),
                        estimated_torque_nm=float(r.data.applied_torque.torch[env_id, j]),
                        estimated_unclipped_torque_nm=float(r.data.computed_torque.torch[env_id, j]),
                        effort_limit_nm=float(r.data.joint_effort_limits.torch[env_id, j]),
                    ) for name, j, col in zip(self.joint_names, self.joint_ids, action_columns)},
                    support_force_current_n={name: float(force[env_id]) for name, force in support.items()},
                    task_reward_per_step={name: float(e.reward_manager._step_reward[env_id, k] * e.step_dt)
                                          for k, name in enumerate(reward_names)},
                ))
            if terminal:
                self.first_episode_done[env_id] = True

    def report(self):
        super().report()
        report = json.loads(self.output_path.read_text())
        report['initial_transition_trace'] = self.trace
        report['reference_audit'] = self.reference_audit
        report['transition_field_notes'] = dict(
            torque='Implicit PD actuator estimate at last physics substep; not measured torque or proof of saturation.',
            support='Wrist/knee/elbow ground-filtered force magnitude at current sensor sample; feet_force is history peak.',
            rewards='Weighted task reward contributions including control dt; no AMP reward is computed in inference play.',
            targets='requested: scale+offset; position_clamped: hard limits; final: actual position/slew-limited command.',
            balance='RPY: xyzw ZYX degrees, positive pitch points +X down; Euler interpretation is limited near 90-degree pitch. Heading velocity: yaw-only frame +X forward/+Y left, horizontal. Angular velocity and ankle positions: full pelvis frame. Ankles are body origins, not contact points or COM projections. Current foot vectors are ground-filtered world-frame forces, not substep averages.',
            feet='Ankle position uses world axes with environment origin subtracted; velocities and RPY use world axes. These are ankle rigid-body origins/orientations, not sole contact-point velocities or calibrated sole angles. Nonzero ankle speed with contact may reflect rolling/pivoting as well as slip; no slip classification is inferred.',
        )
        report['scenario'] = dict(force_levels=list(self.env.cfg.assist_force_levels),
                                  leg_target_hold_after_s=getattr(getattr(self.env, '_diagnostic_leg_target_hold', None), 'after_seconds', None),
                                  birth_frames=self.birth_frames.tolist(),
                                  birth_mode='fixed_reference_frame_control' if bool((self.birth_frames != 0).any()) else 'frame_zero_evaluation',
                                  full_until_s=self.env.cfg.assist_full_until_s,
                                  zero_after_s=self.env.cfg.assist_zero_after_s,
                                  initial_motion_ids='cyclic enabled motion assignment; see rows')
        self.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if not self.complete or self.trace_printed:
            return
        print('\n[早期平衡时序] 各环境首回合前1.5秒，每0.1秒；脚力为当前采样。', flush=True)
        hold_time = report['scenario']['leg_target_hold_after_s']
        if hold_time is not None:
            print(f'[控制覆盖对照] {hold_time:.2f}秒后保持髋膝目标；关节记录为覆盖后的命令，不是原策略输出。', flush=True)
        print('水平速度按骨盆朝向：+X前/+Y左；角速度按完整骨盆坐标。俯仰正值为前倾，大倾角时结合总倾角。', flush=True)
        print('踝位置是相对骨盆的刚体原点，不是接触点或质心投影。', flush=True)
        for env_id in range(self.env.num_envs):
            rows = [r for r in self.trace if r['env_id'] == env_id and r['time_s'] <= 1.500001]
            if not rows: continue
            print(f"环境 {env_id} | 动作 {rows[0]['motion_id']} | 出生帧 {rows[0]['birth_frame']}", flush=True)
            for row in rows:
                vx, vy = row['horizontal_velocity_heading_m_s']
                wx, wy, _ = row['angular_velocity_b_rad_s']
                fr = row['feet_force_current_n']
                fz = [v[2] for v in row['feet_force_current_w_n']]
                ankle = row['ankle_position_relative_pelvis_b_m']
                print(f"  t={row['time_s']:.2f} | 高度 {row['height']:.3f}"
                      f" | 骨盆滚/俯 {row['root_rpy_deg'][0]:+.1f}/{row['root_rpy_deg'][1]:+.1f}°"
                      f" | 躯干滚/俯 {row['torso_rpy_deg'][0]:+.1f}/{row['torso_rpy_deg'][1]:+.1f}°"
                      f" | 总倾角 {row['root_tilt_deg']:.1f}/{row['torso_tilt_deg']:.1f}°", flush=True)
                print(f"    水平vx/vy {vx:+.3f}/{vy:+.3f} m/s | 滚/俯角速度 {wx:+.3f}/{wy:+.3f} rad/s"
                      f" | 当前脚力 {fr[0]:.0f}/{fr[1]:.0f} N | 世界Fz {fz[0]:+.0f}/{fz[1]:+.0f} N"
                      f" | 双脚接触 {all(v > 5 for v in fr)}", flush=True)
                print(f"    左踝相对xyz {ankle[0][0]:+.3f}/{ankle[0][1]:+.3f}/{ankle[0][2]:+.3f} m"
                      f" | 右踝 {ankle[1][0]:+.3f}/{ankle[1][1]:+.3f}/{ankle[1][2]:+.3f} m", flush=True)
        print('[早期平衡时序结束]', flush=True)
        print('\n[足部与关节对照] 首回合0.3～1.0秒；左右脚均为踝刚体，角度不等于足底倾角。', flush=True)
        print('世界坐标xyz；位置已减环境原点。接触时踝移动可能是滑动，也可能绕接触点转动。', flush=True)
        for env_id in range(self.env.num_envs):
            rows = [r for r in self.trace if r['env_id'] == env_id and .299999 <= r['time_s'] <= 1.000001]
            if not rows: continue
            print(f"环境 {env_id} | 动作 {rows[0]['motion_id']} | 出生帧 {rows[0]['birth_frame']}", flush=True)
            for row in rows:
                print(f"  t={row['time_s']:.2f} | 高度 {row['height']:.3f} m", flush=True)
                for i, label in enumerate(('左脚', '右脚')):
                    pos = '/'.join(f'{v:+.3f}' for v in row['ankle_position_w_m'][i])
                    vel = '/'.join(f'{v:+.3f}' for v in row['ankle_velocity_w_m_s'][i])
                    rpy = '/'.join(f'{v:+.1f}' for v in row['ankle_rpy_w_deg'][i])
                    j = row['joints'][('left' if i == 0 else 'right') + '_ankle_pitch_joint']
                    print(f"    {label} xyz {pos} m | 速度 {vel} m/s"
                          f" | 滚/俯/偏 {rpy}° | 当前地面力 {row['feet_force_current_n'][i]:.0f} N"
                          f" | 踝实际/目标 {j['position_rad']:+.3f}/{j['final_target_rad']:+.3f} rad", flush=True)
                for joint_type, label in (('hip_pitch', '髋'), ('knee', '膝'), ('ankle_pitch', '踝')):
                    left = row['joints']['left_' + joint_type + '_joint']
                    right = row['joints']['right_' + joint_type + '_joint']
                    print(f"    {label} 实际→最终目标：左 {left['position_rad']:+.3f}→{left['final_target_rad']:+.3f}"
                          f" | 右 {right['position_rad']:+.3f}→{right['final_target_rad']:+.3f} rad", flush=True)
        print('[足部与关节对照结束]', flush=True)
        print('\n[起身转换时序] 各环境首回合；概览双脚力取历史峰值，详细支撑力为当前采样。', flush=True)
        for env_id in range(self.env.num_envs):
            rows = [r for r in self.trace if r['env_id'] == env_id]
            if not rows: continue
            motion_id = rows[0]['motion_id']
            print(f"环境 {env_id} | 动作 {motion_id}: {Path(self.motion_paths[motion_id]).name}"
                  f" | 出生帧 {rows[0]['birth_frame']}", flush=True)
            for row in rows:
                second = row['time_s']
                if second > .021 and abs(second * 2-round(second * 2)) > 1e-5: continue
                knees = '/'.join(f'{v:.2f}' for v in row['knee_angles'])
                feet = '/'.join(f'{v:.0f}' for v in row['feet_force'])
                print(f"  t={second:5.2f} | 辅助 {row['force_n']:5.1f} N | 高度 {row['height']:.3f} m"
                      f" | 躯干 {row['torso_tilt_deg']:5.1f}° | 膝 {knees}"
                      f" | 双脚力 {feet} N | 前向速度 {row['velocity_b'][0]:+.3f} m/s"
                      f" | 目标超限 {row['raw_target_saturation_ratio']:.0%}", flush=True)
                if 2.0 <= second <= 6.0 and abs(second-round(second)) < 1e-5:
                    forces = row['support_force_current_n']
                    print('    当前地面力：' + ' | '.join(
                        f"{label} {forces['left_'+body+'_link']:.0f}/{forces['right_'+body+'_link']:.0f} N"
                        for label, body in (('手腕', 'wrist_roll'), ('膝', 'knee'), ('肘', 'elbow'))), flush=True)
                    for name, joint in row['joints'].items():
                        print(f"    {name}: 实际 {joint['position_rad']:+.3f} → 原目标 {joint['requested_target_rad']:+.3f}"
                              f" → 最终 {joint['final_target_rad']:+.3f} rad"
                              f" | 速度 {joint['velocity_rad_s']:+.3f}"
                              f" | 估算力矩 {joint['estimated_torque_nm']:+.1f}/{joint['effort_limit_nm']:.1f} Nm", flush=True)
                    print('    本步加权任务奖励：' + ' | '.join(
                        f'{name}={value:+.5f}' for name, value in row['task_reward_per_step'].items()), flush=True)
        print('力矩为隐式 PD 估算，不能单独证明执行器饱和；本次推理不计算 AMP 奖励。', flush=True)
        print('[起身转换时序结束] 0.1 秒完整采样及参考对照已写入 JSON。', flush=True)
        self.trace_printed = True

    def close(self):
        self.env.reward_manager.compute = self.original_compute
