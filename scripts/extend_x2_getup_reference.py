"""Append a foot-constrained stand-up candidate; never overwrite the source."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from convert_x2_reference import angular_velocity_world, export_mujoco_fk
from x2_motion_utils import load_motion


def blend(p0, p1, v0, t, duration):
    """Quintic with initial velocity v0 and zero final velocity/acceleration."""
    u = np.clip(t / duration, 0., 1.)
    delta, velocity = p1 - p0, v0 * duration
    return (p0 + velocity*u + (10*delta-6*velocity)*u**3
            + (-15*delta+8*velocity)*u**4 + (6*delta-3*velocity)*u**5)


def rotation_blend(r0, r1, omega_world, t, duration):
    return r0 * R.from_rotvec(blend(np.zeros(3), (r0.inv()*r1).as_rotvec(),
                                     r0.inv().apply(omega_world), t, duration))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rise', type=float, default=2.)
    p.add_argument('--hold', type=float, default=1.5)
    p.add_argument('--settle', type=float, default=.6)
    p.add_argument('--cut-time', type=float, default=None)
    a = p.parse_args()
    if a.input.resolve() == a.output.resolve() or a.output.exists():
        p.error('Output must be a new file; source and existing outputs are protected.')
    if not np.isfinite([a.rise, a.hold, a.settle]).all() or not 0 < a.settle <= a.rise or a.hold < .5:
        p.error('Require 0 < settle <= rise and hold >= 0.5 seconds.')
    src = load_motion(a.input)
    dt = float(src['metadata']['dt_s'])
    # 保留指定时刻之前的参考，从这里开始接入站直段。
    if a.cut_time is not None:
        if (
            not np.isfinite(a.cut_time)
            or a.cut_time < dt
            or a.cut_time > src["time_s"][-1]
        ):
            p.error("cut-time 必须位于原参考时间范围内")
        frame = round(a.cut_time / dt)              # 计算截取位置的帧索引。
        if not np.isclose(frame * dt, a.cut_time):
            p.error("cut-time 必须与参考帧时间对齐")
        original_count = len(src["time_s"])         # 记录截取之前的总帧数。
        for key, value in list(src.items()):
            if (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == original_count
            ):
                src[key] = value[:frame + 1].copy()
        src["metadata"] = dict(
            src["metadata"],
            cut_time_s=float(frame * dt),
        )
    for value in [a.rise, a.hold, a.settle]:
        if not np.isclose(value/dt, round(value/dt)):
            p.error('Durations must be integer multiples of the source dt.')
    xml = Path(src['metadata']['mujoco_xml'])
    m, d = mujoco.MjModel.from_xml_path(str(xml)), None
    d = mujoco.MjData(m)
    names = src['joint_names'].tolist()
    adr = np.array([int(m.joint(n).qposadr[0]) for n in names])
    limits = np.array([m.jnt_range[m.joint(n).id] for n in names])
    leg = np.array([i for i,n in enumerate(names) if any(s in n for s in ('_hip_', '_knee_', '_ankle_'))])
    upper = np.array([i for i in range(len(names)) if i not in leg])
    knees = [names.index(s+'_knee_joint') for s in ('left','right')]
    feet = [m.body(s+'_sole_reference').id for s in ('left','right')]
    foot_bodies = [m.body(s+'_ankle_roll_link').id for s in ('left','right')]
    root_id = m.body('pelvis').id
    if len(leg) != 12 or len(names) != 31:
        raise ValueError('This helper requires the checked X2 31-joint model.')

    def forward(pos, rot, q):
        d.qpos[:3] = pos
        d.qpos[3:7] = rot.as_quat()[[3,0,1,2]]
        d.qpos[adr] = q
        mujoco.mj_forward(m,d)

    def foot_state():
        return d.xpos[feet].copy(), [R.from_matrix(d.xmat[i].reshape(3,3)) for i in feet]

    q0 = src['joint_pos'][-1].astype(float)
    pos0 = src['root_pos_w'][-1].astype(float)
    rot0 = R.from_quat(src['root_quat_xyzw'][-1])
    forward(src['root_pos_w'][-2], R.from_quat(src['root_quat_xyzw'][-2]), src['joint_pos'][-2])
    feet_prev, rots_prev = foot_state()
    forward(pos0, rot0, q0)
    feet0, rots0 = foot_state()
    feet_velocity = (feet0-feet_prev)/dt
    feet_omega = [(r*rprev.inv()).as_rotvec()/dt for r,rprev in zip(rots0,rots_prev)]
    feet_end_rot = [R.from_euler('z',r.as_euler('xyz')[2]) for r in rots0]
    # Use the original foot mesh, expressed in the sole-reference frame.
    vertices = []
    for body, sole in zip(foot_bodies,feet):
        ids = np.where((m.geom_bodyid == body) & (m.geom_type == mujoco.mjtGeom.mjGEOM_MESH))[0]
        if not len(ids):
            raise ValueError('Foot mesh missing from frozen XML.')
        g = int(ids[0]); mesh = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
        world = R.from_matrix(d.geom_xmat[g].reshape(3,3)).apply(v)+d.geom_xpos[g]
        vertices.append(R.from_matrix(d.xmat[sole].reshape(3,3)).inv().apply(world-d.xpos[sole]))
    ground = float(src['metadata'].get('ground_z',0.))
    clearance0 = np.array([(r.apply(v)+pos)[:,2].min()-ground
                           for r,v,pos in zip(rots0,vertices,feet0)])
    clearance_prev = np.array([(r.apply(v)+pos)[:,2].min()-ground
                               for r,v,pos in zip(rots_prev,vertices,feet_prev)])
    clearance_velocity = (clearance0-clearance_prev)/dt
    feet_end = feet0.copy()
    for i in range(2):
        feet_end[i,2] = ground+.001-feet_end_rot[i].apply(vertices[i])[:,2].min()
    rot_end = R.from_euler('z',rot0.as_euler('xyz')[2])
    q_end = q0.copy()
    for i in upper:
        n = names[i]
        q_end[i] = (.2 if '_shoulder_pitch_' in n else .2 if n=='left_shoulder_roll_joint'
                    else -.2 if n=='right_shoulder_roll_joint' else -.3 if '_elbow_' in n else 0.)

    # First solve an upright endpoint with both feet constrained.
    low = np.r_[pos0[:2]-.3, ground+.61, limits[leg,0]+1e-5]
    high = np.r_[pos0[:2]+.3, ground+.85, limits[leg,1]-1e-5]
    def endpoint_error(x):
        q_end[leg] = x[3:]
        forward(x[:3],rot_end,q_end)
        e = list(((d.xpos[feet]-feet_end)*100).ravel())
        for i,r in zip(feet,feet_end_rot):
            e.extend((r.inv()*R.from_matrix(d.xmat[i].reshape(3,3))).as_rotvec()*20)
        e.extend((q_end[knees]-.3)*.05)
        e.extend((d.subtree_com[root_id,:2]-feet_end[:,:2].mean(0))*.5)
        return e
    initial = np.r_[pos0,q0[leg]]; initial[2] = ground+.65
    fit = least_squares(endpoint_error,np.clip(initial,low+1e-7,high-1e-7),bounds=(low,high),max_nfev=500)
    endpoint_error(fit.x)
    pos_end = fit.x[:3].copy(); q_end = q_end.copy()
    if not fit.success or abs(q_end[knees]).max() >= .45:
        raise RuntimeError('No acceptable endpoint; do not force export.')

    positions, rotations, joints = [], [], []
    errors, angle_errors = [], []
    x = np.r_[pos0,q0[leg]]
    lower = np.r_[pos0[:2]-.4,ground+.3,limits[leg,0]+1e-5]
    higher = np.r_[pos0[:2]+.4,ground+.9,limits[leg,1]-1e-5]
    for frame in range(1,round(a.rise/dt)+1):
        t = frame*dt
        rot = rotation_blend(rot0,rot_end,src['root_ang_vel_w'][-1],t,a.rise)
        desired_pos = blend(pos0,pos_end,src['root_lin_vel_w'][-1],t,a.rise)
        desired_q = blend(q0,q_end,src['joint_vel'][-1],t,a.rise)
        target_pos = blend(feet0,feet_end,feet_velocity,t,a.settle)
        target_rot = [rotation_blend(r0,r1,w,t,a.settle)
                      for r0,r1,w in zip(rots0,feet_end_rot,feet_omega)]
        clearance = blend(clearance0,np.full(2,.001),clearance_velocity,t,a.settle)
        for i in range(2):
            target_pos[i,2] = ground+clearance[i]-target_rot[i].apply(vertices[i])[:,2].min()
        q = desired_q.copy()
        def error(value):
            q[leg] = value[3:]
            forward(value[:3],rot,q)
            e = list(((d.xpos[feet]-target_pos)*100).ravel())
            for i,r in zip(feet,target_rot):
                e.extend((r.inv()*R.from_matrix(d.xmat[i].reshape(3,3))).as_rotvec()*20)
            e.append((value[2]-desired_pos[2])*50)
            e.extend((value[:2]-desired_pos[:2])*.1)
            e.extend((value[3:]-desired_q[leg])*.01)
            return e
        fit = least_squares(error,np.clip(x,lower+1e-7,higher-1e-7),bounds=(lower,higher),max_nfev=150,
                            ftol=1e-10,xtol=1e-10,gtol=1e-10)
        x = fit.x.copy(); error(x)
        pe = float(np.linalg.norm(d.xpos[feet]-target_pos,axis=1).max())
        ae = max(np.linalg.norm((r.inv()*R.from_matrix(d.xmat[i].reshape(3,3))).as_rotvec()) for i,r in zip(feet,target_rot))
        if not fit.success or pe > .001 or ae > np.deg2rad(.5):
            raise RuntimeError(f'IK failed at appended frame {frame}: position={pe}, angle={ae}')
        errors.append(pe); angle_errors.append(ae)
        positions.append(x[:3].copy()); rotations.append(rot.as_quat()); joints.append(q.copy())
    # Hold the actual final IK pose, avoiding a snap to a separately solved endpoint.
    hold = round(a.hold/dt)
    pos = np.vstack([src['root_pos_w'],positions,np.repeat(positions[-1][None],hold,axis=0)])
    quat = np.vstack([src['root_quat_xyzw'],rotations,np.repeat(rotations[-1][None],hold,axis=0)])
    joint = np.vstack([src['joint_pos'],joints,np.repeat(joints[-1][None],hold,axis=0)])
    for i in range(1,len(quat)):
        if np.dot(quat[i-1],quat[i]) < 0: quat[i] *= -1
    rot = R.from_quat(quat)
    lin = np.gradient(pos,dt,axis=0); ang = angular_velocity_world(rot,dt)
    vel = np.gradient(joint,dt,axis=0)
    ids = src['policy_joint_indices']
    metadata = dict(src['metadata'])
    metadata.update(status='synthetic_standing_extension_candidate_NOT_dynamics_validated',
                    source=str(a.input.resolve()),source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
                    duration_s=(len(pos)-1)*dt,extension_rise_s=a.rise,extension_hold_s=a.hold,
                    extension_start_frame=len(src['time_s']),loop=False,
                    interpolation='Quintic task-space boundary velocities + constrained leg IK; regenerated velocities and FK')
    out = dict(time_s=np.arange(len(pos))*dt,joint_names=np.array(names),policy_joint_names=src['policy_joint_names'],
               policy_joint_indices=ids,root_pos_w=pos,root_quat_xyzw=quat,root_lin_vel_w=lin,root_ang_vel_w=ang,
               root_lin_vel_b=rot.inv().apply(lin),root_ang_vel_b=rot.inv().apply(ang),
               projected_gravity_b=rot.inv().apply(np.tile([0.,0.,-1.],(len(pos),1))),
               joint_pos=joint,joint_vel=vel,policy_joint_pos=joint[:,ids],policy_joint_vel=vel[:,ids])
    out.update(export_mujoco_fk(xml,names,pos,quat,joint))
    for key,value in out.items():
        if value.dtype.kind == 'f' and key != 'time_s': out[key] = value.astype(np.float32)
    out['metadata_json'] = np.array(json.dumps(metadata,ensure_ascii=False))
    report = dict(status=metadata['status'],frames=len(pos),duration_s=metadata['duration_s'],
                  original_frames_unchanged=bool(np.array_equal(out['joint_pos'][:len(src['time_s'])],src['joint_pos'])
                    and np.array_equal(out['root_pos_w'][:len(src['time_s'])],src['root_pos_w'])
                    and np.allclose(out['root_quat_xyzw'][:len(src['time_s'])],src['root_quat_xyzw'])),
                  endpoint_height=float(pos[-1,2]-ground),endpoint_knees=joint[-1,knees].tolist(),
                  appended_foot_position_error_max_m=max(errors),appended_foot_angle_error_max_deg=float(np.rad2deg(max(angle_errors))),
                  original_endpoint_foot_clearance_m=clearance0.tolist(),
                  seam_joint_step_max_rad=float(abs(joint[len(src['time_s'])]-joint[len(src['time_s'])-1]).max()),
                  appended_joint_speed_max_rad_s=float(abs(vel[len(src['time_s'])-1:]).max()),
                  hold_velocity_max=float(max(abs(vel[-hold+1:]).max(),abs(lin[-hold+1:]).max(),abs(ang[-hold+1:]).max())),
                  notes=['No self-collision or dynamic-balance pass is asserted.','Do not turn the last-to-first boundary into an expert transition.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**out)
    load_motion(a.output)
    a.output.with_suffix('.extension.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2)); print('[SAVED]',a.output.resolve())


if __name__ == '__main__': main()
