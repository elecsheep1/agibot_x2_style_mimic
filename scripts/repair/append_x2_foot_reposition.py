"""Append an IK-based weight-shift and two-step stance reduction candidate.

This exports kinematic reference states, not a balance controller or measured contacts.
Run with the gmr Python environment. Existing inputs and outputs are protected.
"""

# Support direct execution: python scripts/<category>/<tool>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from retarget.convert_x2_reference import angular_velocity_world, export_mujoco_fk
from common.x2_motion_utils import load_motion


def smooth(u):
    return u**3 * (10. + u * (-15. + 6.*u))


def interpolate(a, b, u):
    return a + (b-a)*smooth(u)


def footprint_margin(points, point):
    hull = ConvexHull(points[:, :2])
    return float(-(hull.equations[:, :2]@point[:2]+hull.equations[:, 2]).max())


def build(args):
    src, final = load_motion(args.input), load_motion(args.stance)
    dt = float(src['metadata']['dt_s'])
    if not np.isclose(dt, final['metadata']['dt_s']):
        raise ValueError('Input and final stance must have equal dt.')
    durations = [args.shift_time, args.step_time, args.transfer_time, args.step_time, args.shift_time]
    if not np.isfinite(durations+[args.hold, args.lift]).all() or min(durations+[args.hold]) <= 0 or args.lift <= 0:
        raise ValueError('All durations and foot lift must be finite and positive.')
    for duration in durations+[args.hold]:
        if not np.isclose(duration/dt, round(duration/dt)):
            raise ValueError('Durations must be multiples of the reference dt.')
    xml = Path(src['metadata']['mujoco_xml']).resolve()
    if xml != Path(final['metadata']['mujoco_xml']).resolve():
        raise ValueError('References must use the same frozen robot XML.')
    names = src['joint_names'].tolist()
    if set(names) != set(final['joint_names'].tolist()):
        raise ValueError('Joint sets do not match.')
    final_order = [final['joint_names'].tolist().index(n) for n in names]
    original_hold = float(src['metadata']['extension_hold_s'])
    last = len(src['time_s'])-round(original_hold/dt)-1
    if last < 2:
        raise ValueError('Cannot locate the start of the original final hold.')
    if (np.max(abs(src['joint_pos'][last:]-src['joint_pos'][last])) > 1e-6
            or np.max(abs(src['root_pos_w'][last:]-src['root_pos_w'][last])) > 1e-6):
        raise ValueError('The removed suffix must be the stationary final hold.')
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    addr = np.array([int(m.joint(n).qposadr[0]) for n in names])
    legs = [i for i,n in enumerate(names) if any(k in n for k in ('_hip_', '_knee_', '_ankle_'))]
    waist = [names.index('waist_'+axis+'_joint') for axis in ('yaw', 'pitch', 'roll')]
    selected = legs+waist
    if len(legs) != 12 or len(names) != 31:
        raise ValueError('Expected the checked 31-joint X2 robot.')
    feet = [m.body(s+'_sole_reference').id for s in ('left', 'right')]
    foot_bodies = [m.body(s+'_ankle_roll_link').id for s in ('left', 'right')]
    torso, pelvis = m.body('torso_link').id, m.body('pelvis').id
    q_start = src['joint_pos'][last].astype(float)
    q_end = final['joint_pos'][-1, final_order].astype(float)
    q = q_start.copy()
    p_start = src['root_pos_w'][last].astype(float)
    p_end = final['root_pos_w'][-1].astype(float)
    initial_rotation = R.from_quat(src['root_quat_xyzw'][last])
    yaw = initial_rotation.as_euler('xyz')[2]
    heading = R.from_euler('z', yaw)
    if np.linalg.norm((heading.inv()*initial_rotation).as_rotvec()) > 1e-4:
        raise ValueError('Start of the reposition stage must be upright.')
    if np.linalg.norm((heading.inv()*R.from_quat(final['root_quat_xyzw'][-1])).as_rotvec()) > 1e-4:
        raise ValueError('Final stance must have the same upright root heading.')
    ground = float(src['metadata'].get('ground_z', 0.))

    def forward(value):
        q[selected] = value[5:]
        rotation = R.from_euler('xyz', [value[3], value[4], yaw])
        d.qpos[:3] = value[:3]
        d.qpos[3:7] = rotation.as_quat()[[3,0,1,2]]
        d.qpos[addr] = q
        # The optimizer only needs FK and mass-weighted subtree COM, not contacts.
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        return rotation

    x_start = np.r_[p_start, 0., 0., q_start[selected]]
    x_end = np.r_[p_end, 0., 0., q_end[selected]]
    forward(x_end)
    end_feet = d.xpos[feet].copy()
    end_com = d.subtree_com[pelvis].copy()
    forward(x_start)
    start_feet = d.xpos[feet].copy()
    start_com = d.subtree_com[pelvis].copy()
    foot_rotations = [R.from_matrix(d.xmat[i].reshape(3,3)) for i in feet]
    vertices, pivots = [], []
    for side, (body, sole) in enumerate(zip(foot_bodies, feet)):
        geoms = np.where((m.geom_bodyid == body) & (m.geom_type == mujoco.mjtGeom.mjGEOM_MESH))[0]
        if not len(geoms):
            raise ValueError('Original foot mesh missing.')
        g = int(geoms[0]); mesh = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
        w = R.from_matrix(d.geom_xmat[g].reshape(3,3)).apply(v)+d.geom_xpos[g]
        v = foot_rotations[side].inv().apply(w-d.xpos[sole])
        vertices.append(v)
        roll = .25 if side == 0 else -.25
        turned = R.from_euler('x', roll).apply(v)
        pivots.append(v[np.argmin(turned[:,2])].copy())
    for a,b in zip(foot_rotations, feet):
        if np.linalg.norm(a.as_euler('xyz')[:2]) > 1e-4:
            raise ValueError('Both initial feet must already be flat.')

    def foot_pose(side, anchor, roll, lift=0.):
        rotation = foot_rotations[side]*R.from_euler('x', roll)
        # Keep the selected edge pivot's XY fixed while the grounded foot rolls.
        position = anchor+foot_rotations[side].apply(pivots[side])-rotation.apply(pivots[side])
        position[2] = ground+.001+lift-rotation.apply(vertices[side])[:,2].min()
        return position, rotation

    left_support = start_feet[0].copy()-heading.apply([0., .04, 0.])
    right_support = end_feet[1].copy()+heading.apply([0., .04, 0.])
    # Validate the intended single-support COM targets against the flat sole mesh.
    for side, anchor, point in [(0,start_feet[0],left_support),(1,end_feet[1],right_support)]:
        world = foot_rotations[side].apply(vertices[side])+anchor
        bottom = world[world[:,2] < world[:,2].min()+.003]
        if footprint_margin(bottom, point) < .005:
            raise ValueError('Single-support COM target is too close to the sole boundary.')
    limits = np.array([m.jnt_range[m.joint(names[i]).id] for i in selected])
    limits[-3:] = [[-.2,.2],[-.25,.25],[-.3,.3]]
    # Avoid the inward roll region where the hip housing approaches the pelvis.
    limits[selected.index(names.index('left_hip_roll_joint')),0] = 0.
    limits[selected.index(names.index('right_hip_roll_joint')),1] = 0.
    low = np.r_[p_start[:2]-.45,ground+.50,-.25,-.15,limits[:,0]+1e-5]
    high = np.r_[p_start[:2]+.45,ground+.76,.25,.15,limits[:,1]-1e-5]
    phases = ['shift_left_unload_right','right_foot_swing','transfer_right_unload_left',
              'left_foot_swing','recenter_and_flatten']
    x = x_start.copy()
    samples, positions, rotations, joint_states, coms = [], [], [], [], []
    offsets = [0.]+list(np.cumsum(durations))
    mid_height = ground+.59
    for phase, duration in enumerate(durations):
        for frame in range(1, round(duration/dt)+1):
            u = frame*dt/duration
            anchors = start_feet.copy()
            rolls, lifts = [0.,0.], [0.,0.]
            if phase == 0:
                com_target = interpolate(start_com,left_support,u)
                rolls[1] = -.25*smooth(u)
                desired_height = float(interpolate(p_start[2],mid_height,u))
                support = [0,1]
            elif phase == 1:
                anchors[1] = interpolate(start_feet[1],end_feet[1],u)
                rolls[1] = float(interpolate(-.25,-.18,u))
                lifts[1] = args.lift*64.*u**3*(1.-u)**3
                com_target = left_support
                desired_height = mid_height
                support = [0] if frame < round(duration/dt) else [0,1]
            elif phase == 2:
                anchors[1] = end_feet[1]
                rolls = [.25*smooth(u),-.18*(1.-smooth(u))]
                com_target = interpolate(left_support,right_support,u)
                desired_height = mid_height
                support = [0,1]
            elif phase == 3:
                anchors[1] = end_feet[1]
                anchors[0] = interpolate(start_feet[0],end_feet[0],u)
                rolls[0] = float(interpolate(.25,.18,u))
                lifts[0] = args.lift*64.*u**3*(1.-u)**3
                com_target = right_support
                desired_height = mid_height
                support = [1] if frame < round(duration/dt) else [0,1]
            else:
                anchors = end_feet.copy()
                rolls[0] = .18*(1.-smooth(u))
                com_target = interpolate(right_support,end_com,u)
                desired_height = float(interpolate(mid_height,p_end[2],u))
                support = [0,1]
            planned = [foot_pose(i,anchors[i],rolls[i],lifts[i]) for i in range(2)]
            foot_targets = np.array([item[0] for item in planned])
            foot_rot = [item[1] for item in planned]
            global_u = (offsets[phase]+frame*dt)/sum(durations)
            template = interpolate(x_start,x_end,global_u)
            previous = x.copy()

            def error(value):
                forward(value)
                e = list(((d.xpos[feet]-foot_targets)*300.).ravel())
                for i,r in zip(feet,foot_rot):
                    e.extend((r.inv()*R.from_matrix(d.xmat[i].reshape(3,3))).as_rotvec()*50.)
                e.extend((d.subtree_com[pelvis,:2]-com_target[:2])*100.)
                e.append((value[2]-desired_height)*5.)
                e.extend(value[3:5]*2.)
                e.extend(q[waist])
                e.extend((value[5:17]-template[5:17])*.03)
                # Warm starts alone do not prevent jumps between redundant IK solutions.
                e.extend((value-previous)*4.)
                if phase == 4:
                    e.extend((value-x_end)*10.*smooth(u))
                return e

            fit = least_squares(error,np.clip(x,low+1e-7,high-1e-7),bounds=(low,high),
                                max_nfev=180,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            x = fit.x.copy(); error(x)
            pe = float(np.linalg.norm(d.xpos[feet]-foot_targets,axis=1).max())
            ae = float(max(np.linalg.norm((r.inv()*R.from_matrix(d.xmat[i].reshape(3,3))).as_rotvec())
                           for i,r in zip(feet,foot_rot)))
            ce = float(np.linalg.norm(d.subtree_com[pelvis,:2]-com_target[:2]))
            world = [R.from_matrix(d.xmat[i].reshape(3,3)).apply(v)+d.xpos[i] for i,v in zip(feet,vertices)]
            clearance = [float(w[:,2].min()-ground) for w in world]
            contact_points = np.vstack([world[i][world[i][:,2]<world[i][:,2].min()+.003] for i in support])
            margin = footprint_margin(contact_points,d.subtree_com[pelvis])
            torso_cos = float(d.xmat[torso].reshape(3,3)[2,2])
            record = dict(phase=phases[phase],time_s=float(src['time_s'][last]+offsets[phase]+frame*dt),
                foot_position_error_m=pe,foot_angle_error_deg=float(np.rad2deg(ae)),com_target_error_m=ce,
                foot_clearance_m=clearance,intended_support=support,com_projected_margin_m=margin,
                torso_tilt_deg=float(np.rad2deg(np.arccos(np.clip(torso_cos,-1,1)))))
            if not fit.success or pe>.001 or ae>np.deg2rad(.5) or ce>.005 or min(clearance)<-.0005:
                raise RuntimeError('IK/clearance check failed: '+json.dumps(record))
            if len(support)==1 and margin<.003:
                raise RuntimeError('Single-support geometric COM check failed: '+json.dumps(record))
            samples.append(record)
            positions.append(x[:3].copy())
            rotations.append(R.from_euler('xyz',[x[3],x[4],yaw]).as_quat())
            joint_states.append(q.copy());coms.append(d.subtree_com[pelvis].copy())
        print('[PHASE]',phases[phase],json.dumps(samples[-1]),flush=True)

    # Match the existing narrow stance, rather than invent another final stance.
    pose_error = float(np.max(abs(joint_states[-1]-q_end)))
    root_error = float(np.linalg.norm(positions[-1]-p_end))
    if pose_error>.003 or root_error>.001:
        raise RuntimeError(f'Final stance mismatch: joint={pose_error}, root={root_error}')
    positions[-1], rotations[-1], joint_states[-1] = p_end.copy(), heading.as_quat(), q_end.copy()
    hold = round(args.hold/dt)
    prefix = last+1
    pos = np.vstack([src['root_pos_w'][:prefix],positions,np.repeat(p_end[None],hold,axis=0)])
    quat = np.vstack([src['root_quat_xyzw'][:prefix],rotations,np.repeat(heading.as_quat()[None],hold,axis=0)])
    joints = np.vstack([src['joint_pos'][:prefix],joint_states,np.repeat(q_end[None],hold,axis=0)])
    for i in range(1,len(quat)):
        if np.dot(quat[i-1],quat[i])<0: quat[i]*=-1
    root_rotation = R.from_quat(quat)
    linear = np.gradient(pos,dt,axis=0)
    angular = angular_velocity_world(root_rotation,dt)
    velocity = np.gradient(joints,dt,axis=0)
    max_speed = float(abs(velocity[last:]).max())
    if max_speed>4.:
        raise RuntimeError(f'Reposition joint speed {max_speed:.3f} rad/s exceeds the candidate 4 rad/s limit.')
    ids = src['policy_joint_indices']
    metadata = dict(src['metadata'])
    metadata.update(status='foot_reposition_candidate_NOT_dynamics_validated',duration_s=(len(pos)-1)*dt,
        source=str(args.input.resolve()),source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        stance_source=str(args.stance.resolve()),stance_sha256=hashlib.sha256(args.stance.read_bytes()).hexdigest(),
        reposition_start_s=float(src['time_s'][last]),reposition_duration_s=sum(durations),
        final_hold_s=args.hold,phase_names=phases,phase_durations_s=durations,loop=False,
        interpolation='Quintic weight transfer, edge roll, sequential airborne foot reposition, whole-body IK; regenerated velocities/FK',
        contact_labels='Intended geometric support only; no measured contact forces or dynamic balance certification')
    out = dict(time_s=np.arange(len(pos))*dt,joint_names=src['joint_names'],policy_joint_names=src['policy_joint_names'],
        policy_joint_indices=ids,root_pos_w=pos,root_quat_xyzw=quat,root_lin_vel_w=linear,root_ang_vel_w=angular,
        root_lin_vel_b=root_rotation.inv().apply(linear),root_ang_vel_b=root_rotation.inv().apply(angular),
        projected_gravity_b=root_rotation.inv().apply(np.tile([0.,0.,-1.],(len(pos),1))),
        joint_pos=joints,joint_vel=velocity,policy_joint_pos=joints[:,ids],policy_joint_vel=velocity[:,ids])
    out.update(export_mujoco_fk(xml,names,pos,quat,joints))
    for key,value in out.items():
        if value.dtype.kind=='f' and key!='time_s': out[key]=value.astype(np.float32)
    out['metadata_json'] = np.array(json.dumps(metadata,ensure_ascii=False))
    report = dict(status=metadata['status'],frames=len(pos),duration_s=metadata['duration_s'],
        preserved_prefix_end_s=float(src['time_s'][last]),preserved_prefix_frames=prefix,
        reposition_duration_s=sum(durations),phase_names=phases,phase_durations_s=durations,
        final_hold_s=args.hold,reposition_joint_speed_max_rad_s=max_speed,
        foot_position_error_max_m=max(s['foot_position_error_m'] for s in samples),
        foot_angle_error_max_deg=max(s['foot_angle_error_deg'] for s in samples),
        com_target_error_max_m=max(s['com_target_error_m'] for s in samples),
        single_support_com_margin_min_m=min(s['com_projected_margin_m'] for s in samples if len(s['intended_support'])==1),
        torso_tilt_max_deg=max(s['torso_tilt_deg'] for s in samples),
        final_stance_joint_error_before_snap_rad=pose_error,final_root_error_before_snap_m=root_error,
        kinematic_checks_passed=True,dynamics_validated=False,self_collision_checked=False,samples=samples)
    return out,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--stance',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shift-time',type=float,default=.9)
    p.add_argument('--step-time',type=float,default=.8)
    p.add_argument('--transfer-time',type=float,default=1.2)
    p.add_argument('--hold',type=float,default=1.5)
    p.add_argument('--lift',type=float,default=.045)
    args=p.parse_args()
    if args.output.exists() or args.output.with_suffix('.reposition.json').exists() or args.output.resolve() in {args.input.resolve(),args.stance.resolve()}:
        p.error('Choose a new output; inputs and existing outputs are protected.')
    result,report=build(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**result)
    load_motion(args.output)
    args.output.with_suffix('.reposition.json').write_text(json.dumps(report,indent=2)+'\n')
    print('[SAVED]',args.output.resolve(),flush=True)
    print('[SUMMARY]',json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)


if __name__=='__main__':main()
