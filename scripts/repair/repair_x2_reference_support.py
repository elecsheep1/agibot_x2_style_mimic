"""Offline UMR stance-foot locking; outputs a new kinematic candidate, not a policy.

An explicit support start is required for each foot. Before support, the source
trajectory is preserved. During a smooth transition the feet move to fixed flat
sole poses. Both legs and root translation can adjust under native collision and
joint-limit constraints. Does not infer forces or validate dynamic balance.
"""

# Support direct execution: python scripts/<category>/<tool>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import clarabel
import mujoco
import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation, Slerp

from repair.repair_x2_reference_feet import geometry, lowest_point
from retarget.convert_x2_reference import export_mujoco_fk
from common.x2_motion_utils import load_motion


def solve(source, output, collision_report, starts, transition=.6, max_iterations=18, knee_floor=0., smooth_interval=None, smooth_strength=0., anchor_report=None, edit_interval=None, root_trajectory_smoothing=False):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    payload=load_motion(source);meta=payload.pop('metadata');dt=float(meta['dt_s'])
    model=mujoco.MjModel.from_xml_path(meta['mujoco_xml']);data=mujoco.MjData(model)
    names=payload['joint_names'].tolist();addresses=[int(model.joint(n).qposadr[0]) for n in names]
    leg_names=[n for n in names if any(k in n for k in ('hip_','knee_','ankle_'))]
    qcols=[int(model.joint(n).qposadr[0]) for n in leg_names]+[0,1,2]
    dofs=[int(model.joint(n).dofadr[0]) for n in leg_names]+[0,1,2]
    q=np.zeros((len(payload['time_s']),model.nq));q[:,:3]=payload['root_pos_w'];q[:,3:7]=payload['root_quat_xyzw'][:,[3,0,1,2]];q[:,addresses]=payload['joint_pos'];original=q.copy()
    frames=len(q);width=len(qcols);size=frames*width
    fixed_anchors=json.loads(Path(anchor_report).read_text())['anchors'] if anchor_report else None
    time=payload['time_s'];active=np.clip((time-min(starts))/transition,0,1);active=active**2*(3-2*active)
    if edit_interval is not None:
        fade=np.minimum(np.clip((time-edit_interval[0])/.25,0,1),np.clip((edit_interval[1]-time)/.25,0,1))
        active*=fade**3*(fade*(fade*6-15)+10)
    geoms,vertices=geometry(model)
    audit=json.loads(Path(collision_report).read_text())
    blocked={tuple(sorted(pair)) for key in ('excluded_adjacent_pairs','excluded_authored_filters') for pair in audit[key]}
    def bname(g):
        n=model.body(int(model.geom_bodyid[g])).name
        return 'base_link' if n=='pelvis' else n
    pairs=[(a,b) for a,b in itertools.combinations(geoms,2) if model.geom_bodyid[a]!=model.geom_bodyid[b] and tuple(sorted((bname(a),bname(b)))) not in blocked]
    target_positions=[];target_rotations=[];weights=[];foot_ids=[];anchors=[]
    for side,start in zip(('left','right'),starts):
        body=int(model.body(side+'_ankle_roll_link').id);foot_ids.append(body)
        poses=[];rots=[]
        for pose in original:
            data.qpos[:]=pose;mujoco.mj_forward(model,data);poses.append(data.xpos[body].copy());rots.append(data.xmat[body].reshape(3,3).copy())
        poses=np.array(poses);rotations=Rotation.from_matrix(np.array(rots))
        # Anchor near the first flattened posture, avoiding later sliding samples.
        f0=int(round(start/dt));f1=min(frames-1,int(round((start+.3)/dt)))
        anchor=poses[f0:f1+1].mean(0);yaw=float(np.angle(np.mean(np.exp(1j*rotations[f0:f1+1].as_euler('xyz')[:,2]))))
        target_rotation=Rotation.from_euler('z',yaw)
        foot_geoms=[g for g in geoms if model.geom_bodyid[g]==body]
        # Compute the actual collision sole offset in this flat orientation.
        data.qpos[:]=original[f1];mujoco.mj_forward(model,data)
        offsets=[]
        for g in foot_geoms:
            if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_SPHERE:raise ValueError('Expected X2 spherical sole samples')
            offsets.append((target_rotation.as_matrix()@model.geom_pos[g])[2]-model.geom_size[g,0])
        anchor[2]=float(meta['ground_z'])+.002-min(offsets)
        if fixed_anchors is not None:
            saved=next(item for item in fixed_anchors if item['side']==side)
            if abs(saved['start_s']-start)>1e-9:raise ValueError('Anchor schedule differs')
            anchor=np.array(saved['ankle_position_w']);yaw=saved['yaw_rad'];target_rotation=Rotation.from_euler('z',yaw)
        w=np.clip((time-start)/transition,0,1);w=w*w*(3-2*w)
        targets=poses*(1-w[:,None])+anchor*w[:,None]
        target_rots=[]
        for f in range(frames):
            if w[f]==0:target_rots.append(rotations[f].as_matrix())
            elif w[f]==1:target_rots.append(target_rotation.as_matrix())
            else:target_rots.append(Slerp([0,1],Rotation.from_matrix([rots[f],target_rotation.as_matrix()]))([w[f]]).as_matrix()[0])
        target_positions.append(targets);target_rotations.append(Rotation.from_matrix(target_rots));weights.append(w)
        anchors.append({'side':side,'start_s':start,'fully_locked_s':start+transition,'ankle_position_w':anchor.tolist(),'yaw_rad':yaw})
    D2=sp.diags([np.ones(frames-2),-2*np.ones(frames-2),np.ones(frames-2)],[0,1,2],shape=(frames-2,frames)).tocsc()
    tracking=np.r_[np.ones(width-3),np.full(3,40.)]
    smooth=np.r_[np.full(width-3,40.),np.full(3,1000.)]
    S=sp.kron(D2.T@D2,sp.diags(smooth))
    if smooth_interval is not None:
        start,end=smooth_interval
        local=np.where((time[1:-1]>=start)&(time[1:-1]<=end),np.sin(np.pi*np.clip((time[1:-1]-start)/(end-start),0,1))**2,0.)
        S=S+sp.kron(D2.T@sp.diags(local)@D2,sp.diags(np.r_[np.full(width-3,smooth_strength),np.full(3,smooth_strength*10.)]))
    Hbase=sp.kron(sp.eye(frames),sp.diags(tracking))+S
    signal=original[:,qcols].copy()
    if not root_trajectory_smoothing:signal[:,-3:]=0
    gbase=np.asarray(S@signal.reshape(-1))
    lo=np.array([model.joint(n).range[0] for n in leg_names]+[-np.inf]*3)
    hi=np.array([model.joint(n).range[1] for n in leg_names]+[np.inf]*3)
    delta=np.zeros(size);records=[]
    for iteration in range(max_iterations):
        rows=[];cols=[];vals=[];rhs=[];blocks=[];linear=gbase.copy();taskrows=[];taskcols=[];taskvals=[];taskrhs=[]
        min_ground=np.inf;min_self=0;foot_error=0
        def constraint(f,jac,distance,margin):
            j=np.asarray(jac)[dofs]
            if np.linalg.norm(j)<1e-10:return
            row=len(rhs);nz=np.flatnonzero(abs(j)>1e-12)
            rows.extend([row]*len(nz));cols.extend((f*width+nz).tolist());vals.extend((-j[nz]).tolist())
            rhs.append(distance-margin-float(j@delta[f*width:(f+1)*width]))
        for f in range(frames):
            data.qpos[:]=q[f];mujoco.mj_forward(model,data)
            for side,body in enumerate(foot_ids):
                w=weights[side][f]
                if w<=0:continue
                jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jacBody(model,data,jp,jr,body)
                J=np.vstack([jp[:,dofs],jr[:,dofs]])
                error=np.r_[target_positions[side][f]-data.xpos[body],(target_rotations[side][f]*Rotation.from_matrix(data.xmat[body].reshape(3,3)).inv()).as_rotvec()]
                if w>.999:foot_error=max(foot_error,float(abs(error).max()))
                # A high fixed-pose objective, with contact smoothly activated.
                scale=np.sqrt(np.r_[np.full(3,1e7),np.full(3,1e5)]*w)
                J*=scale[:,None];goal=(error+np.vstack([jp[:,dofs],jr[:,dofs]])@delta[f*width:(f+1)*width])*scale
                for axis in range(6):
                    row=len(taskrhs);nz=np.flatnonzero(abs(J[axis])>1e-12)
                    taskrows.extend([row]*len(nz));taskcols.extend((f*width+nz).tolist());taskvals.extend(J[axis,nz].tolist());taskrhs.append(goal[axis])
            if active[f]<=0:continue
            for geom in geoms:
                point=lowest_point(model,data,geom,vertices);height=float(point[2])-meta['ground_z'];min_ground=min(min_ground,height)
                if height<.03:
                    j=np.zeros((3,model.nv));mujoco.mj_jac(model,data,j,None,point,int(model.geom_bodyid[geom]));constraint(f,j[2],height,.0018)
            for a,b in pairs:
                # Conservative bounding-sphere cull before exact convex distance.
                if np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b])>model.geom_rbound[a]+model.geom_rbound[b]+.012:continue
                points=np.zeros(6);distance=float(mujoco.mj_geomDistance(model,data,a,b,.012,points));min_self=min(min_self,distance)
                if distance<.01:
                    diff=points[:3]-points[3:];norm=np.linalg.norm(diff)
                    if norm<1e-12:continue
                    normal=diff/norm*(1 if distance>=0 else -1);ja=np.zeros((3,model.nv));jb=ja.copy()
                    mujoco.mj_jac(model,data,ja,None,points[:3],int(model.geom_bodyid[a]));mujoco.mj_jac(model,data,jb,None,points[3:],int(model.geom_bodyid[b]));constraint(f,normal@(ja-jb),distance,0.)
        T=sp.coo_matrix((taskvals,(taskrows,taskcols)),shape=(len(taskrhs),size)).tocsc()
        H=Hbase+T.T@T;linear-=np.asarray(T.T@np.array(taskrhs))
        lower=np.maximum(np.tile(np.r_[np.full(width-3,-.9),np.full(3,-.20)],(frames,1))*active[:,None],lo[None,:]-original[:,qcols]).reshape(-1)
        upper=np.minimum(np.tile(np.r_[np.full(width-3,.9),np.full(3,.20)],(frames,1))*active[:,None],hi[None,:]-original[:,qcols]).reshape(-1)
        trust=np.tile(np.r_[np.full(width-3,.15),np.full(3,.025)],frames)
        if knee_floor>0:
            low2=lower.reshape(frames,width)
            ramp=np.clip((time-(max(starts)+transition-.3))/.6,0,1);ramp=ramp*ramp*(3-2*ramp)
            for k,n in enumerate(leg_names):
                if 'knee' in n:
                    reachable=np.minimum(knee_floor*ramp,original[:,qcols[k]]+delta.reshape(frames,width)[:,k]+.12)
                    low2[:,k]=np.maximum(low2[:,k],reachable-original[:,qcols[k]])
            lower=low2.reshape(-1)
        lower=np.maximum(lower,delta-trust);upper=np.minimum(upper,delta+trust)
        A=sp.coo_matrix((vals,(rows,cols)),shape=(len(rhs),size)).tocsc();A=sp.vstack([A,sp.eye(size),-sp.eye(size)],format='csc');bvec=np.r_[rhs,upper,-lower]
        # Eliminate frames held exactly fixed instead of presenting duplicate
        # zero-width box constraints to the conic solver.
        free=np.flatnonzero(np.repeat(active>1e-9,width))
        reduced_A=A[:,free].tocsc();nonzero=np.asarray(abs(reduced_A).sum(axis=1)).reshape(-1)>1e-14
        if np.any(bvec[~nonzero]<-1e-6):raise RuntimeError('Fixed frame violates geometry/limits')
        reduced_A=reduced_A[nonzero];reduced_b=bvec[nonzero]
        settings=clarabel.DefaultSettings();settings.verbose=False
        solution=clarabel.DefaultSolver(sp.triu(H[free][:,free]/1e5,format='csc'),linear[free]/1e5,reduced_A,reduced_b,[clarabel.NonnegativeConeT(len(reduced_b))],settings).solve()
        if str(solution.status) not in ('Solved','AlmostSolved'):raise RuntimeError(f'Stance QP failed: {solution.status}')
        new=np.zeros(size);new[free]=np.asarray(solution.x)
        # Linearized collision constraints alone do not keep a large nonlinear
        # IK step feasible. Backtrack against actual native geometry before
        # accepting it, so the next QP never starts inside a collision.
        accepted=False
        for backtrack in range(12):
            trial_delta=delta+(new-delta)*(0.5**backtrack)
            trial=original.copy();trial[:,qcols]+=trial_delta.reshape(frames,width)
            feasible=True
            for f in np.flatnonzero(active>1e-9):
                data.qpos[:]=trial[f];mujoco.mj_forward(model,data)
                if min(lowest_point(model,data,g,vertices)[2]-meta['ground_z'] for g in geoms)<.0017:
                    feasible=False;break
                for a,b in pairs:
                    if np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b])>model.geom_rbound[a]+model.geom_rbound[b]+.001:continue
                    if float(mujoco.mj_geomDistance(model,data,a,b,.001,None))<-.00005:
                        feasible=False;break
                if not feasible:break
            if feasible:
                new=trial_delta;accepted=True;break
        if not accepted:raise RuntimeError('No collision-feasible support step')
        change=float(abs(new-delta).max());delta=new;q=original.copy();q[:,qcols]+=delta.reshape(frames,width)
        record={'iteration':iteration,'min_ground_m':min_ground,'min_self_m':min_self,'max_locked_pose_error':foot_error,'max_update':change};records.append(record);print(record,flush=True)
        if iteration>2 and change<1e-5:break
        if iteration>=15 and foot_error<.001 and change<.003:break
    payload['joint_pos']=q[:,addresses].astype(np.float32);payload['root_pos_w']=q[:,:3].astype(np.float32)
    payload['joint_vel']=np.gradient(payload['joint_pos'],dt,axis=0).astype(np.float32);payload['root_lin_vel_w']=np.gradient(payload['root_pos_w'],dt,axis=0).astype(np.float32)
    payload['root_lin_vel_b']=Rotation.from_quat(payload['root_quat_xyzw']).inv().apply(payload['root_lin_vel_w']).astype(np.float32)
    ids=payload['policy_joint_indices'];payload['policy_joint_pos']=payload['joint_pos'][:,ids];payload['policy_joint_vel']=payload['joint_vel'][:,ids]
    payload.update(export_mujoco_fk(meta['mujoco_xml'],names,payload['root_pos_w'],payload['root_quat_xyzw'],payload['joint_pos']))
    min_ground=np.inf;min_self=0.;stats={}
    heights=[[],[]];positions=[[],[]];tilts=[[],[]]
    for pose in q:
        data.qpos[:]=pose;mujoco.mj_forward(model,data)
        min_ground=min(min_ground,min(lowest_point(model,data,g,vertices)[2]-meta['ground_z'] for g in geoms))
        for side,body in enumerate(foot_ids):
            group=[g for g in geoms if model.geom_bodyid[g]==body];heights[side].append(min(lowest_point(model,data,g,vertices)[2]-meta['ground_z'] for g in group));positions[side].append(data.xpos[body].copy());tilts[side].append(np.degrees(np.arccos(np.clip(data.xmat[body].reshape(3,3)[2,2],-1,1))))
        for a,b in pairs:
            if np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b])>model.geom_rbound[a]+model.geom_rbound[b]+.001:continue
            min_self=min(min_self,float(mujoco.mj_geomDistance(model,data,a,b,.001,None)))
    for side,anchor in enumerate(anchors):
        mask=time>=anchor['fully_locked_s'];pos=np.array(positions[side]);h=np.array(heights[side]);v=np.linalg.norm(np.diff(pos[:,:2],axis=0)/dt,axis=1);maskv=mask[:-1]&mask[1:]
        stats[anchor['side']]={'locked_height_min_m':float(h[mask].min()),'locked_height_max_m':float(h[mask].max()),'locked_max_slide_speed_m_s':float(v[maskv].max()),'locked_max_xy_drift_m':float(np.linalg.norm(pos[mask,:2]-np.array(anchor['ankle_position_w'])[:2],axis=1).max()),'locked_max_tilt_deg':float(np.max(np.array(tilts[side])[mask]))}
    report={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'output':str(output),'anchors':anchors,'anchor_report':str(anchor_report) if anchor_report else None,'edit_interval':edit_interval,'knee_floor':knee_floor,'smooth_interval':smooth_interval,'smooth_strength':smooth_strength,'root_trajectory_smoothing':root_trajectory_smoothing,'min_ground_m':float(min_ground),'min_self_m':min_self,'feet':stats,'max_leg_correction_rad':float(abs(delta.reshape(frames,width)[:,:-3]).max()),'max_root_correction_m':float(abs(delta.reshape(frames,width)[:,-3:]).max()),'iterations':records,'validation':'Native kinematic geometry only; no dynamics validation; original references/manifests untouched.'}
    print(json.dumps(report,indent=2),flush=True)
    if min_ground<-.0001 or min_self<-.0001:raise RuntimeError('Final ground/self collision validation failed')
    if any(v['locked_max_slide_speed_m_s']>.02 or v['locked_height_max_m']>.006 for v in stats.values()):raise RuntimeError('Foot lock/ground validation failed')
    meta['support_repair']=report;payload['metadata_json']=np.array(json.dumps(meta));output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**payload);load_motion(output);output.with_suffix('.repair.json').write_text(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--collision-report',type=Path,required=True);p.add_argument('--support-start',type=float,nargs=2,required=True,metavar=('LEFT_S','RIGHT_S'));p.add_argument('--transition',type=float,default=.6);p.add_argument('--iterations',type=int,default=18);p.add_argument('--knee-floor',type=float,default=0.);p.add_argument('--smooth-interval',nargs=2,type=float);p.add_argument('--smooth-strength',type=float,default=0.);p.add_argument('--anchor-report',type=Path);p.add_argument('--edit-interval',nargs=2,type=float);a=p.parse_args()
    if not np.isfinite([*a.support_start,a.transition]).all() or min(a.support_start)<0 or a.transition<=0 or a.iterations<1:p.error('Invalid support schedule')
    if not np.isfinite([a.knee_floor,a.smooth_strength]).all() or a.knee_floor<0 or a.smooth_strength<0:p.error('Invalid knee/smoothing settings')
    if a.smooth_interval is not None and (not np.isfinite(a.smooth_interval).all() or a.smooth_interval[0]<0 or a.smooth_interval[1]<=a.smooth_interval[0]):p.error('Invalid smoothing interval')
    solve(a.input,a.output,a.collision_report,a.support_start,a.transition,a.iterations,a.knee_floor,a.smooth_interval,a.smooth_strength,a.anchor_report,a.edit_interval)
if __name__=='__main__':main()
