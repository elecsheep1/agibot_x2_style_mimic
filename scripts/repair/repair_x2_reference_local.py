"""Local UMR hand-support stabilization and SO(3) pelvis smoothing.

Preserves source files and recomputes all affected velocities and FK arrays.
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
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from repair.repair_x2_reference_feet import geometry, lowest_point
from retarget.convert_x2_reference import export_mujoco_fk, angular_velocity_world
from common.x2_motion_utils import load_motion


def smoothstep(x):
    x=np.clip(x,0,1);return x*x*x*(x*(x*6-15)+10)


def export(source,output,z,meta,report):
    dt=meta['dt_s'];rotation=Rotation.from_quat(z['root_quat_xyzw'])
    z['joint_vel']=np.gradient(z['joint_pos'],dt,axis=0).astype(np.float32)
    z['root_lin_vel_w']=np.gradient(z['root_pos_w'],dt,axis=0).astype(np.float32)
    z['root_ang_vel_w']=angular_velocity_world(rotation,dt).astype(np.float32)
    z['root_lin_vel_b']=rotation.inv().apply(z['root_lin_vel_w']).astype(np.float32)
    z['root_ang_vel_b']=rotation.inv().apply(z['root_ang_vel_w']).astype(np.float32)
    z['projected_gravity_b']=rotation.inv().apply(np.tile([0,0,-1.],(len(z['time_s']),1))).astype(np.float32)
    ids=z['policy_joint_indices'];z['policy_joint_pos']=z['joint_pos'][:,ids];z['policy_joint_vel']=z['joint_vel'][:,ids]
    z.update(export_mujoco_fk(meta['mujoco_xml'],z['joint_names'].tolist(),z['root_pos_w'],z['root_quat_xyzw'],z['joint_pos']))
    report.update(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output=str(output))
    meta['local_repair']=report;z['metadata_json']=np.array(json.dumps(meta));output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**z);load_motion(output);output.with_suffix('.repair.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def root_smooth(source,output,interval,sigma):
    z=load_motion(source);meta=z.pop('metadata');t=z['time_s'];dt=meta['dt_s'];R=Rotation.from_quat(z['root_quat_xyzw']);base=R[int(round(interval[0]/dt))]
    tangent=(R*base.inv()).as_rotvec();filtered=gaussian_filter1d(tangent,sigma/dt,axis=0,mode='nearest')
    blend=smoothstep((t-interval[0])/.2)*smoothstep((interval[1]-t)/.2)
    rot=Rotation.from_rotvec(tangent+(filtered-tangent)*blend[:,None])*base
    quats=rot.as_quat();before=z['root_quat_xyzw'].copy();mask=blend>0;quats[~mask]=before[~mask]
    for f in range(1,len(quats)):
        if quats[f]@quats[f-1]<0:quats[f]*=-1
    z['root_quat_xyzw']=quats.astype(np.float32)
    delta=(rot*R.inv()).magnitude();report={'mode':'local SO3 pelvis smoothing','interval_s':interval,'sigma_s':sigma,'max_rotation_change_deg':float(np.degrees(delta).max())}
    export(source,output,z,meta,report)


def joint_smooth(source,output,interval,sigma,pattern):
    z=load_motion(source);meta=z.pop('metadata');t=z['time_s'];dt=meta['dt_s']
    blend=smoothstep((t-interval[0])/.25)*smoothstep((interval[1]-t)/.25)
    ids=[i for i,n in enumerate(z['joint_names']) if pattern in n]
    if not ids:raise ValueError('No joints match smoothing pattern')
    q=z['joint_pos'][:,ids].copy();target=gaussian_filter1d(q,sigma/dt,axis=0,mode='nearest')
    z['joint_pos'][:,ids]=(q+(target-q)*blend[:,None]).astype(np.float32)
    report={'mode':'local joint trajectory smoothing','interval_s':interval,'sigma_s':sigma,'joint_pattern':pattern,'max_change_rad':float(abs(z['joint_pos'][:,ids]-q).max())}
    export(source,output,z,meta,report)


def hand_stabilize(source,output,interval,collision_report,target_mode="lock",sigma=.10,max_change=.06):
    z=load_motion(source);meta=z.pop('metadata');dt=meta['dt_s'];t=z['time_s'];model=mujoco.MjModel.from_xml_path(meta['mujoco_xml']);data=mujoco.MjData(model)
    names=z['joint_names'].tolist();addresses=[int(model.joint(n).qposadr[0]) for n in names];arm_names=[n for n in names if any(k in n for k in ('shoulder_','elbow_','wrist_'))]
    arm_idx=[names.index(n) for n in arm_names];qcols=[int(model.joint(n).qposadr[0]) for n in arm_names];dofs=[int(model.joint(n).dofadr[0]) for n in arm_names];width=len(qcols)
    q=np.zeros((len(t),model.nq));q[:,:3]=z['root_pos_w'];q[:,3:7]=z['root_quat_xyzw'][:,[3,0,1,2]];q[:,addresses]=z['joint_pos'];original=q.copy()
    blend=smoothstep((t-interval[0])/.3)*smoothstep((interval[1]-t)/.3);frames=np.flatnonzero(blend>0)
    bodyids=[int(model.body(s+'_wrist_roll_link').id) for s in ('left','right')]
    originalpos=z['body_pos_w'][:,[z['body_names'].tolist().index(s+'_wrist_roll_link') for s in ('left','right')]]
    originalquat=z['body_quat_xyzw'][:,[z['body_names'].tolist().index(s+'_wrist_roll_link') for s in ('left','right')]]
    center=(t>=interval[0]+.3)&(t<=interval[1]-.3)
    anchorpos=originalpos[center].mean(0);anchorrot=[Rotation.from_quat(originalquat[center,s]).mean() for s in range(2)]
    if target_mode=='smooth':
        desired_positions=gaussian_filter1d(originalpos,sigma/dt,axis=0,mode='nearest')
    else:
        desired_positions=np.broadcast_to(anchorpos,originalpos.shape)
    targets=originalpos*(1-blend[:,None,None])+desired_positions*blend[:,None,None]
    targetrot=[]
    for s in range(2):
        R=Rotation.from_quat(originalquat[:,s])
        if target_mode=='smooth':
            base=R[int(round(np.mean(interval)/dt))]
            tangent=(R*base.inv()).as_rotvec()
            desired=Rotation.from_rotvec(gaussian_filter1d(tangent,sigma/dt,axis=0,mode='nearest'))*base
            err=(desired*R.inv()).as_rotvec()
        else:
            err=(anchorrot[s]*R.inv()).as_rotvec()
        targetrot.append(Rotation.from_rotvec(err*blend[:,None])*R)
    geoms,vertices=geometry(model);audit=json.loads(Path(collision_report).read_text());blocked={tuple(sorted(p)) for key in ('excluded_adjacent_pairs','excluded_authored_filters') for p in audit[key]}
    def bname(g):
        n=model.body(int(model.geom_bodyid[g])).name;return 'base_link' if n=='pelvis' else n
    pairs=[(a,b) for a,b in itertools.combinations(geoms,2) if model.geom_bodyid[a]!=model.geom_bodyid[b] and tuple(sorted((bname(a),bname(b)))) not in blocked and any(k in bname(a)+bname(b) for k in ('shoulder','elbow','wrist'))]
    lo=np.array([model.joint(n).range[0] for n in arm_names]);hi=np.array([model.joint(n).range[1] for n in arm_names]);history=[]
    for f in frames:
        x=np.zeros(width)
        for iteration in range(6):
            data.qpos[:]=q[f];mujoco.mj_forward(model,data);H=np.eye(width);g=np.zeros(width);rows=[];rhs=[]
            for s,body in enumerate(bodyids):
                jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jacBody(model,data,jp,jr,body);J=np.vstack([jp[:,dofs],jr[:,dofs]])
                err=np.r_[targets[f,s]-data.xpos[body],(targetrot[s][f]*Rotation.from_matrix(data.xmat[body].reshape(3,3)).inv()).as_rotvec()]
                w=np.r_[np.full(3,1e8),np.full(3,1e6)];H+=J.T@(w[:,None]*J);g-=J.T@(w*(err+J@x))
            def add(j,distance,margin):
                j=j[dofs]
                if np.linalg.norm(j)>1e-12:rows.append(-j);rhs.append(distance-margin-j@x)
            for geom in geoms:
                if not any(k in bname(geom) for k in ('shoulder','elbow','wrist')):continue
                p=lowest_point(model,data,geom,vertices);h=p[2]-meta['ground_z']
                if h<.02:
                    j=np.zeros((3,model.nv));mujoco.mj_jac(model,data,j,None,p,int(model.geom_bodyid[geom]));add(j[2],h,.0018)
            for a,b in pairs:
                if np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b])>model.geom_rbound[a]+model.geom_rbound[b]+.01:continue
                pts=np.zeros(6);d=float(mujoco.mj_geomDistance(model,data,a,b,.01,pts));v=pts[:3]-pts[3:];norm=np.linalg.norm(v)
                if d<.008 and norm>1e-12:
                    normal=v/norm*(1 if d>=0 else -1);ja=np.zeros((3,model.nv));jb=ja.copy();mujoco.mj_jac(model,data,ja,None,pts[:3],int(model.geom_bodyid[a]));mujoco.mj_jac(model,data,jb,None,pts[3:],int(model.geom_bodyid[b]));add(normal@(ja-jb),d,0.)
            lower=np.maximum(lo-original[f,qcols],-max_change*blend[f]);upper=np.minimum(hi-original[f,qcols],max_change*blend[f]);A=sp.csc_matrix(np.vstack([np.array(rows).reshape(-1,width),np.eye(width),-np.eye(width)]));bvec=np.r_[rhs,upper,-lower]
            settings=clarabel.DefaultSettings();settings.verbose=False;solution=clarabel.DefaultSolver(sp.triu(sp.csc_matrix(H),format='csc'),g,A,bvec,[clarabel.NonnegativeConeT(len(bvec))],settings).solve()
            if str(solution.status) not in ('Solved','AlmostSolved'):raise RuntimeError(f'Hand QP frame {f}: {solution.status}')
            new=np.array(solution.x);change=abs(new-x).max();x=new;q[f,qcols]=original[f,qcols]+x
            if change<1e-8:break
        history.append(float(abs(x).max()))
    z['joint_pos'][:,arm_idx]=q[:,qcols].astype(np.float32)
    report={'mode':'local hand world-pose '+target_mode,'sigma_s':sigma,'max_allowed_joint_change_rad':max_change,'interval_s':interval,'fade_s':.3,'max_joint_change_rad':max(history),'anchor_positions_w':anchorpos.tolist(),'plateau_s':[interval[0]+.3,interval[1]-.3]}
    export(source,output,z,meta,report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['hand','root','joints'],required=True);p.add_argument('--interval',nargs=2,type=float,required=True);p.add_argument('--sigma',type=float,default=.14);p.add_argument('--collision-report',type=Path);p.add_argument('--hand-target',choices=['lock','smooth'],default='lock');p.add_argument('--max-change',type=float,default=.06);p.add_argument('--joint-pattern',default='waist_');a=p.parse_args();source=a.input.resolve();output=a.output.resolve()
    if output.exists():raise FileExistsError(output)
    if not np.isfinite([*a.interval,a.sigma]).all() or a.interval[0]<0 or a.interval[1]<=a.interval[0] or a.sigma<=0 or not np.isfinite(a.max_change) or a.max_change<=0:p.error('Invalid local interval/sigma')
    if a.mode=='hand' and (a.collision_report is None or a.interval[1]-a.interval[0]<=.6):p.error('Hand mode needs a collision report and an interval longer than 0.6 s')
    if a.mode=='root':root_smooth(source,output,a.interval,a.sigma)
    elif a.mode=='joints':joint_smooth(source,output,a.interval,a.sigma,a.joint_pattern)
    else:hand_stabilize(source,output,a.interval,a.collision_report,a.hand_target,a.sigma,a.max_change)
if __name__=='__main__':main()
