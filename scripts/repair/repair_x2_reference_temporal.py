"""Jointly optimize a whole arm/waist trajectory with temporal and geometry constraints.

Unlike framewise wrist IK, regularizes the entire selected chain through time.
Always creates a new reference. Root trajectory is fixed to the input, while an
optional separate reference supplies desired world-body trajectories.
"""

# Support direct execution: python scripts/<category>/<tool>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import itertools
import json
from pathlib import Path

import clarabel
import mujoco
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from repair.repair_x2_reference_feet import geometry,lowest_point
from repair.repair_x2_reference_local import export,smoothstep
from common.x2_motion_utils import load_motion


def repair(source,output,reference,collision_report,mode,interval,sigma,max_change=.3,side='both'):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    z=load_motion(source);meta=z.pop('metadata');ref=load_motion(reference or source);dt=meta['dt_s'];time=z['time_s'];frames=len(time)
    if not np.array_equal(time,ref['time_s']):raise ValueError('Reference time axis differs')
    model=mujoco.MjModel.from_xml_path(meta['mujoco_xml']);data=mujoco.MjData(model);names=z['joint_names'].tolist();addresses=[int(model.joint(n).qposadr[0]) for n in names]
    patterns={'arms':('shoulder_','elbow_','wrist_'),'waist':('waist_',),'legs':('hip_','knee_','ankle_')}
    selected=[n for n in names if any(k in n for k in patterns[mode])]
    if side!='both':selected=[n for n in selected if n.startswith(side+'_')]
    if not selected:raise ValueError('No selected joints')
    selected_ids=[names.index(n) for n in selected];qcols=[int(model.joint(n).qposadr[0]) for n in selected];dofs=[int(model.joint(n).dofadr[0]) for n in selected];width=len(selected);size=width*frames
    q=np.zeros((frames,model.nq));q[:,:3]=z['root_pos_w'];q[:,3:7]=z['root_quat_xyzw'][:,[3,0,1,2]];q[:,addresses]=z['joint_pos'];original=q.copy()
    blend=smoothstep((time-interval[0])/.4)*smoothstep((interval[1]-time)/.4)
    if interval[0]==0:blend=smoothstep((interval[1]-time)/.4)
    body_names={'arms':['left_wrist_roll_link','right_wrist_roll_link','left_elbow_link','right_elbow_link'], 'waist':['torso_link'], 'legs':['left_ankle_roll_link','right_ankle_roll_link','left_knee_link','right_knee_link']}[mode]
    tasks=[]
    for name in body_names:
        body=int(model.body(name).id);idx=ref['body_names'].tolist().index(name);positions=ref['body_pos_w'][:,idx]
        rotations=Rotation.from_quat(ref['body_quat_xyzw'][:,idx]);base=rotations[int(round(np.mean(interval)/dt))]
        smoothrot=Rotation.from_rotvec(gaussian_filter1d((rotations*base.inv()).as_rotvec(),sigma/dt,axis=0,mode='nearest'))*base
        smoothpos=gaussian_filter1d(positions,sigma/dt,axis=0,mode='nearest')
        sourceidx=z['body_names'].tolist().index(name);sourceR=Rotation.from_quat(z['body_quat_xyzw'][:,sourceidx]);error=(smoothrot*sourceR.inv()).as_rotvec();targetR=Rotation.from_rotvec(error*blend[:,None])*sourceR
        targetpos=z['body_pos_w'][:,sourceidx]*(1-blend[:,None])+smoothpos*blend[:,None]
        endpoint='wrist' in name or 'ankle' in name
        weight=np.r_[np.full(3,1e4 if endpoint else 500.),np.full(3,100. if endpoint else 10.)] if mode!='waist' else np.r_[np.zeros(3),np.full(3,3000.)]
        tasks.append((body,targetpos,targetR,weight))
    D2=sp.diags([np.ones(frames-2),-2*np.ones(frames-2),np.ones(frames-2)],[0,1,2],shape=(frames-2,frames)).tocsc()
    D3=sp.diags([-np.ones(frames-3),3*np.ones(frames-3),-3*np.ones(frames-3),np.ones(frames-3)],[0,1,2,3],shape=(frames-3,frames)).tocsc()
    S=sp.kron(400.*D2.T@D2+4000.*D3.T@D3,sp.eye(width)).tocsc()
    Hbase=10.*sp.eye(size)+S;signal=original[:,qcols].reshape(-1)
    desired=gaussian_filter1d(ref['joint_pos'][:,selected_ids],sigma/dt,axis=0,mode='nearest')
    desired=original[:,qcols]*(1-blend[:,None])+desired*blend[:,None]
    gbase=10.*(signal-desired.reshape(-1))+np.asarray(S@signal)
    geoms,vertices=geometry(model);filters=json.loads(Path(collision_report).read_text());blocked={tuple(sorted(p)) for key in ('excluded_adjacent_pairs','excluded_authored_filters') for p in filters[key]}
    def name(g):
        n=model.body(int(model.geom_bodyid[g])).name;return 'base_link' if n=='pelvis' else n
    pairs=[(a,b) for a,b in itertools.combinations(geoms,2) if model.geom_bodyid[a]!=model.geom_bodyid[b] and tuple(sorted((name(a),name(b)))) not in blocked]
    lo=np.array([model.joint(n).range[0] for n in selected]);hi=np.array([model.joint(n).range[1] for n in selected]);delta=np.zeros(size);history=[];free=np.flatnonzero(np.repeat(blend>1e-9,width))
    # Preserve a valid source's smaller clearance near fixed edit boundaries.
    # Requiring a new 1.8 mm clearance with an almost-zero edit allowance can
    # make an otherwise valid, nonpenetrating source artificially infeasible.
    floor_margin={}
    for f in np.flatnonzero(blend>1e-9):
        data.qpos[:]=original[f];mujoco.mj_forward(model,data)
        for geom in geoms:
            height=lowest_point(model,data,geom,vertices)[2]-meta['ground_z']
            floor_margin[f,geom]=min(.0018,max(0.,height-1e-7))
    for iteration in range(12):
        rr=[];cc=[];vv=[];rhs=[];tr=[];tc=[];tv=[];target=[]
        def add_constraint(f,j,distance,margin):
            j=j[dofs];nz=np.flatnonzero(abs(j)>1e-12)
            if not len(nz):return
            row=len(rhs);rr.extend([row]*len(nz));cc.extend((f*width+nz).tolist());vv.extend((-j[nz]).tolist());rhs.append(distance-margin-j@delta[f*width:(f+1)*width])
        for f in np.flatnonzero(blend>1e-9):
            data.qpos[:]=q[f];mujoco.mj_forward(model,data)
            for body,pos,rot,weight in tasks:
                jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jacBody(model,data,jp,jr,body);J=np.vstack([jp[:,dofs],jr[:,dofs]])
                err=np.r_[pos[f]-data.xpos[body],(rot[f]*Rotation.from_matrix(data.xmat[body].reshape(3,3)).inv()).as_rotvec()];scale=np.sqrt(weight);goal=(err+J@delta[f*width:(f+1)*width])*scale;J=J*scale[:,None]
                for axis in range(6):
                    if scale[axis]==0:continue
                    row=len(target);nz=np.flatnonzero(abs(J[axis])>1e-12);tr.extend([row]*len(nz));tc.extend((f*width+nz).tolist());tv.extend(J[axis,nz].tolist());target.append(goal[axis])
            for geom in geoms:
                point=lowest_point(model,data,geom,vertices);height=point[2]-meta['ground_z']
                if height<.025:
                    j=np.zeros((3,model.nv));mujoco.mj_jac(model,data,j,None,point,int(model.geom_bodyid[geom]));add_constraint(f,j[2],height,floor_margin[f,geom])
            for a,b in pairs:
                if np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b])>model.geom_rbound[a]+model.geom_rbound[b]+.012:continue
                points=np.zeros(6);distance=float(mujoco.mj_geomDistance(model,data,a,b,.012,points));diff=points[:3]-points[3:];norm=np.linalg.norm(diff)
                if distance<.01 and norm>1e-12:
                    normal=diff/norm*(1 if distance>=0 else -1);ja=np.zeros((3,model.nv));jb=ja.copy();mujoco.mj_jac(model,data,ja,None,points[:3],int(model.geom_bodyid[a]));mujoco.mj_jac(model,data,jb,None,points[3:],int(model.geom_bodyid[b]));add_constraint(f,normal@(ja-jb),distance,0.)
        T=sp.coo_matrix((tv,(tr,tc)),shape=(len(target),size)).tocsc();H=Hbase+T.T@T;g=gbase-np.asarray(T.T@np.array(target))
        lower=np.maximum(lo[None,:]-original[:,qcols],-max_change*blend[:,None]).reshape(-1);upper=np.minimum(hi[None,:]-original[:,qcols],max_change*blend[:,None]).reshape(-1)
        lower=np.maximum(lower,delta-.10);upper=np.minimum(upper,delta+.10)
        A=sp.coo_matrix((vv,(rr,cc)),shape=(len(rhs),size)).tocsc();A=sp.vstack([A,sp.eye(size),-sp.eye(size)],format='csc');bvec=np.r_[rhs,upper,-lower]
        A=A[:,free].tocsc();keep=np.asarray(abs(A).sum(axis=1)).reshape(-1)>1e-14
        if np.any(bvec[~keep]<-1e-6):raise RuntimeError('Fixed-frame constraint violated')
        A=A[keep];bvec=bvec[keep];settings=clarabel.DefaultSettings();settings.verbose=False
        solution=clarabel.DefaultSolver(sp.triu(H[free][:,free]/1e3,format='csc'),g[free]/1e3,A,bvec,[clarabel.NonnegativeConeT(len(bvec))],settings).solve()
        if str(solution.status) not in ('Solved','AlmostSolved'):raise RuntimeError(f'Temporal QP failed: {solution.status}')
        new=np.zeros(size);new[free]=solution.x;change=float(abs(new-delta).max());delta=new;q=original.copy();q[:,qcols]+=delta.reshape(frames,width);history.append({'iteration':iteration,'max_update':change});print(history[-1],flush=True)
        if iteration>2 and change<1e-5:break
    z['joint_pos'][:,selected_ids]=q[:,qcols].astype(np.float32)
    report={'mode':'joint temporal '+mode+' chain optimization','interval_s':interval,'sigma_s':sigma,'joint_names':selected,'max_joint_change_rad':float(abs(delta).max()),'targets':'smoothed world chain poses' if mode!='waist' else 'smoothed world torso orientation','reference':str(reference or source),'iterations':history,'limitation':'kinematic/native geometry only; requires visual review and independent audit'}
    export(source,output,z,meta,report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--collision-report',type=Path,required=True);p.add_argument('--mode',choices=['arms','waist','legs'],required=True);p.add_argument('--interval',type=float,nargs=2,required=True);p.add_argument('--sigma',type=float,default=.14);p.add_argument('--max-change',type=float,default=.3);a=p.parse_args()
    if not np.isfinite([*a.interval,a.sigma,a.max_change]).all() or a.interval[0]<0 or a.interval[1]<=a.interval[0] or a.sigma<=0 or a.max_change<=0:p.error('Invalid temporal repair settings')
    repair(a.input,a.output,a.reference,a.collision_report,a.mode,a.interval,a.sigma,a.max_change)
