"""Reduce material contact-point slip while preserving reference timing and support height.

Writes a separate dataset; run check_x2_retarget.py on the output before use.
Wrists use the middle get-up phase, feet include subsequent near-floor frames.
Near-floor masks are frozen from the source, so lifting
an extremity cannot masquerade as a reduction in slipping. Rolling is permitted:
only the currently lowest material point is constrained, not the whole rigid body.
"""
import argparse
import json
from pathlib import Path
import sys

import clarabel
import mujoco
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.x2_motion_utils import load_motion
from checks.check_x2_retarget import geometry_lowest, load_input
from retarget.convert_x2_reference import export_mujoco_fk
from repair.repair_x2_joint_limit_margin import smoothstep


def solve(source, destination, iterations=8):
    model, original, times, names, jids, adr, ground, _, xml = load_input(source)
    z = load_motion(source)
    metadata = z.pop('metadata')
    data = mujoco.MjData(model)
    n = len(times)
    dt = float(times[1]-times[0])
    middle = (original[:, 2]-ground >= .2) & (original[:, 2]-ground < .5)
    blend = smoothstep((times-times[middle][0]+.5)/.4)*smoothstep((times[-1]-times)/.4)
    joints = [i for i, name in enumerate(z['joint_names'].tolist()) if 'head' not in name]
    qadr = np.r_[np.arange(3), np.asarray(adr)[joints]]
    dofs = np.r_[np.arange(3), [int(model.jnt_dofadr[jids[i]]) for i in joints]]
    width = len(qadr)
    size = n*width
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] and (model.geom_contype[g] or model.geom_conaffinity[g])]
    bodies = [model.body(s+'_'+b).id for s in ('left','right') for b in ('wrist_roll_link','ankle_roll_link')]
    bg = {b: [g for g in geoms if model.geom_bodyid[g] == b] for b in bodies}
    masks = {}
    source_height = {}
    for b in bodies:
        heights=[]
        for f in range(n):
            data.qpos[:]=original[f];mujoco.mj_forward(model,data)
            heights.append(min(geometry_lowest(model,data,g)[2]-ground for g in bg[b]))
        source_height[b]=np.asarray(heights)
        phase = middle if 'wrist' in model.body(b).name else (original[:,2]-ground>=.2)
        near=(source_height[b]<.01)&(source_height[b]>=-.001)&phase
        edges=np.diff(np.r_[False,near,False].astype(int))
        mask=np.zeros(n,dtype=bool)
        for a,e in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            if (e-a)*dt >= .1: mask[a:e]=True
        masks[b]=mask
    # Squared deviations, plus smooth acceleration/jerk of the correction.
    d2=sp.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n))
    d3=sp.diags([-np.ones(n-3),3*np.ones(n-3),-3*np.ones(n-3),np.ones(n-3)],[0,1,2,3],shape=(n-3,n))
    track=np.r_[100.,100.,250.,np.full(width-3,1.)]
    smoothing=sp.kron(100*(d2.T@d2)+1000*(d3.T@d3),sp.diags(np.r_[200.,200.,400.,np.ones(width-3)])).tocsc()
    base_h=sp.kron(sp.eye(n),sp.diags(track))+smoothing
    smooth_linear=np.asarray(smoothing@original[:,qadr].ravel()).ravel()
    max_delta=np.r_[.08,.08,.035,np.full(width-3,.45)]
    lower=-blend[:,None]*max_delta;upper=blend[:,None]*max_delta
    limits=z['joint_hard_limits'][joints].astype(float)
    margin=np.zeros((n,len(joints)))
    for c,j in enumerate(joints):
        if any(s in str(z['joint_names'][j]) for s in ('wrist_pitch','wrist_roll','ankle_pitch','ankle_roll')):
            # Never reduce a source margin that is already smaller outside the repaired middle stage.
            margin[:,c]=np.minimum(.035*blend,np.minimum(original[:,adr[j]]-limits[c,0],limits[c,1]-original[:,adr[j]]))
    lower[:,3:]=np.maximum(lower[:,3:],limits[None,:,0]+margin-original[:,np.asarray(adr)[joints]])
    upper[:,3:]=np.minimum(upper[:,3:],limits[None,:,1]-margin-original[:,np.asarray(adr)[joints]])
    active=np.flatnonzero((upper-lower).ravel()>1e-9)
    delta=np.zeros((n,width)); current=original.copy()
    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
    history=[]
    def evaluate(q, collect=False):
        rows=[]; cols=[]; vals=[]; rhs=[]; gr=[];gc=[];gv=[];gb=[];metrics={b:[] for b in bodies};minimum=1.
        for f in range(n):
            data.qpos[:]=q[f];mujoco.mj_forward(model,data)
            points={b:min((geometry_lowest(model,data,g).copy() for g in bg[b]),key=lambda p:p[2]) for b in bodies}
            # Native collision surfaces, including the head and forearms.
            for g in geoms:
                p=geometry_lowest(model,data,g).copy();h=p[2]-ground;minimum=min(minimum,h)
                if not collect or blend[f]<=0 or h>.025:continue
                mujoco.mj_jac(model,data,jp,jr,p,int(model.geom_bodyid[g]));j=jp[2,dofs].copy()
                r=len(gb);gr.extend([r]*width);gc.extend((f*width+np.arange(width)).tolist());gv.extend((-j).tolist());gb.append(h-.0002-float(j@delta[f]))
            for b in bodies:
                if not masks[b][f]:continue
                data.qpos[:]=q[f];mujoco.mj_forward(model,data)
                p=points[b];local=data.xmat[b].reshape(3,3).T@(p-data.xpos[b])
                a=max(0,f-1);e=min(n-1,f+1);span=(e-a)*dt
                velocities=[];jacobians=[]
                for k in (a,e):
                    data.qpos[:]=q[k];mujoco.mj_forward(model,data)
                    point=data.xpos[b]+data.xmat[b].reshape(3,3)@local
                    velocities.append(point.copy())
                    if collect:
                        mujoco.mj_jac(model,data,jp,jr,point,b);jacobians.append(jp[:2,dofs].copy())
                v=(velocities[1]-velocities[0])/span
                metrics[b].append(float(np.linalg.norm(v[:2])))
                if collect:
                    for axis in range(2):
                        r=len(rhs);target=-v[axis]
                        for k,j,sign in zip((a,e),jacobians,(-1.,1.)):
                            coeff=sign*j[axis]/span
                            rows.extend([r]*width);cols.extend((k*width+np.arange(width)).tolist());vals.extend(coeff.tolist());target+=float(coeff@delta[k])
                        rhs.append(target)
                    # Preserve the original contact height rather than raising a hand to evade the mask.
                    data.qpos[:]=q[f];mujoco.mj_forward(model,data)
                    mujoco.mj_jac(model,data,jp,jr,p,b);j=jp[2,dofs].copy();r=len(gb)
                    gr.extend([r]*width);gc.extend((f*width+np.arange(width)).tolist());gv.extend(j.tolist());gb.append(max(.004,source_height[b][f]+.003)-(p[2]-ground)+float(j@delta[f]))
        summary={model.body(b).name:{'count':len(v),'mean_m_s':float(np.mean(v)),'p90_m_s':float(np.quantile(v,.9)),'max_m_s':float(max(v))} for b,v in metrics.items()}
        summary['minimum_clearance_m']=float(minimum)
        if not collect:return summary
        m=sp.coo_matrix((vals,(rows,cols)),shape=(len(rhs),size)).tocsc()
        g=sp.coo_matrix((gv,(gr,gc)),shape=(len(gb),size)).tocsc()
        return summary,m,np.asarray(rhs),g,np.asarray(gb)
    before=evaluate(current)
    for it in range(iterations):
        summary,m,target,g,gb=evaluate(current,True)
        weight=25.
        h=(base_h+weight*(m.T@m)).tocsc();linear=smooth_linear-weight*np.asarray(m.T@target).ravel()
        trust=np.r_[.008,.008,.006,np.full(width-3,.07)]
        lo=np.maximum(lower,delta-trust).ravel()[active];hi=np.minimum(upper,delta+trust).ravel()[active]
        a=sp.vstack([sp.eye(len(active)),-sp.eye(len(active)),g[:,active]]).tocsc();bb=np.r_[hi,-lo,gb]
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100
        result=clarabel.DefaultSolver(sp.triu(h[active][:,active]).tocsc(),linear[active],a,bb,[clarabel.NonnegativeConeT(len(bb))],settings).solve()
        if str(result.status) not in ('Solved','AlmostSolved'):raise RuntimeError(f'Contact QP: {result.status}')
        flat=np.zeros(size);flat[active]=result.x;delta=flat.reshape(n,width)
        current=original.copy();current[:,qadr]+=delta
        result_metrics=evaluate(current);history.append(result_metrics)
        print(source.name,it+1,json.dumps(result_metrics),flush=True)
    after=evaluate(current)
    z['root_pos_w']=current[:,:3].astype(np.float32)
    z['root_lin_vel_w']=np.gradient(z['root_pos_w'],dt,axis=0).astype(np.float32)
    # Recompute local root velocity using source root orientation (unchanged).
    from scipy.spatial.transform import Rotation
    z['root_lin_vel_b']=Rotation.from_quat(z['root_quat_xyzw']).inv().apply(z['root_lin_vel_w']).astype(np.float32)
    z['joint_pos']=current[:,adr].astype(np.float32)
    from repair.repair_x2_joint_limit_margin import compensate_ground
    clearance_repair=compensate_ground(z,metadata,blend)
    z['joint_vel']=np.gradient(z['joint_pos'],dt,axis=0).astype(np.float32)
    ids=z['policy_joint_indices'];z['policy_joint_pos']=z['joint_pos'][:,ids];z['policy_joint_vel']=z['joint_vel'][:,ids]
    z.update(export_mujoco_fk(str(xml),z['joint_names'].tolist(),z['root_pos_w'],z['root_quat_xyzw'],z['joint_pos']))
    for key in ('body_lin_vel_w','body_ang_vel_w'):z.pop(key,None)
    current[:,:3]=z['root_pos_w'];current[:,adr]=z['joint_pos']
    after=evaluate(current)
    if after['minimum_clearance_m'] < -.001:
        raise RuntimeError('Final reference exceeds 1 mm ground penetration')
    if not np.array_equal(z['joint_pos'][[0,-1]],original[[0,-1]][:,adr].astype(np.float32)):
        raise RuntimeError('Endpoint joint pose changed')
    if not np.array_equal(z['root_pos_w'][[0,-1]],original[[0,-1],:3].astype(np.float32)):
        raise RuntimeError('Endpoint root position changed')
    report={'clearance_repair':clearance_repair,'source':str(source.resolve()),'before':before,'after':after,'history':history,'max_root_change_m':np.max(abs(delta[:,:3]),axis=0).tolist(),'max_joint_change_rad':float(abs(delta[:,3:]).max())}
    metadata['contact_slip_repair']=report;z['metadata_json']=np.array(json.dumps(metadata))
    destination.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(destination,**z);load_motion(destination)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--iterations',type=int,default=8);args=p.parse_args()
    if args.iterations < 1:
        p.error('--iterations must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=False);reports=[];entries=[]
    for entry in json.loads(args.manifest.read_text())['motions']:
        source=(args.manifest.parent/entry['file']).resolve();dest=args.output_dir/'candidates'/source.name
        backup=args.output_dir/'before'/source.name;backup.parent.mkdir(exist_ok=True);backup.write_bytes(source.read_bytes())
        reports.append(solve(source,dest,args.iterations));entries.append(dict(entry,file=source.name))
    for folder in ('before','candidates'):(args.output_dir/folder/'motions.json').write_text(json.dumps({'motions':entries},indent=2))
    (args.output_dir/'repair.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__':main()
