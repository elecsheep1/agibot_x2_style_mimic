"""Create backed-up X2 reference candidates with smooth wrist/ankle limit margins."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.x2_motion_utils import load_motion
from retarget.convert_x2_reference import export_mujoco_fk


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10.0 - 15.0*x + 6.0*x*x)


def interior(q, low, high, margin):
    """Monotone mapping, identical beyond a 3*margin boundary band (C2 join)."""
    if high-low <= 6*margin:
        raise ValueError('Joint range too narrow for requested margin')
    result = q.copy()
    a = np.clip((q-low)/(3*margin), 0, 1)
    b = np.clip((high-q)/(3*margin), 0, 1)
    result += margin*(1-a)**3
    result -= margin*(1-b)**3
    return result



def compensate_ground(z, meta, blend):
    """Small smooth arm/leg compensation; root and edited wrist/ankle stay fixed."""
    import mujoco
    import clarabel
    import scipy.sparse as sp
    from checks.check_x2_retarget import geometry_lowest
    model=mujoco.MjModel.from_xml_path(meta['mujoco_xml']);data=mujoco.MjData(model)
    names=z['joint_names'].tolist();free=[i for i,n in enumerate(names) if any(k in n for k in ('hip_','knee_','shoulder_','elbow_','wrist_yaw'))]
    dofs=[int(model.joint(names[i]).dofadr[0]) for i in free];adr=[int(model.joint(n).qposadr[0]) for n in names]
    geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g] and (model.geom_contype[g] or model.geom_conaffinity[g])]
    n=len(blend);width=len(free);size=n*width;base=z['joint_pos'].astype(float).copy();current=np.zeros((n,width))
    D2=sp.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n));D3=sp.diags([-np.ones(n-3),3*np.ones(n-3),-3*np.ones(n-3),np.ones(n-3)],[0,1,2,3],shape=(n-3,n))
    P=sp.triu(sp.kron(sp.eye(n)+200*(D2.T@D2)+2000*(D3.T@D3),sp.eye(width))).tocsc()
    low=np.maximum(z['joint_hard_limits'][free,0]-base[:,free],-.04*blend[:,None]).ravel();high=np.minimum(z['joint_hard_limits'][free,1]-base[:,free],.04*blend[:,None]).ravel()
    jac=np.zeros((3,model.nv));jacr=np.zeros_like(jac);minimum=0.
    for iteration in range(5):
        rr=[];cc=[];vv=[];bounds=[];minimum=1.
        for f in range(n):
            data.qpos[:3]=z['root_pos_w'][f];data.qpos[3:7]=z['root_quat_xyzw'][f,[3,0,1,2]];q=base[f].copy();q[free]+=current[f];data.qpos[adr]=q;mujoco.mj_forward(model,data)
            for g in geoms:
                point=geometry_lowest(model,data,g).copy();height=point[2]-meta.get('ground_z',0);minimum=min(minimum,height)
                if blend[f]<=0 or height>.015:continue
                mujoco.mj_jac(model,data,jac,jacr,point,int(model.geom_bodyid[g]));j=jac[2,dofs]
                if np.linalg.norm(j)<1e-9:
                    if height<-.001:raise RuntimeError('Ground collision cannot be corrected with selected joints')
                    continue
                row=len(bounds);rr.extend([row]*width);cc.extend((f*width+np.arange(width)).tolist());vv.extend((-j).tolist());bounds.append(height-.0012-float(j@current[f]))
        print("ground_iteration", iteration, "minimum", minimum, flush=True)
        if minimum>=-.0008:break
        ground=sp.coo_matrix((vv,(rr,cc)),shape=(len(bounds),size)).tocsc();A=sp.vstack([sp.eye(size),-sp.eye(size),ground]).tocsc();b=np.r_[high,-low,bounds]
        settings=clarabel.DefaultSettings();settings.verbose=False
        solution=clarabel.DefaultSolver(P,np.zeros(size),A,b,[clarabel.NonnegativeConeT(len(b))],settings).solve()
        if str(solution.status) not in ('Solved','AlmostSolved'):raise RuntimeError(str(solution.status))
        current=np.asarray(solution.x).reshape(n,width)
    else:raise RuntimeError('Ground clearance compensation did not converge')
    current[blend<=0]=0.0
    z['joint_pos'][:,free]=(base[:,free]+current).astype(np.float32)
    return {'minimum_surface_clearance_m':minimum,'max_compensation_rad':float(abs(current).max()),'compensated_joints':[names[i] for i in free],'iterations':iteration+1}

def repair(source, destination, margin):
    z = load_motion(source)
    meta = z.pop('metadata')
    original = z['joint_pos'].copy()
    t = z['time_s']
    middle = (z['root_pos_w'][:, 2]-meta.get('ground_z', 0) >= .2) & (z['root_pos_w'][:, 2]-meta.get('ground_z', 0) < .5)
    if not middle.any():
        raise ValueError('No middle-stage frames')
    start, end = float(t[middle][0]-.5), float(t[middle][-1]+.5)
    blend = smoothstep((t-start)/.4)*smoothstep((end-t)/.4)
    if not np.all(blend[middle] == 1):
        raise ValueError('Middle stage not fully covered')
    rows = []
    for j, name in enumerate(z['joint_names'].tolist()):
        if not any(k in name for k in ('wrist_pitch', 'wrist_roll', 'ankle_pitch', 'ankle_roll')):
            continue
        low, high = z['joint_hard_limits'][j]
        before = original[:, j]
        if np.minimum(before[middle]-low, high-before[middle]).min() >= .01:
            continue
        desired = interior(before.astype(float), float(low), float(high), margin)
        z['joint_pos'][:, j] = (before+blend*(desired-before)).astype(np.float32)
        after = z['joint_pos'][:, j]
        assert np.minimum(after[middle]-low, high-after[middle]).min() >= margin-1e-6
        dt = float(meta['dt_s'])
        rows.append(dict(joint=name, max_change_rad=float(abs(after-before).max()),
                         before_near_limit_fraction=float((np.minimum(before[middle]-low, high-before[middle])<.01).mean()),
                         after_near_limit_fraction=float((np.minimum(after[middle]-low, high-after[middle])<.01).mean()),
                         minimum_middle_margin_rad=float(np.minimum(after[middle]-low, high-after[middle]).min()),
                         before_peak_velocity=float(abs(np.gradient(before,dt)).max()),
                         after_peak_velocity=float(abs(np.gradient(after,dt)).max())))
    compensation = compensate_ground(z, meta, blend)
    assert np.array_equal(z['joint_pos'][[0,-1]], original[[0,-1]])
    z['joint_vel'] = np.gradient(z['joint_pos'], float(meta['dt_s']), axis=0).astype(np.float32)
    ids = z['policy_joint_indices']
    z['policy_joint_pos'] = z['joint_pos'][:,ids]
    z['policy_joint_vel'] = z['joint_vel'][:,ids]
    z.update(export_mujoco_fk(meta['mujoco_xml'], z['joint_names'].tolist(), z['root_pos_w'], z['root_quat_xyzw'], z['joint_pos']))
    # FK cache velocities would be stale after editing joint trajectories.
    for key in ('body_lin_vel_w','body_ang_vel_w'):
        z.pop(key, None)
    report = dict(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  interval_s=[start,end], margin_rad=margin, joints=rows,
                  root_trajectory_changed=False, first_and_last_joint_pose_changed=False, ground_compensation=compensation)
    meta['joint_limit_margin_repair'] = report
    z['metadata_json'] = np.array(json.dumps(meta))
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    np.savez_compressed(destination, **z)
    load_motion(destination)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--margin',type=float,default=.035)
    args=parser.parse_args()
    if not np.isfinite(args.margin) or args.margin<=0:
        parser.error('margin must be positive and finite')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    backup=args.output_dir/'before';candidate=args.output_dir/'candidates';backup.mkdir();candidate.mkdir()
    manifest=json.loads(args.manifest.read_text());before=[];after=[];reports=[]
    for entry in manifest['motions']:
        source=(args.manifest.parent/entry['file']).resolve();b=backup/source.name;b.write_bytes(source.read_bytes())
        c=candidate/source.name;reports.append(repair(source,c,args.margin))
        before.append(dict(entry,file=str(b.resolve())));after.append(dict(entry,file=str(c.resolve())))
    (backup/'motions.json').write_text(json.dumps({'motions':before},indent=2))
    (candidate/'motions.json').write_text(json.dumps({'motions':after},indent=2))
    (args.output_dir/'repair.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))


if __name__=='__main__':
    main()
