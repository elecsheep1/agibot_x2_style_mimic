"""Offline bounded ankle smoothing with native collision/ground constraints.

Creates a new reference; never overwrites a training motion. Root orientation,
root XY and all non-ankle joints are preserved. Geometry checks are kinematic.
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
from scipy.spatial.transform import Rotation

from retarget.convert_x2_reference import export_mujoco_fk
from common.x2_motion_utils import load_motion


def geometry(model):
    geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] and
             (model.geom_contype[i] or model.geom_conaffinity[i])]
    vertices = {}
    for i in geoms:
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[i]
            start = model.mesh_vertadr[mesh]
            vertices[i] = model.mesh_vert[start:start + model.mesh_vertnum[mesh]].copy()
    return geoms, vertices


def lowest_point(model, data, geom, vertices):
    pos = data.geom_xpos[geom]
    R = data.geom_xmat[geom].reshape(3, 3)
    size = model.geom_size[geom]
    kind = model.geom_type[geom]
    if geom in vertices:
        world = vertices[geom] @ R.T + pos
        return world[np.argmin(world[:, 2])]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return pos - np.array([0., 0., size[0]])
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        row = R[2]; local = np.zeros(3); norm = np.linalg.norm(row[:2])
        if norm > 1e-12:
            local[:2] = -size[0] * row[:2] / norm
        local[2] = -size[1] * np.sign(row[2])
        return R @ local + pos
    raise ValueError(f"Unsupported collision geometry {geom}: {kind}")


def repair(source, output, collision_report, strength=30., max_change=.10, foot_strength=60.):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output == source:
        raise FileExistsError(output)
    motion = load_motion(source)
    meta = motion.pop('metadata')
    model = mujoco.MjModel.from_xml_path(meta['mujoco_xml'])
    data = mujoco.MjData(model)
    names = motion['joint_names'].tolist()
    ankle_names = [f'{side}_ankle_{axis}_joint' for side in ('left', 'right') for axis in ('pitch', 'roll')]
    joint_ids = [names.index(n) for n in ankle_names]
    addresses = [int(model.joint(n).qposadr[0]) for n in names]
    q = np.zeros((len(motion['time_s']), model.nq))
    q[:, :3] = motion['root_pos_w']; q[:, 3:7] = motion['root_quat_xyzw'][:, [3, 0, 1, 2]]
    q[:, addresses] = motion['joint_pos']; original = q.copy()
    # Optimize four ankle angles and a small vertical root correction per frame.
    qcols = [int(model.joint(n).qposadr[0]) for n in ankle_names] + [2]
    dofs = [int(model.joint(n).dofadr[0]) for n in ankle_names] + [2]
    frames, width = len(q), len(qcols); size = frames * width
    D2 = sp.diags([np.ones(frames-2), -2*np.ones(frames-2), np.ones(frames-2)],
                  [0, 1, 2], shape=(frames-2, frames)).tocsc()
    tracking = sp.diags([1., 1., 1., 1., 2000.])
    smoothing = sp.diags([strength]*4 + [2000.])
    H = sp.kron(sp.eye(frames), tracking) + sp.kron(D2.T @ D2, smoothing)
    # Include world foot rotation smoothness: ankle angles alone can become
    # smoother while parent knee/hip motion still makes the foot jerk.
    foot_rows=[]; foot_cols=[]; foot_vals=[]; rotation_residual=[]
    for side_index, side in enumerate(('left', 'right')):
        body=int(model.body(side+'_ankle_roll_link').id)
        rotations=[]; jacobians=[]
        for pose in original:
            data.qpos[:]=pose; mujoco.mj_forward(model,data)
            rotations.append(data.xmat[body].reshape(3,3).copy())
            jac=np.zeros((3,model.nv));mujoco.mj_jacBody(model,data,None,jac,body)
            jacobians.append(jac[:,dofs])
        rotation=Rotation.from_matrix(np.asarray(rotations))
        increments=(rotation[1:]*rotation[:-1].inv()).as_rotvec()
        residual=np.diff(increments,axis=0)
        rotation_residual.extend(residual.reshape(-1))
        for t in range(frames-2):
            for f, coefficient in ((t,1.),(t+1,-2.),(t+2,1.)):
                for axis in range(3):
                    for k in np.flatnonzero(abs(jacobians[f][axis])>1e-12):
                        foot_rows.append(side_index*(frames-2)*3+t*3+axis)
                        foot_cols.append(f*width+int(k))
                        foot_vals.append(coefficient*jacobians[f][axis,k])
    M=sp.coo_matrix((foot_vals,(foot_rows,foot_cols)),shape=(2*(frames-2)*3,size)).tocsc()
    H=H+foot_strength*(M.T@M)
    P = sp.triu(H, format='csc')
    # Smooth the actual ankle trajectory, rather than just smoothing corrections.
    signal = original[:, qcols].copy(); signal[:, -1] = 0.
    linear = np.asarray(sp.kron(D2.T @ D2, smoothing) @ signal.reshape(-1)).reshape(-1)
    linear += foot_strength*np.asarray(M.T@np.asarray(rotation_residual)).reshape(-1)
    geoms, vertices = geometry(model)
    audit = json.loads(Path(collision_report).read_text())
    blocked = {tuple(sorted(pair)) for key in ('excluded_adjacent_pairs', 'excluded_authored_filters') for pair in audit[key]}
    def bodyname(g):
        n = model.body(int(model.geom_bodyid[g])).name
        return 'base_link' if n == 'pelvis' else n
    # Only pairs containing ankle-dependent geometry can change in this repair.
    affected = {g for g in geoms if 'ankle' in bodyname(g)}
    pairs = [(a, b) for a, b in itertools.combinations(geoms, 2)
             if (a in affected or b in affected) and model.geom_bodyid[a] != model.geom_bodyid[b]
             and tuple(sorted((bodyname(a), bodyname(b)))) not in blocked]
    low = np.array([model.joint(n).range[0] for n in ankle_names] + [-np.inf])
    high = np.array([model.joint(n).range[1] for n in ankle_names] + [np.inf])
    records = []; correction = np.zeros(size)
    for iteration in range(10):
        rows=[]; cols=[]; vals=[]; rhs=[]; min_ground=np.inf; min_self=0.
        def constraint(f, jac, distance, margin):
            j = np.asarray(jac)[dofs]
            if np.linalg.norm(j) < 1e-10:
                return
            r=len(rhs)
            for k in np.flatnonzero(abs(j) > 1e-12):
                rows.append(r); cols.append(f*width+int(k)); vals.append(-j[k])
            rhs.append(distance-margin-float(j @ correction[f*width:(f+1)*width]))
        for f in range(frames):
            data.qpos[:] = q[f]; mujoco.mj_forward(model, data)
            for g in geoms:
                point = lowest_point(model, data, g, vertices)
                height = float(point[2]) - meta['ground_z']; min_ground=min(min_ground,height)
                if height < .025:
                    jac=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jac,None,point,int(model.geom_bodyid[g]))
                    # Preserve existing sub-mm tolerances for unchanged body geometry.
                    constraint(f,jac[2],height,.0018)
            for a,b in pairs:
                # A broad distance cutoff avoids constructing irrelevant constraints.
                points=np.zeros(6);dist=float(mujoco.mj_geomDistance(model,data,a,b,.012,points))
                min_self=min(min_self,dist)
                if dist < .010:
                    delta=points[:3]-points[3:];norm=np.linalg.norm(delta)
                    if norm < 1e-12: continue
                    normal=delta/norm*(1 if dist>=0 else -1)
                    ja=np.zeros((3,model.nv));jb=ja.copy()
                    mujoco.mj_jac(model,data,ja,None,points[:3],int(model.geom_bodyid[a]))
                    mujoco.mj_jac(model,data,jb,None,points[3:],int(model.geom_bodyid[b]))
                    constraint(f,normal@(ja-jb),dist,0.)
        A=sp.coo_matrix((vals,(rows,cols)),shape=(len(rhs),size)).tocsc()
        lower=np.maximum(np.tile([-max_change]*4+[0.],frames), (low[None,:]-original[:,qcols]).reshape(-1))
        upper=np.minimum(np.tile([max_change]*4+[.01],frames), (high[None,:]-original[:,qcols]).reshape(-1))
        # Keep the exact first/last poses, including their local finite-difference velocities.
        lower[:2*width]=upper[:2*width]=0.;lower[-2*width:]=upper[-2*width:]=0.
        A=sp.vstack([A,sp.eye(size),-sp.eye(size)],format='csc');bvec=np.r_[rhs,upper,-lower]
        settings=clarabel.DefaultSettings();settings.verbose=False
        solution=clarabel.DefaultSolver(P,linear,A,bvec,[clarabel.NonnegativeConeT(len(bvec))],settings).solve()
        if str(solution.status) not in ('Solved','AlmostSolved'):
            raise RuntimeError(f'Foot repair QP failed: {solution.status}')
        new=np.asarray(solution.x);change=float(abs(new-correction).max());correction=new
        q=original.copy();q[:,qcols]+=correction.reshape(frames,width)
        record=dict(iteration=iteration,min_ground_m=min_ground,min_affected_self_m=min_self,max_update=change)
        records.append(record);print(record,flush=True)
        if iteration>0 and change<1e-6:break
    motion['joint_pos']=q[:,addresses].astype(np.float32)
    motion['root_pos_w']=q[:,:3].astype(np.float32)
    dt=float(meta['dt_s']);motion['joint_vel']=np.gradient(motion['joint_pos'],dt,axis=0).astype(np.float32)
    motion['root_lin_vel_w']=np.gradient(motion['root_pos_w'],dt,axis=0).astype(np.float32)
    motion['root_lin_vel_b']=Rotation.from_quat(motion['root_quat_xyzw']).inv().apply(motion['root_lin_vel_w']).astype(np.float32)
    policy_ids=motion['policy_joint_indices'];motion['policy_joint_pos']=motion['joint_pos'][:,policy_ids];motion['policy_joint_vel']=motion['joint_vel'][:,policy_ids]
    motion.update(export_mujoco_fk(meta['mujoco_xml'],names,motion['root_pos_w'],motion['root_quat_xyzw'],motion['joint_pos']))
    # Re-evaluate the final geometry independently of the optimization linearization.
    min_ground=np.inf;min_self=0.
    for pose in q:
        data.qpos[:]=pose;mujoco.mj_forward(model,data)
        min_ground=min(min_ground,min(float(lowest_point(model,data,g,vertices)[2])-meta['ground_z'] for g in geoms))
        for a,b in pairs:
            min_self=min(min_self,float(mujoco.mj_geomDistance(model,data,a,b,.001,None)))
    if min_ground < -.0001 or min_self < -.0001:
        raise RuntimeError(f'Final geometry check failed: ground={min_ground}, self={min_self}')
    metrics={}
    for j,n in zip(joint_ids,ankle_names):
        before=original[:,addresses[j]];after=motion['joint_pos'][:,j]
        metrics[n]={}
        for key,v in [('before',before),('after',after)]:
            metrics[n][key]={'peak_acceleration_rad_s2':float(abs(np.diff(v,2)/dt**2).max()),
                            'rms_acceleration_rad_s2':float(np.sqrt(np.mean((np.diff(v,2)/dt**2)**2)))}
        metrics[n]['max_change_rad']=float(abs(after-before).max())
    foot_metrics={}
    for side in ('left','right'):
        body_index=motion['body_names'].tolist().index(side+'_ankle_roll_link')
        original_motion=load_motion(source)
        foot_metrics[side]={}
        for label, payload in (('before',original_motion),('after',motion)):
            rotation=Rotation.from_quat(payload['body_quat_xyzw'][:,body_index])
            omega=(rotation[1:]*rotation[:-1].inv()).as_rotvec()/dt
            acceleration=np.diff(omega,axis=0)/dt
            foot_metrics[side][label]={
                'peak_rad_s2':float(np.linalg.norm(acceleration,axis=1).max()),
                'component_rms_rad_s2':float(np.sqrt(np.mean(acceleration**2))),
            }
    report=dict(foot_world_angular_acceleration=foot_metrics,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output=str(output),
                method='bounded ankle and world-foot-rotation second-difference QP; native ground/self constraints',
                foot_strength=foot_strength,
                strength=strength,max_ankle_change_rad=max_change,max_root_lift_m=float(correction.reshape(frames,width)[:,-1].max()),
                min_ground_m=min_ground,min_affected_self_m=min_self,frames=frames,metrics=metrics,iterations=records,
                limitations='Kinematic repair only; no dynamics/support validation. Original motion and training manifest untouched.')
    meta['foot_repair']=report.copy();motion['metadata_json']=np.array(json.dumps(meta,ensure_ascii=False))
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**motion)
    load_motion(output)
    output.with_suffix('.repair.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--collision-report',type=Path,required=True);p.add_argument('--strength',type=float,default=30.)
    p.add_argument('--max-change',type=float,default=.10)
    p.add_argument('--foot-strength',type=float,default=60.);a=p.parse_args()
    if not np.isfinite([a.strength,a.max_change,a.foot_strength]).all() or a.strength<0 or a.max_change<=0 or a.foot_strength<0:p.error('Invalid smoothing parameters')
    repair(a.input,a.output,a.collision_report,a.strength,a.max_change,a.foot_strength)

if __name__=='__main__':main()
