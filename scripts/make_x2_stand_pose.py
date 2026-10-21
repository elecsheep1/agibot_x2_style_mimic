"""Generate a stationary X2 stance for balance experiments, not a get-up transition."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull

from convert_x2_reference import export_mujoco_fk
from x2_motion_utils import load_motion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=float,default=.34,help='Lateral separation of the two sole-reference origins, m.')
    p.add_argument('--seconds',type=float,default=3.)
    a = p.parse_args()
    if a.output.exists() or a.output.resolve()==a.input.resolve():
        p.error('Choose a new output file; input/existing outputs are protected.')
    if not np.isfinite([a.width,a.seconds]).all() or not .2 <= a.width <= .6 or a.seconds < 1.:
        p.error('This X2 helper accepts width 0.2..0.6 m and duration >= 1 s.')
    src = load_motion(a.input)
    dt = float(src['metadata']['dt_s'])
    if not np.isclose(a.seconds/dt,round(a.seconds/dt)):
        p.error('seconds must align to the source dt.')
    xml = Path(src['metadata']['mujoco_xml'])
    m = mujoco.MjModel.from_xml_path(str(xml));d = mujoco.MjData(m)
    names = src['joint_names'].tolist()
    addr = np.array([int(m.joint(n).qposadr[0]) for n in names])
    bounds = np.array([m.jnt_range[m.joint(n).id] for n in names])
    legs = np.array([i for i,n in enumerate(names) if any(k in n for k in ('_hip_','_knee_','_ankle_'))])
    knees = [names.index(s+'_knee_joint') for s in ('left','right')]
    feet = [m.body(s+'_sole_reference').id for s in ('left','right')]
    foot_bodies = [m.body(s+'_ankle_roll_link').id for s in ('left','right')]
    pelvis = m.body('pelvis').id
    if len(names)!=31 or len(legs)!=12:
        raise ValueError('Expected the checked X2 31-joint model.')
    q = src['joint_pos'][-1].astype(float).copy()
    pos0 = src['root_pos_w'][-1].astype(float)
    r0 = R.from_quat(src['root_quat_xyzw'][-1])
    upright = R.from_euler('z',r0.as_euler('xyz')[2])

    def forward(pos,rot):
        d.qpos[:3]=pos;d.qpos[3:7]=rot.as_quat()[[3,0,1,2]];d.qpos[addr]=q
        mujoco.mj_forward(m,d)

    forward(pos0,r0)
    before = d.xpos[feet].copy()
    local = upright.inv().apply(before)
    old_width = float(local[0,1]-local[1,1])
    if old_width <= 0:
        raise ValueError('Unexpected left/right ordering; inspect coordinate frames.')
    center = local.mean(axis=0)
    local[0,1]=center[1]+a.width/2
    local[1,1]=center[1]-a.width/2
    target = upright.apply(local)
    foot_rot = [R.from_euler('z',R.from_matrix(d.xmat[i].reshape(3,3)).as_euler('xyz')[2]) for i in feet]
    ground = float(src['metadata'].get('ground_z',0.))
    vertices = []
    for side,(body,sole) in enumerate(zip(foot_bodies,feet)):
        geoms = np.where((m.geom_bodyid==body)&(m.geom_type==mujoco.mjtGeom.mjGEOM_MESH))[0]
        if not len(geoms):raise ValueError('Missing original foot mesh.')
        g=int(geoms[0]);mesh=m.geom_dataid[g]
        v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
        world=R.from_matrix(d.geom_xmat[g].reshape(3,3)).apply(v)+d.geom_xpos[g]
        v=R.from_matrix(d.xmat[sole].reshape(3,3)).inv().apply(world-d.xpos[sole])
        vertices.append(v)
        target[side,2]=ground+.001-foot_rot[side].apply(v)[:,2].min()
    com_target=target[:,:2].mean(axis=0)

    def error(x):
        q[legs]=x[3:]
        forward(x[:3],upright)
        e=list(((d.xpos[feet]-target)*100).ravel())
        for i,r in zip(feet,foot_rot):
            e.extend((r.inv()*R.from_matrix(d.xmat[i].reshape(3,3))).as_rotvec()*20)
        e.extend((q[knees]-.3)*.05)
        e.extend((d.subtree_com[pelvis,:2]-com_target)*.5)
        return e
    low=np.r_[pos0[:2]-.3,ground+.61,bounds[legs,0]+1e-5]
    high=np.r_[pos0[:2]+.3,ground+.85,bounds[legs,1]-1e-5]
    fit=least_squares(error,np.clip(np.r_[pos0,q[legs]],low+1e-7,high-1e-7),bounds=(low,high),
                      max_nfev=500,ftol=1e-12,xtol=1e-12,gtol=1e-12)
    error(fit.x)
    pe=float(np.linalg.norm(d.xpos[feet]-target,axis=1).max())
    ae=float(max(np.linalg.norm((r.inv()*R.from_matrix(d.xmat[i].reshape(3,3))).as_rotvec()) for i,r in zip(feet,foot_rot)))
    if not fit.success or pe>.001 or ae>np.deg2rad(.5) or abs(q[knees]).max()>=.45:
        raise RuntimeError('Endpoint IK failed; no output exported.')
    world=[];clearance=[]
    for i,r,v in zip(feet,foot_rot,vertices):
        w=r.apply(v)+d.xpos[i];clearance.append(float(w[:,2].min()-ground))
        # A projected sole-geometry footprint, not a measured contact polygon.
        world.append(w[w[:,2]<w[:,2].min()+.003])
    hull=ConvexHull(np.vstack(world)[:,:2])
    com=d.subtree_com[pelvis].copy()
    margin=float(-(hull.equations[:,:2]@com[:2]+hull.equations[:,2]).max())
    if margin <= 0:raise RuntimeError('COM projection is outside the approximate foot footprint.')
    count=round(a.seconds/dt)+1
    positions=np.repeat(fit.x[None,:3],count,axis=0)
    quats=np.repeat(upright.as_quat()[None],count,axis=0)
    joints=np.repeat(q[None],count,axis=0)
    zero=np.zeros((count,3));ids=src['policy_joint_indices']
    metadata=dict(src['metadata'])
    metadata.update(status='stationary_stance_candidate_NOT_dynamic_balance_validated',
        purpose='standing-only balance experiment; not a get-up reference or foot-reposition trajectory',
        source=str(a.input.resolve()),source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        source_frames=len(src['time_s']),source_fps=1/dt,source_duration_s=float(src['time_s'][-1]),
        duration_s=a.seconds,stance_width_m=a.width,loop=False,
        interpolation='one constrained IK pose repeated; all state velocities zero')
    out=dict(time_s=np.arange(count)*dt,joint_names=src['joint_names'],policy_joint_names=src['policy_joint_names'],
        policy_joint_indices=ids,root_pos_w=positions,root_quat_xyzw=quats,
        root_lin_vel_w=zero.copy(),root_ang_vel_w=zero.copy(),root_lin_vel_b=zero.copy(),root_ang_vel_b=zero.copy(),
        projected_gravity_b=np.tile([0.,0.,-1.],(count,1)),joint_pos=joints,joint_vel=np.zeros_like(joints),
        policy_joint_pos=joints[:,ids],policy_joint_vel=np.zeros_like(joints[:,ids]))
    out.update(export_mujoco_fk(xml,names,positions,quats,joints))
    for key,value in out.items():
        if value.dtype.kind=='f' and key!='time_s':out[key]=value.astype(np.float32)
    out['metadata_json']=np.array(json.dumps(metadata,ensure_ascii=False))
    report=dict(status=metadata['status'],purpose=metadata['purpose'],old_sole_width_m=old_width,
        new_sole_width_m=float(upright.inv().apply(d.xpos[feet[0]]-d.xpos[feet[1]])[1]),
        height_m=float(fit.x[2]-ground),knees_rad=q[knees].tolist(),foot_clearance_m=clearance,
        foot_position_error_m=pe,foot_rotation_error_deg=float(np.rad2deg(ae)),
        com_footprint_margin_m=margin,leg_joint_angles={names[i]:float(q[i]) for i in legs},
        kinematic_checks_passed=True,dynamic_balance_passed=None,self_collision_checked=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**out);load_motion(a.output)
    a.output.with_suffix('.stance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print('[SAVED]',a.output.resolve())


if __name__=='__main__':main()
