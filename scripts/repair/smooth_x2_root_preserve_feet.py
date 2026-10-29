"""Smooth pelvis translation while retaining both feet's original world poses."""

# Support direct execution: python scripts/<category>/<tool>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from repair.repair_x2_reference_local import export, smoothstep
from common.x2_motion_utils import load_motion


def repair(source, output, interval=(3.9,5.8), sigma=.10, smooth_rotation=False):
    z=load_motion(source);meta=z.pop('metadata');dt=meta['dt_s'];t=z['time_s']
    model=mujoco.MjModel.from_xml_path(meta['mujoco_xml']);data=mujoco.MjData(model)
    names=z['joint_names'].tolist();addresses=[int(model.joint(n).qposadr[0]) for n in names]
    selected=[j for j,n in enumerate(names) if any(k in n for k in ('hip_','knee_','ankle_'))]
    cols=np.array(addresses)[selected];dofs=[int(model.joint(names[j]).dofadr[0]) for j in selected]
    limits=np.array([model.joint(names[j]).range for j in selected])
    feet=[model.body(s+'_ankle_roll_link').id for s in ('left','right')]
    original=z['root_pos_w'].copy()
    blend=smoothstep((t-interval[0])/.2)*smoothstep((interval[1]-t)/.2)
    target=original+(gaussian_filter1d(original,sigma/dt,axis=0,mode='nearest')-original)*blend[:,None]
    root_rotation=Rotation.from_quat(z['root_quat_xyzw'])
    if smooth_rotation:
        base=root_rotation[int(round(np.mean(interval)/dt))]
        tangent=(root_rotation*base.inv()).as_rotvec()
        filtered=gaussian_filter1d(tangent,sigma/dt,axis=0,mode='nearest')
        smoothed_rotation=Rotation.from_rotvec(tangent+(filtered-tangent)*blend[:,None])*base
    else:smoothed_rotation=root_rotation
    positions=[];rotations=[]
    q=np.zeros((len(t),model.nq));q[:,:3]=original;q[:,3:7]=z['root_quat_xyzw'][:,[3,0,1,2]];q[:,addresses]=z['joint_pos']
    for pose in q:
        data.qpos[:]=pose;mujoco.mj_forward(model,data)
        positions.append(data.xpos[feet].copy());rotations.append(data.xmat[feet].reshape(2,3,3).copy())
    max_error=0.
    for f in np.flatnonzero(blend>0):
        data.qpos[:]=q[f];data.qpos[:3]=target[f]
        data.qpos[3:7]=smoothed_rotation[f].as_quat()[[3,0,1,2]]
        desired=[Rotation.from_matrix(rotations[f][s]) for s in range(2)]
        def residual(x):
            data.qpos[cols]=x;mujoco.mj_forward(model,data)
            return np.concatenate([np.r_[10*(data.xpos[b]-positions[f][s]),
                (Rotation.from_matrix(data.xmat[b].reshape(3,3))*desired[s].inv()).as_rotvec()] for s,b in enumerate(feet)])
        solution=least_squares(residual,np.clip(q[f,cols],limits[:,0]+1e-8,limits[:,1]-1e-8),
                               bounds=(limits[:,0],limits[:,1]),max_nfev=150,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        error=float(abs(residual(solution.x)).max());max_error=max(max_error,error)
        if error>1e-4:raise RuntimeError(f'Foot pose cannot be retained at {t[f]}: {error}')
        z['joint_pos'][f,selected]=solution.x
    z['root_pos_w']=target.astype(np.float32)
    z['root_quat_xyzw']=smoothed_rotation.as_quat().astype(np.float32)
    export(source,output,z,meta,{'mode':'root translation smoothing with fixed world-foot IK',
           'interval_s':interval,'sigma_s':sigma,'smooth_rotation':smooth_rotation,'max_weighted_foot_error':max_error,
           'max_root_change_m':float(np.linalg.norm(target-original,axis=1).max())})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    repair(a.input,a.output)
