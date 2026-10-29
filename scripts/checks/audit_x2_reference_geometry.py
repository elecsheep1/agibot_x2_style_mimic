"""Independent native-geometry audit of an exported X2 motion (no dynamics)."""

# Support direct execution: python scripts/<category>/<tool>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from repair.repair_x2_reference_feet import geometry,lowest_point
from common.x2_motion_utils import load_motion


def audit(path,collision_report,output):
    z=load_motion(path);meta=z['metadata'];model=mujoco.MjModel.from_xml_path(meta['mujoco_xml']);data=mujoco.MjData(model);names=z['joint_names'].tolist();addresses=[int(model.joint(n).qposadr[0]) for n in names]
    geoms,vertices=geometry(model);filters=json.loads(Path(collision_report).read_text());blocked={tuple(sorted(p)) for key in ('excluded_adjacent_pairs','excluded_authored_filters') for p in filters[key]}
    def name(g):
        n=model.body(int(model.geom_bodyid[g])).name;return 'base_link' if n=='pelvis' else n
    pairs=[(a,b) for a,b in itertools.combinations(geoms,2) if model.geom_bodyid[a]!=model.geom_bodyid[b] and tuple(sorted((name(a),name(b)))) not in blocked]
    minimum_ground=np.inf;minimum_self=0.;bad_ground=[];bad_self=[];heights=[]
    for f in range(len(z['time_s'])):
        data.qpos[:3]=z['root_pos_w'][f];data.qpos[3:7]=z['root_quat_xyzw'][f][[3,0,1,2]];data.qpos[addresses]=z['joint_pos'][f];mujoco.mj_forward(model,data)
        h=min(float(lowest_point(model,data,g,vertices)[2])-meta['ground_z'] for g in geoms);minimum_ground=min(minimum_ground,h)
        if h<-.0001:bad_ground.append(f)
        for a,b in pairs:
            if np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b])>model.geom_rbound[a]+model.geom_rbound[b]+.001:continue
            distance=float(mujoco.mj_geomDistance(model,data,a,b,.001,None));minimum_self=min(minimum_self,distance)
            if distance<-.0001:bad_self.append({'frame':f,'pair':[name(a),name(b)],'distance_m':distance})
        feet=[]
        for side in ('left','right'):
            group=[g for g in geoms if name(g)==side+'_ankle_roll_link'];feet.append(min(float(lowest_point(model,data,g,vertices)[2])-meta['ground_z'] for g in group))
        heights.append(feet)
    limits=np.array([model.joint(n).range for n in names]);q=z['joint_pos'];violation=float(np.maximum(limits[:,0]-q,q-limits[:,1]).max());velocity=np.diff(q,axis=0)/meta['dt_s'];vlim=np.array([np.inf]*len(names))
    r={'motion':str(path),'frames':len(q),'min_ground_m':minimum_ground,'min_self_m':minimum_self,'ground_violating_frames':bad_ground,'self_penetrations':bad_self,'max_hard_limit_violation_rad':max(0.,violation),'feet_min_height_per_frame_m':heights,'peak_joint_speed_rad_s':{n:float(abs(velocity[:,j]).max()) for j,n in enumerate(names)},'passed':not bad_ground and not bad_self and violation<1e-6,'scope':'native geometric constraints only; no dynamics or USD hull audit'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k not in ('feet_min_height_per_frame_m','peak_joint_speed_rad_s','self_penetrations')});print('self_penetrations',bad_self[:3])
    if not r['passed']:raise RuntimeError('Geometry audit failed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--collision-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.motion,a.collision_report,a.output)
