"""Project smoothed late support feet onto the floor with bounded corrections."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import mujoco
from scipy.optimize import least_squares
from common.x2_motion_utils import load_motion
from repair.repair_x2_reference_feet import geometry, lowest_point
from repair.repair_x2_reference_local import export, smoothstep

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    p.error('Output exists')
z = load_motion(a.input)
meta = z.pop('metadata')
m = mujoco.MjModel.from_xml_path(meta['mujoco_xml'])
d = mujoco.MjData(m)
names = z['joint_names'].tolist()
adr = [int(m.joint(n).qposadr[0]) for n in names]
ids = [names.index(s + n) for s in ('left', 'right') for n in ('_hip_pitch_joint', '_knee_joint', '_ankle_pitch_joint', '_ankle_roll_joint')]
geoms, vertices = geometry(m)
feet = [[g for g in geoms if m.body(m.geom_bodyid[g]).name == s + '_ankle_roll_link'] for s in ('left', 'right')]
for f in np.flatnonzero(z['time_s'] > 5.8):
    base = z['root_pos_w'][f].astype(float).copy()
    q = z['joint_pos'][f].astype(float).copy()
    original = q.copy()
    def residual(x):
        q[ids] = original[ids] + x[1:]
        d.qpos[:3] = base + [0., 0., x[0]]
        d.qpos[3:7] = z['root_quat_xyzw'][f][[3, 0, 1, 2]]
        d.qpos[adr] = q
        mujoco.mj_forward(m, d)
        gaps = [min(lowest_point(m, d, g, vertices)[2] for g in group) - meta['ground_z'] for group in feet]
        return np.r_[(np.array(gaps) - .002) * 1000., x * 2.]
    lower = np.r_[-.06, np.maximum(-.12, [m.joint(names[i]).range[0] - original[i] for i in ids])]
    upper = np.r_[.06, np.minimum(.12, [m.joint(names[i]).range[1] - original[i] for i in ids])]
    fit = least_squares(residual, np.clip(np.zeros(9), lower + 1e-8, upper - 1e-8), bounds=(lower, upper), diff_step=1e-4, max_nfev=60)
    residual(fit.x)
    z['root_pos_w'][f] = base + [0., 0., fit.x[0]]
    z['joint_pos'][f] = q
export(a.input, a.output, z, meta, {'mode': 'bounded bilateral support projection'})
