"""Mirror X2 references across world XZ; infer joint signs from native axes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import hashlib
import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from common.x2_motion_utils import load_motion
from repair.repair_x2_reference_local import export


def opposite(name):
    if name.startswith('left_'): return 'right_' + name[5:]
    if name.startswith('right_'): return 'left_' + name[6:]
    return name


def mirror(source, output):
    z = load_motion(source); meta = z.pop('metadata'); meta = copy.deepcopy(meta)
    names = z['joint_names'].tolist()
    model = mujoco.MjModel.from_xml_path(meta['mujoco_xml'])
    d = mujoco.MjData(model); mujoco.mj_forward(model, d)
    S = np.diag([1., -1., 1.])
    permutation = [names.index(opposite(n)) for n in names]
    signs = []
    for name in names:
        target, origin = model.joint(name).id, model.joint(opposite(name)).id
        dot = float(d.xaxis[target] @ (-S @ d.xaxis[origin]))
        if abs(dot) < .9999: raise ValueError(f'Non-symmetric joint axes: {name}: {dot}')
        signs.append(1. if dot > 0 else -1.)
    original = z['joint_pos'].copy()
    z['joint_pos'] = (original[:, permutation] * np.array(signs)).astype(np.float32)
    double = z['joint_pos'][:, permutation] * np.array(signs)
    assert np.allclose(double, original, atol=1e-6)
    limits = np.array([model.jnt_range[model.joint(n).id] for n in names])
    excess = np.maximum(limits[:, 0] - z['joint_pos'], z['joint_pos'] - limits[:, 1])
    if excess.max() > 1e-4: raise ValueError(f'Mirrored joint limit violation: {excess.max()}')
    z['root_pos_w'] = (z['root_pos_w'] @ S).astype(np.float32)
    r = Rotation.from_quat(z['root_quat_xyzw']).as_matrix()
    q = Rotation.from_matrix(S @ r @ S).as_quat()
    for i in range(1, len(q)):
        if q[i] @ q[i-1] < 0: q[i] *= -1
    z['root_quat_xyzw'] = q.astype(np.float32)
    meta['mirror_augmentation'] = {'source': str(source.resolve()), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'plane': 'world XZ', 'joint_signs': dict(zip(names, signs)), 'body_states': 'recomputed using native FK; velocities recomputed from mirrored trajectories'}
    export(source, output, z, meta, {'method': 'XZ mirror with native-axis-derived left/right mapping', 'double_mirror_max_joint_error_rad': float(np.max(abs(double-original)))})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--manifest', type=Path, required=True); p.add_argument('--add-to-manifest', action='store_true'); a = p.parse_args()
    content = json.loads(a.manifest.read_text()); originals = [e for e in content['motions'] if '_mirror.' not in e['file']]
    pending = []
    for e in originals:
        source = a.manifest.parent / e['file']; output = source.with_name(source.stem + '_mirror.npz')
        mirror(source, output); pending.append(dict(file=output.name, weight=e.get('weight', 1.)))
    if a.add_to_manifest:
        existing = {e['file'] for e in content['motions']}
        content['motions'].extend(e for e in pending if e['file'] not in existing)
        a.manifest.write_text(json.dumps(content, ensure_ascii=False, indent=2)+'\n')
