"""Train-only dynamic recovery states; deliberately separate from settled fallen banks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from common.fallen_bank import load_bank, sha256, TRAIN

SCHEMA = 'x2_pre_entrapment_v1'
FIELDS = ('root_pose', 'root_velocity', 'joint_pos', 'joint_vel', 'parent_row',
          'family_id', 'split', 'time_s', 'lead_seconds', 'trigger_pair',
          'ground_min_separation_m', 'self_min_separation_m')


def validate_rows(data, parent):
    n = len(data['root_pose'])
    j = len(data['joint_names'])
    for name, shape in {'root_pose': (n, 7), 'root_velocity': (n, 6),
                        'joint_pos': (n, j), 'joint_vel': (n, j),
                        'trigger_pair': (n, 2)}.items():
        if data[name].shape != shape:
            raise ValueError(f'Invalid {name} shape')
    for name in FIELDS:
        if not np.isfinite(data[name]).all():
            raise ValueError(f'Nonfinite {name}')
        if name not in ('root_pose', 'root_velocity', 'joint_pos', 'joint_vel', 'trigger_pair') and data[name].shape != (n,):
            raise ValueError(f'Invalid {name} shape')
    if not n or not np.allclose(np.linalg.norm(data['root_pose'][:, 3:], axis=1), 1., atol=1e-4):
        raise ValueError('Empty bank or invalid quaternion')
    rows = data['parent_row']
    if rows.dtype.kind not in 'iu' or (rows < 0).any() or (rows >= len(parent['split'])).any():
        raise ValueError('Invalid parent row')
    if not (data['split'] == TRAIN).all() or not (parent['split'][rows] == TRAIN).all():
        raise ValueError('Holdout contamination')
    if not np.array_equal(data['family_id'], parent['family_id'][rows]):
        raise ValueError('Family lineage mismatch')
    names = data['joint_names'].tolist()
    if len(set(names)) != len(names) or set(names) != set(parent['joint_names'].tolist()):
        raise ValueError('Invalid joint mapping')
    if (data['lead_seconds'] < 0).any() or (data['time_s'] < 0).any():
        raise ValueError('Invalid time')


def save(path, rows, joint_names, metadata):
    path = Path(path)
    if not rows:
        raise ValueError('No replay-validated candidates; inspect report')
    arrays = {k: np.stack([r[k] for r in rows]) for k in FIELDS}
    arrays['joint_names'] = np.asarray(joint_names)
    parent = load_bank(metadata['parent_bank'])
    if sha256(metadata['parent_bank']) != metadata['parent_sha256']:
        raise ValueError('Parent bank changed')
    validate_rows(arrays, parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays, metadata_json=np.asarray(json.dumps(dict(metadata, schema=SCHEMA))))


def load(path, parent_path, joint_names=None, *, require_reset_protocol=True):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files if k != 'metadata_json'}
        meta = json.loads(str(z['metadata_json'].item()))
    if meta.get('schema') != SCHEMA or meta.get('quat_order') != 'xyzw' or meta.get('velocity_frame') != 'world_axes' or meta.get('root_frame') != 'link_origin' or meta.get('position_frame') != 'environment_local':
        raise ValueError('Unsupported dynamic state conventions')
    if meta.get('parent_sha256') != sha256(parent_path):
        raise ValueError('Wrong parent bank')
    if meta.get('replay_validated') is not True:
        raise ValueError('Replay validation incomplete')
    if require_reset_protocol and meta.get('reset_protocol') != 'clear_action_and_observation_history_after_state_write_v1':
        raise ValueError('Dynamic states need revalidation with the production reset protocol')
    validate_rows(data, load_bank(parent_path))
    if joint_names is not None:
        names = data['joint_names'].tolist()
        if len(joint_names) != len(names) or set(joint_names) != set(names):
            raise ValueError('Joint sets differ')
        order = [names.index(name) for name in joint_names]
        for key in ('joint_pos', 'joint_vel'):
            data[key] = data[key][:, order]
        data['joint_names'] = np.asarray(joint_names)
    data['metadata'] = meta
    return data
