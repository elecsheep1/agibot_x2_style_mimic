"""Script-side fallen-bank schema v1. No dependency on mdp/getup_births.py."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'x2_fallen_bank_v1'
SOURCE_NEAR, SOURCE_FALL = 0, 1
TRAIN, HOLDOUT = 0, 1
ARRAY_FIELDS = ('root_pose', 'root_velocity', 'joint_pos', 'joint_vel', 'source_type',
                'motion_id', 'family_id', 'split', 'root_gravity_b', 'quality')
QUALITY_COLUMNS = ('ground_force_n', 'self_force_n', 'ground_min_separation_m',
                   'self_min_separation_m', 'root_linear_speed', 'root_angular_speed',
                   'joint_speed_rms', 'replay_peak_linear_speed', 'replay_peak_angular_speed')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def asset_hashes(root):
    """Fingerprint all USD layers and meshes in this asset directory."""
    root = Path(root)
    extensions = {'.usd', '.usda', '.usdc', '.stl', '.obj', '.ply'}
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix.lower() in extensions}


def family_split(seed, family, holdout_fraction):
    token = hashlib.sha256(f'{seed}:{family}'.encode()).digest()[:8]
    return int(int.from_bytes(token, 'big') / 2**64 < holdout_fraction)


def dedup_key(joint_pos, gravity, height):
    # Translation/yaw independent. Approximate quantized dedup, not a metric guarantee.
    q = np.rint(np.asarray(joint_pos) / .03).astype(np.int32)
    g = np.rint(np.asarray(gravity) / .05).astype(np.int32)
    return tuple(np.r_[q, g, int(round(float(height) / .02))].tolist())


def load_bank(path, *, joint_names=None):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files if k != 'metadata_json'}
        meta = json.loads(str(z['metadata_json'].item()))
    if meta.get('schema') != SCHEMA or meta.get('quat_order') != 'xyzw':
        raise ValueError('Unsupported fallen bank schema/quaternion convention')
    if meta.get('root_frame') != 'link_origin' or meta.get('position_frame') != 'environment_local':
        raise ValueError('Unsupported root or position frame')
    if meta.get('additional_height_offset_m') != 0.0:
        raise ValueError('Bank must contain settled poses, with no extra spawn height')
    names = data['joint_names'].tolist()
    if len(set(names)) != len(names):
        raise ValueError('Duplicate joint names')
    n, j = len(data['root_pose']), len(names)
    if not n:
        raise ValueError('Empty bank')
    shapes = {'root_pose': (n, 7), 'root_velocity': (n, 6), 'joint_pos': (n, j),
              'joint_vel': (n, j), 'source_type': (n,), 'motion_id': (n,),
              'family_id': (n,), 'split': (n,), 'root_gravity_b': (n, 3),
              'quality': (n, len(QUALITY_COLUMNS))}
    for k, shape in shapes.items():
        if data[k].shape != shape or not np.isfinite(data[k]).all():
            raise ValueError(f'Invalid array {k}: {data[k].shape}, expected {shape}')
    if not np.allclose(np.linalg.norm(data['root_pose'][:, 3:7], axis=1), 1., atol=1e-4):
        raise ValueError('Non-unit root quaternion')
    for k in ['source_type', 'split']:
        if not np.isin(data[k], [0, 1]).all():
            raise ValueError(f'Invalid {k}')
    if not np.isin(data['source_type'], [SOURCE_NEAR, SOURCE_FALL]).all():
        raise ValueError('Invalid source')
    near = data['source_type'] == SOURCE_NEAR
    if (data['motion_id'][near] < 0).any() or (data['motion_id'][~near] != -1).any():
        raise ValueError('Real falls must have motion_id=-1; near states need a source motion')
    train_families = set(data['family_id'][data['split'] == TRAIN].tolist())
    test_families = set(data['family_id'][data['split'] == HOLDOUT].tolist())
    if train_families & test_families:
        raise ValueError('Family leakage between train and holdout')
    if joint_names is not None:
        if len(joint_names) != j or set(joint_names) != set(names):
            raise ValueError('Bank/robot joint sets differ')
        order = [names.index(name) for name in joint_names]
        data['joint_pos'] = data['joint_pos'][:, order]
        data['joint_vel'] = data['joint_vel'][:, order]
        data['joint_names'] = np.asarray(joint_names)
    data['metadata'] = meta
    return data


def save_bank(path, rows, joint_names, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError('No validated states; inspect collection report')
    values = {k: np.stack([row[k] for row in rows]) for k in ARRAY_FIELDS}
    meta = dict(metadata, schema=SCHEMA, quat_order='xyzw', root_frame='link_origin',
                position_frame='environment_local', velocity_frame='world_axes',
                additional_height_offset_m=0.0, quality_columns=list(QUALITY_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **values, joint_names=np.asarray(joint_names),
                            metadata_json=np.asarray(json.dumps(meta, ensure_ascii=False)))
    return load_bank(path)


def summary(data):
    return {'count': len(data['root_pose']), 'by_source_split': {
        f'{source}/{split}': int(((data['source_type'] == s) & (data['split'] == t)).sum())
        for s, source in [(SOURCE_NEAR, 'near_reference'), (SOURCE_FALL, 'simulated_fall')]
        for t, split in [(TRAIN, 'train'), (HOLDOUT, 'holdout')]}}


def reset_from_bank_for_eval(
    env, env_ids, bank_path, bank_split, bank_source, manifest_file, policy_joint_names,
    start_phase=0.0, end_phase=0.0, use_reference_velocity=False,
    zero_initial_frame_velocity=False, height_offset=0.01,
    initial_frame_probability=0.5, transition_frame_ranges=None,
    perturb_initial_probability=0.0, perturb_state_bank=None, bank_rows=None, bank_center_xy=False,
):
    """Independent evaluation reset. Never calls the user's unfinished bank loader."""
    import torch
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.getup import (
        reset_from_motion_dataset, _write_reference_state,
    )
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    # Initialize reference metadata needed by GetUpAssist without bank augmentation.
    reference_params = dict(manifest_file=manifest_file,policy_joint_names=policy_joint_names,
        use_reference_velocity=use_reference_velocity,zero_initial_frame_velocity=zero_initial_frame_velocity,
        height_offset=height_offset,initial_frame_probability=initial_frame_probability,
        perturb_initial_probability=0.,perturb_state_bank=None,
        transition_frame_ranges=None,start_phase=0.,end_phase=0.)
    reset_from_motion_dataset(env, ids, **reference_params)
    if not hasattr(env, '_fallen_eval_bank'):
        data = load_bank(bank_path, joint_names=env.scene['robot'].joint_names)
        from x2_style_mimic.assets.x2_ultra_cfg import X2_ULTRA_USD
        if data['metadata']['asset_hashes'] != asset_hashes(X2_ULTRA_USD.parent):
            raise ValueError('Bank asset fingerprint differs from current robot')
        selected = data['split'] == (HOLDOUT if bank_split == 'holdout' else TRAIN)
        if bank_source != 'all':
            selected &= data['source_type'] == (SOURCE_NEAR if bank_source == 'near' else SOURCE_FALL)
        rows = np.flatnonzero(selected)
        if bank_rows is not None:
            requested = np.asarray(bank_rows, dtype=np.int64)
            if len(set(requested.tolist())) != len(requested) or not np.isin(requested, rows).all():
                raise ValueError('Requested bank rows are duplicate or outside selected split/source')
            rows = requested
        if not len(rows):
            raise ValueError('No bank states in requested source/split')
        env._fallen_eval_bank = (data, rows)
        env._fallen_eval_counter = torch.arange(env.num_envs, device=env.device)
        env._fallen_eval_serial = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        env._fallen_eval_index = torch.full_like(env._fallen_eval_serial, -1)
    data, rows = env._fallen_eval_bank
    serial = env._fallen_eval_counter[ids].clone()
    index = rows[(serial % len(rows)).cpu().numpy()]
    env._fallen_eval_counter[ids] += env.num_envs
    env._fallen_eval_serial[ids] = serial
    env._fallen_eval_index[ids] = torch.as_tensor(index, device=env.device)
    def tensor(key):
        return torch.as_tensor(data[key][index], dtype=torch.float32, device=env.device)
    pose = tensor('root_pose')
    if bank_center_xy:
        pose[:, :2] = 0.0
    pose[:, :3] += env.scene.env_origins[ids]
    _write_reference_state(env, ids, env.scene['robot'], pose, tensor('root_velocity'),
                           tensor('joint_pos'), tensor('joint_vel'))
    source = torch.as_tensor(data['source_type'][index], device=env.device)
    env._getup_birth_types[ids] = torch.where(source == SOURCE_NEAR, 1, 3)
    env._getup_reference_motion_ids[ids] = torch.as_tensor(data['motion_id'][index], device=env.device)
