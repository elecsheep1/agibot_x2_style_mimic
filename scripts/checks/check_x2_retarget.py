"""Offline X2 retarget QA. No simulation stepping, training, or motion edits.

Canonical training NPZ or raw UMR qpos NPZ; raw MuJoCo quaternion order is wxyz.
Exit 0: no hard failure (warnings may remain), 1: failed/review with --strict,
2: an input could not be checked. See docs/retarget_quality_check.md.
"""

# Support direct execution: python scripts/<category>/<tool>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import itertools
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from common.x2_motion_utils import load_motion


DEFAULTS = dict(penetration_m=0.001, joint_limit_rad=1e-5,
                joint_acceleration_rad_s2=100.0, joint_jerk_rad_s3=5000.0,
                joint_high_frequency_rad=0.025, body_acceleration_m_s2=35.0,
                body_angular_acceleration_rad_s2=100.0, floating_m=0.015,
                contact_height_m=0.008, sliding_m_s=0.08,
                flat_foot_degrees=30.0, standing_root_height_m=0.45,
                standing_tilt_degrees=60.0, minimum_contact_duration_s=0.1)
AUTHORED_FILTERS = [
    ('left_ankle_roll_link', 'left_knee_link'),
    ('right_ankle_roll_link', 'right_knee_link'),
    ('left_wrist_roll_link', 'left_wrist_yaw_link'),
    ('right_wrist_roll_link', 'right_wrist_yaw_link'),
    ('head_pitch_link', 'torso_link')]


def events(mask, times, values, check, subject, severity, threshold):
    """Keep every contiguous excursion, including a single-frame discontinuity."""
    indices = np.flatnonzero(mask)
    result = []
    for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        if not len(group):
            continue
        peak = int(group[np.argmax(np.asarray(values)[group])])
        result.append(dict(check=check, subject=subject, severity=severity,
                           start_s=float(times[group[0]]), end_s=float(times[group[-1]]),
                           peak_s=float(times[peak]), peak=float(values[peak]),
                           threshold=threshold, frames=len(group)))
    return result


def load_input(path, model_override=None):
    with np.load(path, allow_pickle=False) as archive:
        canonical = 'metadata_json' in archive.files
        if not canonical:
            qpos = np.asarray(archive['qpos'], dtype=float)
            fps = float(archive['fps'].reshape(-1)[0])
            xml = model_override or str(archive['robot_xml'].item())
            ground = float(archive['ground_z'].reshape(-1)[0]) if 'ground_z' in archive else 0.
    if canonical:
        z = load_motion(path)
        xml = model_override or z['metadata']['mujoco_xml']
        ground = float(z['metadata'].get('ground_z', 0.))
    xml = Path(xml).expanduser()
    if not xml.is_absolute():
        xml = path.parent / xml
    model = mujoco.MjModel.from_xml_path(str(xml))
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
        raise ValueError('Expected X2 pelvis free joint first in the MuJoCo model')
    ids = [j for j in range(1, model.njnt)]
    if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in ids):
        raise ValueError('Only X2 free-root plus hinge joints are supported')
    names = [model.joint(j).name for j in ids]
    addresses = [int(model.jnt_qposadr[j]) for j in ids]
    if canonical:
        if set(z['joint_names'].tolist()) != set(names):
            raise ValueError('Motion/model joint sets differ')
        order = [z['joint_names'].tolist().index(n) for n in names]
        qpos = np.tile(model.qpos0, (len(z['time_s']), 1))
        qpos[:, :3] = z['root_pos_w']
        qpos[:, 3:7] = z['root_quat_xyzw'][:, [3, 0, 1, 2]]
        qpos[:, addresses] = z['joint_pos'][:, order]
        times = np.asarray(z['time_s'], dtype=float)
    else:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('fps must be finite and positive')
        times = np.arange(len(qpos)) / fps
        z = None
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) < 5:
        raise ValueError('Need at least five frames and qpos matching model.nq')
    if not np.isfinite(qpos).all() or not np.isfinite(times).all():
        raise ValueError('Nonfinite motion values')
    dt = np.diff(times)
    if np.any(dt <= 0) or not np.allclose(dt, dt[0], rtol=1e-4, atol=1e-7):
        raise ValueError('Time must increase uniformly')
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., atol=1e-5):
        raise ValueError('Root quaternions are not unit length')
    return model, qpos, times, names, ids, addresses, ground, z, xml


def geometry_lowest(model, data, g):
    """Lowest collision-surface point; never treat an ankle origin as the sole."""
    pos = data.geom_xpos[g]
    rot = data.geom_xmat[g].reshape(3, 3)
    size = model.geom_size[g]
    kind = model.geom_type[g]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start + count] @ rot.T + pos
        return vertices[np.argmin(vertices[:, 2])]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return pos - [0, 0, size[0]]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return pos + rot @ (-np.sign(rot[2]) * size)
    if kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        local = np.array([0., 0., -np.sign(rot[2, 2]) * size[1]])
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return pos + rot @ local - [0, 0, size[0]]
        radial = np.linalg.norm(rot[2, :2])
        if radial > 1e-12:
            local[:2] = -size[0] * rot[2, :2] / radial
        return pos + rot @ local
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        direction = size ** 2 * rot[2]
        return pos - rot @ (direction / np.linalg.norm(size * rot[2]))
    raise ValueError(f'Unsupported collision geom type {kind}: {model.geom(g).name}')


def angular_acceleration(quats, dt):
    rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
    velocity = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / dt
    return np.linalg.norm(np.diff(velocity, axis=0) / dt, axis=1)


def audit(path, config, model_override=None, contacts=None):
    model, qp, times, names, ids, adr, ground, z, xml = load_input(path, model_override)
    if contacts is not None and (not isinstance(contacts, dict) or set(contacts) != {'left', 'right'}):
        raise ValueError('Contact schedule must contain exactly left and right interval lists')
    dt = times[1] - times[0]
    issues = []
    def add(mask, ts, values, check, subject, severity, threshold):
        issues.extend(events(mask, ts, values, check, subject, severity, threshold))
    q = qp[:, adr]
    limits = model.jnt_range[ids]
    excess = np.maximum(limits[:, 0] - q, q - limits[:, 1])
    velocity = np.diff(q, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    residual = abs(q - gaussian_filter1d(q, sigma=max(1., .06 / dt), axis=0, mode='nearest'))
    for j, name in enumerate(names):
        if model.jnt_limited[ids[j]]:
            add(excess[:, j] > config['joint_limit_rad'], times, excess[:, j],
                'joint_limit_rad', name, 'error', config['joint_limit_rad'])
        for values, ts, key in [(abs(acceleration[:, j]), times[1:-1], 'joint_acceleration_rad_s2'),
                                (abs(jerk[:, j]), times[1:-2], 'joint_jerk_rad_s3'),
                                (residual[2:-2, j], times[2:-2], 'joint_high_frequency_rad')]:
            add(values > config[key], ts, values, key, name, 'warning', config[key])
        if name in config.get('joint_velocity_limits_rad_s', {}):
            limit = float(config['joint_velocity_limits_rad_s'][name])
            if not np.isfinite(limit) or limit <= 0:
                raise ValueError(f'Invalid velocity limit for {name}')
            add(abs(velocity[:, j]) > limit, times[1:], abs(velocity[:, j]),
                'joint_speed_rad_s', name, 'warning', limit)
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] and
             (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not geoms:
        raise ValueError('Model has no robot collision geoms')
    blocked = {tuple(sorted(p)) for p in config.get('excluded_body_pairs', AUTHORED_FILTERS)}
    excluded = []
    pairs = []
    for a, b in itertools.combinations(geoms, 2):
        ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
        pair = tuple(sorted((model.body(ba).name, model.body(bb).name)))
        adjacent = ba == bb or model.body_parentid[ba] == bb or model.body_parentid[bb] == ba
        # Authored MuJoCo body excludes are additional to the explicit X2 filters.
        signature = (min(ba, bb) << 16) + max(ba, bb)
        if adjacent or pair in blocked or signature in model.exclude_signature:
            excluded.append(pair)
        else:
            pairs.append((a, b))
    n = len(qp)
    positions = np.zeros((n, model.nbody, 3))
    quats = np.zeros((n, model.nbody, 4))
    heights = np.zeros((n, model.nbody)) + np.inf
    foot_points = np.zeros((n, 2, 3))
    foot_tilt = np.zeros((n, 2))
    feet = [model.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    if any(not any(model.geom_bodyid[g] == b for g in geoms) for b in feet):
        raise ValueError('Each foot must have collision geometry')
    ground_depth = np.zeros(n)
    collision = {}
    data = mujoco.MjData(model)
    for f in range(n):
        data.qpos[:] = qp[f]
        mujoco.mj_forward(model, data)
        positions[f], quats[f] = data.xpos, data.xquat
        for g in geoms:
            b = int(model.geom_bodyid[g])
            heights[f, b] = min(heights[f, b], geometry_lowest(model, data, g)[2] - ground)
        ground_depth[f] = max(0., -np.min(heights[f]))
        for side, b in enumerate(feet):
            # Mean sphere-bottom/geometry support points gives a sole proxy with
            # stable indexing. Sliding only evaluated for nearly flat, low feet.
            group = [g for g in geoms if model.geom_bodyid[g] == b]
            foot_points[f, side] = np.mean([geometry_lowest(model, data, g) for g in group], axis=0)
            foot_tilt[f, side] = np.degrees(np.arccos(np.clip(data.xmat[b].reshape(3, 3)[2, 2], -1, 1)))
        for a, b in pairs:
            if np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b]) > model.geom_rbound[a] + model.geom_rbound[b] + config['penetration_m']:
                continue
            distance = float(mujoco.mj_geomDistance(model, data, a, b, config['penetration_m'], None))
            if distance < -config['penetration_m']:
                key = (model.body(int(model.geom_bodyid[a])).name,
                       model.body(int(model.geom_bodyid[b])).name)
                collision.setdefault(key, np.zeros(n))[f] = max(collision.get(key, np.zeros(n))[f], -distance)
    add(ground_depth > config['penetration_m'], times, ground_depth, 'ground_penetration_m',
        'robot collision surface', 'error', config['penetration_m'])
    for pair, depths in collision.items():
        add(depths > config['penetration_m'], times, depths, 'self_penetration_m',
            ' / '.join(pair), 'error', config['penetration_m'])
    for b in range(1, model.nbody):
        name = model.body(b).name
        acc = np.linalg.norm(np.diff(positions[:, b], n=2, axis=0) / dt ** 2, axis=1)
        aa = angular_acceleration(quats[:, b], dt)
        for values, key in [(acc, 'body_acceleration_m_s2'), (aa, 'body_angular_acceleration_rad_s2')]:
            add(values > config[key], times[1:-1], values, key, name, 'warning', config[key])
    torso = model.body('torso_link').id
    torso_up = Rotation.from_quat(quats[:, torso, [1, 2, 3, 0]]).as_matrix()[:, 2, 2]
    standing = (qp[:, 2] - ground > config['standing_root_height_m']) & (torso_up > np.cos(np.radians(config['standing_tilt_degrees'])))
    # A raised foot is not an error during swing/hand-supported get-up. Only
    # infer floating when upright and ALL robot collision surfaces clear ground.
    clearance = heights.min(axis=1)
    floating = standing & (clearance > config['floating_m'])
    add(floating, times, clearance, 'inferred_unsupported_upright_m', 'all robot surfaces',
        'warning', config['floating_m'])
    for side, b in enumerate(feet):
        label = ('left', 'right')[side]
        stance = (heights[:, b] <= config['contact_height_m']) & (foot_tilt[:, side] < config['flat_foot_degrees'])
        if contacts is not None:
            stance = np.zeros(n, dtype=bool)
            for start, end in contacts.get(label, []):
                if not np.isfinite([start, end]).all() or end < start:
                    raise ValueError('Invalid contact interval')
                stance |= (times >= start) & (times <= end)
            add(stance & (heights[:, b] > config['floating_m']), times, heights[:, b],
                'scheduled_stance_floating_m', label, 'warning', config['floating_m'])
        speed = np.linalg.norm(np.diff(foot_points[:, side, :2], axis=0) / dt, axis=1)
        valid = stance[1:] & stance[:-1] & (foot_tilt[1:, side] < config['flat_foot_degrees']) & (foot_tilt[:-1, side] < config['flat_foot_degrees'])
        # Require a sustained stance, but retain brief speed spikes within it.
        for group in np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid)) > 1) + 1):
            if len(group) * dt < config['minimum_contact_duration_s']:
                valid[group] = False
        add(valid & (speed > config['sliding_m_s']), times[1:], speed,
            'sole_proxy_sliding_m_s', label, 'warning', config['sliding_m_s'])
    if z is not None:
        body_order = [model.body(str(name)).id for name in z['body_names']]
        error = np.linalg.norm(positions[:, body_order] - z['body_pos_w'], axis=2).max(axis=1)
        add(error > .001, times, error, 'stored_body_fk_position_m', 'body arrays', 'error', .001)
        stored = Rotation.from_quat(z['body_quat_xyzw'].reshape(-1, 4))
        actual = Rotation.from_quat(quats[:, body_order][:, :, [1, 2, 3, 0]].reshape(-1, 4))
        error = (stored * actual.inv()).magnitude().reshape(n, -1).max(axis=1)
        add(error > .001, times, error, 'stored_body_fk_rotation_rad', 'body arrays', 'error', .001)
    issues.sort(key=lambda e: (e['severity'] != 'error', e['start_s'], e['check'], e['subject']))
    status = 'fail' if any(e['severity'] == 'error' for e in issues) else ('review' if issues else 'pass')
    return dict(motion=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                model=str(xml), frames=n, duration_s=float(times[-1]-times[0]), fps=float(1/dt),
                status=status, thresholds=config, events=issues,
                excluded_collision_body_pairs=sorted(set(excluded)),
                contact_mode='user schedule' if contacts is not None else 'geometry heuristic',
                minimum_surface_clearance_m=float(clearance.min()),
                peak_joint_speed_rad_s=dict(zip(names, abs(velocity).max(axis=0).tolist())),
                unchecked=['Dynamic feasibility, actuator torque and balance stability',
                           'Human/IK target reachability without original target residuals',
                           'Isaac USD collision hulls; native MuJoCo collision geometry only'] +
                          ([] if all(name in config.get('joint_velocity_limits_rad_s', {}) for name in names)
                           else ['Joint speed limits missing for: ' + ', '.join(name for name in names if name not in config.get('joint_velocity_limits_rad_s', {}))]),
                note='pass means only these offline checks passed; warning spikes can be intentional motion')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion', type=Path, action='append', help='Repeat for multiple NPZ files')
    p.add_argument('--manifest', type=Path, help='Training motions.json; default active prone_getup manifest')
    p.add_argument('--model', type=Path, help='Override MuJoCo XML')
    p.add_argument('--config', type=Path, help='JSON threshold overrides and joint_velocity_limits_rad_s')
    p.add_argument('--urdf', type=Path, help='Read named joint speed limits from the matching robot URDF')
    p.add_argument('--contacts', type=Path, help='Single-motion JSON: {left:[[start,end]],right:[[start,end]]}')
    p.add_argument('--output-dir', type=Path, default=Path('logs/retarget_quality'))
    p.add_argument('--strict', action='store_true', help='Warnings also cause exit 1')
    a = p.parse_args()
    config = dict(DEFAULTS)
    if a.config:
        overrides = json.loads(a.config.read_text())
        unknown = set(overrides) - set(DEFAULTS) - {'joint_velocity_limits_rad_s', 'excluded_body_pairs'}
        if unknown:
            p.error(f'Unknown config keys: {sorted(unknown)}')
        config.update(overrides)
    for key in DEFAULTS:
        if not isinstance(config[key], (int, float)) or not np.isfinite(config[key]) or config[key] <= 0:
            p.error(f'{key} must be finite and positive')
    if a.urdf:
        speed_limits = {}
        for joint in ET.parse(a.urdf).getroot().findall('joint'):
            limit = joint.find('limit')
            if limit is not None and 'velocity' in limit.attrib:
                speed_limits[joint.attrib['name']] = float(limit.attrib['velocity'])
        if not speed_limits:
            p.error('URDF has no joint velocity limits')
        config['joint_velocity_limits_rad_s'] = {**speed_limits, **config.get('joint_velocity_limits_rad_s', {})}
    paths = list(a.motion or [])
    if a.manifest or not paths:
        manifest = a.manifest or Path(__file__).resolve().parents[2] / 'data/motions/prone_getup/motions.json'
        paths.extend(manifest.parent / entry['file'] for entry in json.loads(manifest.read_text())['motions'])
    paths = list(dict.fromkeys(path.expanduser().resolve() for path in paths))
    if a.contacts and len(paths) != 1:
        p.error('--contacts requires exactly one motion')
    contacts = json.loads(a.contacts.read_text()) if a.contacts else None
    a.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in paths:
        print(f'Checking {path.name} ...', flush=True)
        try:
            result = audit(path, config, str(a.model.resolve()) if a.model else None, contacts)
        except Exception as error:
            result = dict(motion=str(path), status='invalid', error=f'{type(error).__name__}: {error}', events=[])
        results.append(result)
        suffix = hashlib.sha256(str(path).encode()).hexdigest()[:8]
        (a.output_dir / f'{path.stem}_{suffix}.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"  {result['status']}: {len(result['events'])} events" + (f"; {result['error']}" if 'error' in result else ''), flush=True)
    (a.output_dir / 'summary.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
    lines = ['# X2 retarget 离线质量检查', '', 'error 为硬约束/数据错误；warning 为待复查线索。通过不代表动力学可执行。', '']
    for r in results:
        lines += [f"## {Path(r['motion']).name}: {r['status']}", '', f"输入：`{r['motion']}`", '']
        if 'error' in r:
            lines += [r['error'], '']
            continue
        lines += ['|级别|检查|部位|时段 s|峰值时刻 s|峰值 / 阈值|', '|---|---|---|---|---|---|']
        for e in r['events']:
            lines.append(f"|{e['severity']}|{e['check']}|{e['subject']}|{e['start_s']:.3f}–{e['end_s']:.3f}|{e['peak_s']:.3f}|{e['peak']:.5g} / {e['threshold']:.5g}|")
        lines += ['', '未验证：' + '；'.join(r['unchecked']), '']
    (a.output_dir / 'report.md').write_text('\n'.join(lines))
    print(f'Report: {a.output_dir.resolve() / "report.md"}')
    return 2 if any(r['status'] == 'invalid' for r in results) else int(any(r['status'] == 'fail' or (a.strict and r['status'] == 'review') for r in results))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Configuration/input error: {error}', file=sys.stderr)
        raise SystemExit(2)
