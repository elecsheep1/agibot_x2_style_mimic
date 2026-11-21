"""Repair the fourth retimed get-up's final bilateral support transition."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from common.x2_motion_utils import load_motion
from repair.repair_x2_reference_feet import geometry, lowest_point
from repair.repair_x2_reference_local import export, smoothstep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file')
    z = load_motion(args.input)
    meta = z.pop('metadata')
    model = mujoco.MjModel.from_xml_path(meta['mujoco_xml'])
    data = mujoco.MjData(model)
    names = z['joint_names'].tolist()
    addresses = [int(model.joint(n).qposadr[0]) for n in names]
    legs = [i for i, n in enumerate(names) if any(k in n for k in ('hip_', 'knee_', 'ankle_'))]
    times = z['time_s']
    original = z['joint_pos'].astype(float).copy()
    blend = smoothstep((times - 5.8) / .4)
    filtered = gaussian_filter1d(original, .12 / meta['dt_s'], axis=0, mode='nearest')
    desired = original + blend[:, None] * (filtered - original)
    # Release the hands from the thighs as bilateral foot support takes over.
    arm_blend = smoothstep((times - 6.4) / .7)
    for index, name in enumerate(names):
        if any(part in name for part in ('shoulder_', 'elbow_', 'wrist_')):
            target = .3 if 'elbow_' in name else 0.
            if 'shoulder_roll' in name:
                target = .25 if name.startswith('left_') else -.25
            target = float(np.clip(target, *model.joint(name).range))
            desired[:, index] += arm_blend * (target - desired[:, index])
    geoms, vertices = geometry(model)
    feet = [model.body(s + '_ankle_roll_link').id for s in ('left', 'right')]

    def forward(position, joints, frame):
        data.qpos[:3] = position
        data.qpos[3:7] = z['root_quat_xyzw'][frame][[3, 0, 1, 2]]
        data.qpos[addresses] = joints
        mujoco.mj_forward(model, data)

    # Anchor each foot near its first sustained support posture. Flatten the
    # complete foot collision mesh, including heel and forefoot support.
    targets = []
    for body, start in zip(feet, (6.1, 5.9)):
        frame = int(round((start + .35) / meta['dt_s']))
        forward(z['root_pos_w'][frame], original[frame], frame)
        rotation = Rotation.from_euler('z', Rotation.from_matrix(data.xmat[body].reshape(3, 3)).as_euler('xyz')[2])
        position = data.xpos[body].copy()
        local_points = []
        for geom in geoms:
            if model.geom_bodyid[geom] == body and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE:
                center = model.geom_pos[geom]
                local_points.append(np.array([center + [0., 0., -model.geom_size[geom, 0]]]))
            if model.geom_bodyid[geom] == body and geom in vertices:
                world = vertices[geom] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                local_points.append((world - data.xpos[body]) @ data.xmat[body].reshape(3, 3))
        if not local_points:
            raise ValueError('Foot collision mesh missing')
        points = np.concatenate(local_points)
        position[2] = meta['ground_z'] + .0015 - rotation.apply(points)[:, 2].min()
        targets.append((position, rotation, start))
    low = np.r_[[-1., -1., -.25], [model.joint(names[i]).range[0] + 1e-6 for i in legs]]
    high = np.r_[[1., 1., .25], [model.joint(names[i]).range[1] - 1e-6 for i in legs]]
    max_error = 0.
    previous = None
    for frame in np.flatnonzero(times > 5.8):
        base = z['root_pos_w'][frame].astype(float).copy()
        q = desired[frame].copy()
        forward(base, original[frame], frame)
        near_ground = [g for g in geoms if lowest_point(model, data, g, vertices)[2] - meta['ground_z'] < .08]
        poses = [(data.xpos[b].copy(), Rotation.from_matrix(data.xmat[b].reshape(3, 3))) for b in feet]
        goals = []
        for (pos, rot), (target, target_rot, start) in zip(poses, targets):
            weight = float(smoothstep((times[frame] - start) / .55))
            goal_rot = Slerp([0, 1], Rotation.from_matrix([rot.as_matrix(), target_rot.as_matrix()]))([weight])[0]
            goals.append((pos + weight * (target - pos), goal_rot))
        def residual(value):
            q[legs] = value[3:]
            forward(base + value[:3], q, frame)
            error = []
            for body, (pos, rot) in zip(feet, goals):
                error.extend((data.xpos[body] - pos) * 300.)
                error.extend((rot.inv() * Rotation.from_matrix(data.xmat[body].reshape(3, 3))).as_rotvec() * 30.)
            error.extend(value[:3] * 3.)
            error.extend((value[3:] - desired[frame, legs]) * .3)
            for geom in near_ground:
                gap = lowest_point(model, data, geom, vertices)[2] - meta['ground_z']
                error.append(min(gap - .0015, 0.) * 5000.)
            return error
        initial = np.r_[np.zeros(3), desired[frame, legs]] if previous is None else np.r_[np.zeros(3), previous]
        fit = least_squares(residual, np.clip(initial, low + 1e-7, high - 1e-7), bounds=(low, high), diff_step=1e-4, max_nfev=150)
        residual(fit.x)
        max_error = max(max_error, max(np.linalg.norm(data.xpos[b] - goal[0]) for b, goal in zip(feet, goals)))
        z['root_pos_w'][frame] = base + fit.x[:3]
        z['joint_pos'][frame] = q
        previous = q[legs].copy()
        if frame % 25 == 0:
            print(f'frame {frame}/{len(times)} error={max_error:.6f}', flush=True)
    export(args.input, args.output, z, meta, {'mode': 'final bilateral foot pose constraints and temporal leg smoothing', 'start_s': 5.8, 'max_foot_position_error_m': max_error})


if __name__ == '__main__':
    main()
