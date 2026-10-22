"""Offline reference timing audit; geometric foot proxies are not contact measurements."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(path):
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    meta = json.loads(str(d['metadata_json'].item()))
    t = d['time_s'] - d['time_s'][0]
    dt = float(np.median(np.diff(t)))
    height = d['root_pos_w'][:, 2] - float(meta['ground_z'])
    def tilt(q):
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        return np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 0]**2 + q[..., 1]**2), -1, 1)))
    torso_id = d['body_names'].tolist().index('torso_link')
    torso_tilt = tilt(d['body_quat_xyzw'][:, torso_id])
    root_tilt = tilt(d['root_quat_xyzw'])
    joint_ids = [d['joint_names'].tolist().index(n) for n in ['left_knee_joint', 'right_knee_joint']]
    knee = np.abs(d['joint_pos'][:, joint_ids])
    foot_ids = [d['body_names'].tolist().index(n) for n in ['left_ankle_roll_link', 'right_ankle_roll_link']]
    feet = d['body_pos_w'][:, foot_ids]
    foot_z = feet[..., 2] - float(meta['ground_z'])
    foot_speed = np.linalg.norm(np.gradient(feet, dt, axis=0), axis=-1)
    speed = np.linalg.norm(d['root_lin_vel_w'], axis=-1)
    low_feet = (foot_z < .12).all(axis=-1) & (foot_z > -.02).all(axis=-1)
    still_feet = (foot_speed < .2).all(axis=-1)
    candidate = (height > .5) & (root_tilt < np.degrees(.35)) & (torso_tilt < np.degrees(.55)) & (knee < 1.5).all(axis=-1)
    def first(mask, duration=.1):
        length = max(1, int(np.ceil(duration / dt))); run = 0
        for i, value in enumerate(mask):
            run = run + 1 if value else 0
            if run >= length: return round(float(t[i-length+1]), 3)
        return None
    rows = []
    for second in [0, 1, 2, 3, 4, 5, 6, 8, 10, float(t[-1])]:
        if second > t[-1] + 1e-6: continue
        i = int(np.abs(t-second).argmin())
        if rows and rows[-1]['frame'] == i: continue
        rows.append(dict(frame=i, time_s=float(t[i]), height=float(height[i]),
                         torso_tilt_deg=float(torso_tilt[i]), root_tilt_deg=float(root_tilt[i]),
                         knee_angles=d['joint_pos'][i, joint_ids].tolist(),
                         ankle_height=foot_z[i].tolist(), ankle_speed=foot_speed[i].tolist(),
                         root_speed=float(speed[i])))
    limits = d['joint_hard_limits']
    violation = (d['joint_pos'] < limits[:, 0] - 1e-5) | (d['joint_pos'] > limits[:, 1] + 1e-5)
    return dict(file=str(path), duration_s=float(t[-1]),
                first_height_040_s=first(height > .4), first_height_050_s=first(height > .5),
                first_knees_under_15_s=first((knee < 1.5).all(axis=-1)),
                first_low_still_ankles_proxy_s=first(low_feet & still_feet),
                first_pose_candidate_s=first(candidate),
                frames_with_joint_limit_violation=int(violation.any(axis=-1).sum()),
                initial_joint_limit_violations=int(violation[0].sum()), samples=rows,
                note='Ankle height/speed are geometric proxies, not measured contact or physical feasibility.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/motions/prone_getup/motions.json')
    parser.add_argument('--output', default='logs/reference_physics/multimotion_timing.json')
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    entries = json.loads(manifest.read_text())['motions']
    report = [analyze((manifest.parent / e['file']).resolve()) for e in entries if e.get('weight', 1) > 0]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[参考时序诊断] 以下时刻要求连续满足至少 0.1 秒；不是接触/动力学验收。')
    for r in report:
        print(f"\n{Path(r['file']).name} | 时长 {r['duration_s']:.2f} s")
        print(f"  高度>0.40: {r['first_height_040_s']} s | 高度>0.50: {r['first_height_050_s']} s"
              f" | 姿态候选: {r['first_pose_candidate_s']} s")
        print(f"  双踝低位静止代理: {r['first_low_still_ankles_proxy_s']} s"
              f" | 初始超限关节 {r['initial_joint_limit_violations']} | 超限帧 {r['frames_with_joint_limit_violation']}")
        for row in r['samples']:
            print(f"  t={row['time_s']:.2f} | 高度 {row['height']:.3f} | 躯干 {row['torso_tilt_deg']:.1f}°"
                  f" | 膝 {[round(v, 2) for v in row['knee_angles']]}"
                  f" | 踝高 {[round(v, 3) for v in row['ankle_height']]}")
    print(f'\n完整报告：{output.resolve()}')


if __name__ == '__main__':
    main()
