"""Summarize targeted before/after measurements for the prone_getup2 repair."""

# Support direct execution: python scripts/<category>/<tool>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from common.x2_motion_utils import load_motion

ROOT=Path(__file__).resolve().parents[2]
WINDOWS=[
 [('left_wrist_roll_link',2.6,3.1),('right_hip_roll_link',2.9,3.5)],
 [('right_elbow_link',0,.2),('left_wrist_roll_link',4,4.6),('right_wrist_roll_link',6,6.4)],
 [('right_elbow_link',0,.2),('left_elbow_link',3,4),('right_wrist_roll_link',8.3,8.9)],
 [('left_wrist_roll_link',1.6,2.1),('left_hip_roll_link',3.8,4.3),('torso_link',5,7)],
]


def angular_peak(z,body,start,end):
    dt=z['metadata']['dt_s'];j=z['body_names'].tolist().index(body)
    R=Rotation.from_quat(z['body_quat_xyzw'][:,j])
    v=(R[1:]*R[:-1].inv()).as_rotvec()/dt
    a=np.linalg.norm(np.diff(v,axis=0)/dt,axis=1)
    t=z['time_s'][1:-1];mask=(t>=start)&(t<=end)
    return float(a[mask].max())


def main():
    directory=ROOT/'data/motions/prone_getup2'
    manifest=json.loads((directory/'motions.json').read_text())
    rows=[]
    for i,entry in enumerate(manifest['motions']):
        before=load_motion(ROOT/'data/motions/prone_getup'/entry['file'])
        after=load_motion(directory/entry['file'])
        support=json.loads((directory/entry['file']).with_suffix('.repair.json').read_text())
        row={'motion':entry['file'],'body_angular_acceleration_rad_s2':[],
             'max_joint_change_rad':float(abs(before['joint_pos']-after['joint_pos']).max()),
             'max_root_translation_change_m':float(np.linalg.norm(before['root_pos_w']-after['root_pos_w'],axis=1).max()),
             'feet':support['feet']}
        for body,start,end in WINDOWS[i]:
            row['body_angular_acceleration_rad_s2'].append({'body':body,'interval_s':[start,end],
                'before':angular_peak(before,body,start,end),'after':angular_peak(after,body,start,end)})
        rows.append(row)
    (directory/'repair_comparison.json').write_text(json.dumps(rows,indent=2))
    lines=['# prone_getup2 修复对照','','原始 prone_getup 未修改；这是独立运动学修复数据集。','',
           '|动作|部位|时段 s|修复前角加速度 rad/s²|修复后|','|---|---|---|---|---|']
    for row in rows:
        for item in row['body_angular_acceleration_rad_s2']:
            lines.append(f"|{row['motion']}|{item['body']}|{item['interval_s']}|{item['before']:.2f}|{item['after']:.2f}|")
    lines+=['','完整脚部接触稳定性、改动幅度见 repair_comparison.json；告警和未验证项见质量报告。',
            '支撑阶段为人工指定，仍不代表动力学或控制策略验收通过。']
    (directory/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
