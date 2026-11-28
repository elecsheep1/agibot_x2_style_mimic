"""X2 参考库与可微 FK 适配：与 AMP 加载器和起身出生库无关。"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..supermimic.kinematics import DifferentiableKinematics
from .rotations import to_xyzw, to_wxyz, quat_apply, quat_apply_inverse, projected_gravity, heading_quat

TRACKED_BODIES = (
    'base_link', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link',
    'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'torso_link',
    'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link',
    'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link',
)


class X2Kinematics(nn.Module):
    """用 USD 校准 URDF 关节原点；输入跟踪状态，输出世界系刚体速度。"""
    def __init__(self, urdf, joint_names, control_ids, usd=None):
        super().__init__()
        self.fk = DifferentiableKinematics(urdf, joint_names, TRACKED_BODIES)
        if usd is not None:
            self.fk.align_joint_origins_from_usd(usd)
        self.register_buffer('control_ids', torch.as_tensor(control_ids, dtype=torch.long))

    def body_indices(self, names):
        return torch.tensor([TRACKED_BODIES.index(n) for n in names], device=self.control_ids.device)

    def body_index(self, name):
        return TRACKED_BODIES.index(name)

    def forward(self, state):
        shape = state.shape[:-1]
        s = state.reshape(-1, 75)
        q = s[:, 3:7]
        pose = torch.cat((s[:, :3], to_xyzw(q)), -1)
        velocity = torch.cat((s[:, 38:41], quat_apply(q, s[:, 41:44])), -1)
        poses, velocities = self.fk(pose, velocity, s[:, 7:38], s[:, 44:75])
        def shaped(x):
            return x.reshape(*shape, *x.shape[1:])
        return dict(pos=shaped(poses[..., :3]), quat=shaped(to_wxyz(poses[..., 3:])),
                    lin_vel=shaped(velocities[..., :3]), ang_vel=shaped(velocities[..., 3:]))


@dataclass
class Reference:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor


@dataclass
class Bodies:
    pos_w: torch.Tensor
    quat_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor


class MotionLibrary:
    """固定控制周期的数据，名称重排后缓存 FK；所有参考帧使用同一运动学口径。"""
    def __init__(self, manifest, joint_names, default_joint_pos, fk, dt=0.02, bin_frames=50):
        self.fk = fk
        self.control_ids = fk.control_ids
        self.default_joint_pos = default_joint_pos.detach().clone()
        self.device = default_joint_pos.device
        path = Path(manifest).expanduser().resolve()
        entries = json.loads(path.read_text())['motions']
        if not entries:
            raise ValueError('参考动作清单不能为空')
        self.paths, clips = [], []
        for entry in entries:
            motion_path = (path.parent / entry['file']).resolve()
            with np.load(motion_path, allow_pickle=False) as data:
                names = data['joint_names'].tolist()
                if len(names) != 31 or set(names) != set(joint_names):
                    raise ValueError(f'参考关节集合与 X2 不一致：{motion_path}')
                order = [names.index(name) for name in joint_names]
                times = np.asarray(data['time_s'])
                if len(times) < 2 or not np.allclose(np.diff(times), dt, atol=1e-5):
                    raise ValueError(f'参考周期必须是 {dt} 秒：{motion_path}')
                def tensor(key, width):
                    value = np.asarray(data[key])
                    if value.shape != (len(times), width) or not np.isfinite(value).all():
                        raise ValueError(f'参考字段形状或数值无效：{key}')
                    return torch.as_tensor(value, device=self.device, dtype=torch.float32).clone()
                p = tensor('root_pos_w', 3)
                ground = float(json.loads(str(data['metadata_json'].item()))['ground_z'])
                if not np.isfinite(ground):
                    raise ValueError('参考地面高度必须有限')
                p[:, 2] -= ground
                q = tensor('root_quat_xyzw', 4)
                if (q.norm(dim=-1) < 1e-6).any():
                    raise ValueError('参考包含零四元数')
                q = to_wxyz(q / q.norm(dim=-1, keepdim=True))
                v = tensor('root_lin_vel_w', 3)
                w = quat_apply_inverse(q, tensor('root_ang_vel_w', 3))
                clips.append(torch.cat((p, q, tensor('joint_pos',31)[:,order], v, w,
                                        tensor('joint_vel',31)[:,order]), -1))
                self.paths.append(str(motion_path))
        self.states = torch.cat(clips)
        lengths = torch.tensor([len(c) for c in clips], device=self.device)
        self.starts = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)[:-1]))
        self.ends = self.starts + lengths - 1
        self.frame_clip = torch.repeat_interleave(torch.arange(len(clips), device=self.device), lengths)
        self.last = self.ends[self.frame_clip]
        # 避免一次 FK 整个大型动作库造成峰值显存过大。
        chunks = [fk(self.states[i:i+1024]) for i in range(0,len(self.states),1024)]
        self.body_data = {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}
        self.bin_starts = torch.cat([torch.arange(int(s),int(e)+1,bin_frames,device=self.device)
                                     for s,e in zip(self.starts,self.ends)])
        self.bin_ends = torch.minimum(self.bin_starts + bin_frames - 1, self.last[self.bin_starts])
        self.frame_bin = torch.empty(len(self.states), device=self.device, dtype=torch.long)
        for i,(s,e) in enumerate(zip(self.bin_starts,self.bin_ends)):
            self.frame_bin[s:e+1] = i
        self.num_bins = len(self.bin_starts)

    def ref(self, frames):
        s = self.states[frames]
        return Reference(s[..., :3], s[..., 3:7])

    def bodies(self, frames):
        b = self.body_data
        return Bodies(b['pos'][frames], b['quat'][frames], b['lin_vel'][frames], b['ang_vel'][frames])

    def last_frame_of(self, frames):
        return self.last[frames]

    def goal_window(self, frame, count):
        # 目标含当前帧及其后的 count-1 帧，与复现项目相同。
        ids = torch.minimum(frame[...,None]+torch.arange(count,device=self.device),self.last[frame][...,None])
        s = self.states[ids]
        features = torch.cat((s[...,7:38]-self.default_joint_pos,s[...,44:75],projected_gravity(s[...,3:7]),
                              s[...,41:44],quat_apply_inverse(heading_quat(s[...,3:7]),s[...,38:41]),s[...,2:3]),-1)
        return features.flatten(-2)
