"""多动作参考库：供重置器和 AMP 共用。"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from .amp import AMPReferenceMotion, name_indices

class MotionDataset:
    def __init__(
        self,
        manifest_file,
        robot_joint_names,
        policy_joint_names,
        device,
        step_dt=0.02,
    ):
        self.device = torch.device(device)
        self.step_dt = float(step_dt)
        self.paths = []
        self.clips = []
        self.amp_states = []

        manifest = Path(manifest_file).expanduser().resolve()
        entries = json.loads(manifest.read_text(encoding="utf-8"))["motions"]

        if not entries:
            raise ValueError("motions 清单不能为空")

        robot_joint_names = list(robot_joint_names)
        policy_joint_names = list(policy_joint_names)
        weights = []

        for entry in entries:
            path = (manifest.parent / entry["file"]).resolve()
            weight = float(entry.get("weight", 1.0))

            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"{path.name}: 权重必须非负且有限")
            if path in self.paths:
                raise ValueError(f"{path.name}: 重复的路径")
            if not path.is_file():
                raise FileNotFoundError(f"{path.name}: 文件不存在")

            # 复用原加载器：检查时间间隔、坐标格式，
            # 按名称映射策略关节和 AMP 关键刚体。
            amp_motion  =AMPReferenceMotion(
                motion_file=str(path),
                joint_names=policy_joint_names,
                device=self.device,
                step_dt=self.step_dt,
            )

            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(
                    str(data["metadata_json"].item())
                )
                source_names = data["joint_names"].tolist()

                if (len(source_names) != len(robot_joint_names)
                    or set(source_names) != set(robot_joint_names)
                ):
                    raise ValueError(f"{path.name}: 完整关节集合与机器人不一致")

                joint_ids = name_indices(source_names, robot_joint_names)
                count = len(data["time_s"])
                if count < 5:
                    raise ValueError(f"{path.name}: 参考动作至少需要 5 帧")

                def read_tensor(key, expected_shape):
                    values = np.asarray(data[key])
                    if values.shape != expected_shape:
                        raise ValueError( f"{path.name}: {key} 形状为 "
                                        f"{values.shape}，期望 {expected_shape}")
                    if not np.isfinite(values).all():
                        raise ValueError(f"{path.name}: {key} 包含非有限值")
                    return torch.tensor(
                        values,
                        dtype=torch.float32,
                        device=self.device,
                    )

                root_pos = read_tensor("root_pos_w", (count, 3))
                ground_z = float(metadata["ground_z"])
                if not math.isfinite(ground_z):
                    raise ValueError(f"{path.name}: ground_z 必须为有限数")
                # 所有动作统一为地面 z=0。
                root_pos[:, 2] -= ground_z

                root_quat = read_tensor("root_quat_xyzw", (count, 4))
                quat_norm = root_quat.norm(dim=-1, keepdim=True)
                if bool((quat_norm < 1e-6).any()):
                    raise ValueError(f"{path.name}: root_quat_xyzw 包含零四元数")
                root_quat = root_quat / quat_norm

                root_lin_vel = read_tensor("root_lin_vel_w", (count, 3))
                root_ang_vel = read_tensor("root_ang_vel_w", (count, 3))
                joint_pos = read_tensor("joint_pos", (count, len(robot_joint_names)))[:, joint_ids]
                joint_vel = read_tensor("joint_vel", (count, len(robot_joint_names)))[:, joint_ids]

                clip = {
                    "root_pose": torch.cat((root_pos, root_quat), dim=-1),
                    "root_velocity": torch.cat((root_lin_vel, root_ang_vel), dim=-1),
                    "joint_pos": joint_pos,
                    "joint_vel": joint_vel,
                }

            self.paths.append(path)
            self.clips.append(clip)
            self.amp_states.append(amp_motion.states)
            weights.append(weight)

        if sum(weights) <= 0.0:
            raise ValueError("motions 权重总和必须大于零")

        self.weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=self.device,
        )
        self.weights /= self.weights.sum()  # 归一化为概率分布

        self.lengths = torch.tensor(
            [len(clip["root_pose"]) for clip in self.clips],
            dtype=torch.long,
            device=self.device,
        )

    def _sample_motion_ids(self, count):
        """按权重采样动作索引。"""
        if count <= 0:
            raise ValueError("count 必须为正数")

        return torch.multinomial(
            self.weights,
            num_samples=count,
            replacement=True,
        )

    def sample_reset_states(
        self,
        num_envs,
        start_phase=0.0,
        end_phase=0.0,
        initial_frame_probability=0.5,
        transition_frame_ranges=None,
    ):
        """先选动作，再选择首帧或该动作的转换帧。"""
        if not (0.0 <= start_phase <= end_phase <= 1.0):
            raise ValueError(
                "必须满足 0 <= start_phase <= end_phase <= 1"
            )

        motion_ids = self._sample_motion_ids(num_envs)
        last = self.lengths[motion_ids] - 1

        if transition_frame_ranges is None:
            # 保留原采样方式，兼容现有训练和 play。
            lower = torch.floor(start_phase * last).long()
            upper = torch.floor(end_phase * last).long()
        else:
            if not (0.0 <= initial_frame_probability <= 1.0):
                raise ValueError(
                    "initial_frame_probability 必须在 [0, 1] 内"
                )

            if len(transition_frame_ranges) != len(self.clips):
                raise ValueError(
                    "转换区间数量必须与 motions 中的动作数量一致"
                )

            for motion_id, bounds in enumerate(
                transition_frame_ranges
            ):
                if len(bounds) != 2:
                    raise ValueError("每个转换区间必须包含起止帧")

                lo, hi = bounds
                length = int(self.lengths[motion_id].item())

                if (
                    not isinstance(lo, int)
                    or not isinstance(hi, int)
                    or not 0 <= lo <= hi < length
                ):
                    raise ValueError(
                        f"动作 {motion_id} 的转换区间无效："
                        f"{bounds}，总帧数 {length}"
                    )

            ranges = torch.tensor(
                transition_frame_ranges,
                dtype=torch.long,
                device=self.device,
            )

            lower = ranges[motion_ids, 0]
            upper = ranges[motion_ids, 1]

        # 在每个环境对应的闭区间 [lower, upper] 内均匀抽帧。
        frame_ids = lower + torch.floor(
            torch.rand(num_envs, device=self.device)
            * (upper - lower + 1)
        ).long()

        if transition_frame_ranges is not None:
            # 每个环境独立决定是否改为该动作第0帧。
            use_initial = (
                torch.rand(num_envs, device=self.device)
                < initial_frame_probability
            )
            frame_ids = torch.where(
                use_initial,
                torch.zeros_like(frame_ids),
                frame_ids,
            )

        result = {
            key: torch.empty(
                (num_envs, *value.shape[1:]),
                dtype=value.dtype,
                device=self.device,
            )
            for key, value in self.clips[0].items()
        }

        for motion_id, clip in enumerate(self.clips):
            rows = torch.where(motion_ids == motion_id)[0]
            frames = frame_ids[rows]

            for key in result:
                result[key][rows] = clip[key][frames]

        result["motion_ids"] = motion_ids
        result["frame_ids"] = frame_ids
        return result

    def sample_amp_windows(self, batch_size, num_steps=5):
        """返回 [batch, time, feature]，窗口不跨动作边界。"""
        if num_steps < 2:
            raise ValueError("AMP 窗口至少需要两帧")
        if bool((self.lengths < num_steps).any()):
            raise ValueError("motions 中至少有一条动作长度小于 num_steps")

        motion_ids = self._sample_motion_ids(batch_size)
        valid_starts = self.lengths[motion_ids] - num_steps + 1
        start_frames = torch.floor(
            torch.rand(batch_size, device=self.device) * valid_starts
        ).long()

        feature_dim = self.amp_states[0].shape[-1]
        windows = torch.empty(
            (batch_size, num_steps, feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        offsets = torch.arange(num_steps, device=self.device)

        for motion_id, states in enumerate(self.amp_states):
            rows = torch.where(motion_ids == motion_id)[0]
            indices = start_frames[rows, None] + offsets[None, :]
            windows[rows] = states[indices]

        return {
            "states": windows,
            "motion_ids": motion_ids,
            "start_frames": start_frames,
        }