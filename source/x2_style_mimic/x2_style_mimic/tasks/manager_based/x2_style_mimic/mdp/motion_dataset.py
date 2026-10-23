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
        self.paths = []         # 参考动作文件路径列表
        self.clips = []         # 参考动作片段字典列表，包含 root_pose、root_velocity、joint_pos、joint_vel
        self.amp_states = []    # AMP 观测状态列表，供 AMP 判别器训练使用

        manifest = Path(manifest_file).expanduser().resolve()   # 绝对路径
        entries = json.loads(manifest.read_text(encoding="utf-8"))["motions"]   # 读取 motions 清单

        if not entries:
            raise ValueError("motions 清单不能为空")

        # 读取动作
        robot_joint_names = list(robot_joint_names)
        policy_joint_names = list(policy_joint_names)
        weights = []

        for entry in entries:
            path = (manifest.parent / entry["file"]).resolve()  # 获取动作文件的绝对路径
            weight = float(entry.get("weight", 1.0))            # 获取动作的权重，默认为 1.0

            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"{path.name}: 权重必须非负且有限")
            if path in self.paths:
                raise ValueError(f"{path.name}: 重复的路径")
            if not path.is_file():
                raise FileNotFoundError(f"{path.name}: 文件不存在")

            # 复用原加载器：检查时间间隔、坐标格式，
            # 按名称映射策略关节和 AMP 关键刚体。
            # 使用 AMPReferenceMotion 加载当前动作。生成 AMP 需要的状态
            amp_motion = AMPReferenceMotion(
                motion_file=str(path),
                joint_names=policy_joint_names,
                device=self.device,
                step_dt=self.step_dt,
            )

            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                source_names = data["joint_names"].tolist()

                if (len(source_names) != len(robot_joint_names)
                    or set(source_names) != set(robot_joint_names)
                ):
                    raise ValueError(f"{path.name}: 完整关节集合与机器人不一致")

                # 按名称映射索引，确保关节顺序与机器人一致。
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
                # 读取关节位置和速度，并按机器人关节顺序重排。
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

        # 计算每条动作的帧数，供采样时使用。
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
        start_phase=0.0,                    # 起始相位
        end_phase=0.0,                      # 结束相位
        initial_frame_probability=0.5,      # 出生时使用动作第0帧的概率
        transition_frame_ranges=None,       # 每条动作的转换帧区间 [(lo, hi), ...]，None 时使用 start/end_phase
    ):
        """先选动作，再选择首帧或该动作的转换帧。"""
        if not (0.0 <= start_phase <= end_phase <= 1.0):
            raise ValueError("必须满足 0 <= start_phase <= end_phase <= 1")

        # 根据动作权重，给每个环境选择一条动作。
        motion_ids = self._sample_motion_ids(num_envs)
        # 取得每个环境所选动作的最后一帧编号。
        last = self.lengths[motion_ids] - 1

        if transition_frame_ranges is None:
            # 保留原采样方式，兼容现有训练和 play。
            lower = torch.floor(start_phase * last).long()
            upper = torch.floor(end_phase * last).long()
        else:
            if not (0.0 <= initial_frame_probability <= 1.0):
                raise ValueError("initial_frame_probability 必须在 [0, 1] 内")
            if len(transition_frame_ranges) != len(self.clips):
                raise ValueError("转换区间数量必须与 motions 中的动作数量一致")

            for motion_id, bounds in enumerate(transition_frame_ranges):
                if len(bounds) != 2:
                    raise ValueError("每个转换区间必须包含起止帧")

                lo, hi = bounds
                # 取得当前动作总帧数。
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
                (num_envs, *value.shape[1:]),       # 它去掉了原始动作的第一维“帧数”。
                dtype=value.dtype,                  # 把原来的：[帧数, 特征数] 变成了：[环境数, 特征数]
                device=self.device,
            )
            for key, value in self.clips[0].items() # 借用 motions 中第一条动作的特征维度和数据类型。
        }

        for motion_id, clip in enumerate(self.clips):
            # 找出使用当前动作的环境
            rows = torch.where(motion_ids == motion_id)[0]
            # 取出这些环境对应的帧编号
            frames = frame_ids[rows]

            # 遍历所有状态字段
            for key in result:
                # 写入结果对应环境的位置
                result[key][rows] = clip[key][frames]

        result["motion_ids"] = motion_ids
        result["frame_ids"] = frame_ids
        return result

    def sample_amp_windows(self, batch_size, num_steps=5):
        """返回 [batch, time, feature]，窗口不跨动作边界。用于 AMP 训练。"""
        if num_steps < 2:
            raise ValueError("AMP 窗口至少需要两帧")
        if bool((self.lengths < num_steps).any()):
            raise ValueError("motions 中至少有一条动作帧数小于 num_steps")

        motion_ids = self._sample_motion_ids(batch_size)            # 选择 batch_size 条动作。
        # 计算每个样本的合法起始帧数量
        '''
        对于每条动作，其合法的起始帧数量为：动作总帧数 - 窗口大小 + 1
        '''
        valid_starts = self.lengths[motion_ids] - num_steps + 1
        # 随机选择每个窗口的起始帧
        start_frames = torch.floor(
            torch.rand(batch_size, device=self.device) * valid_starts
        ).long()

        feature_dim = self.amp_states[0].shape[-1]
        windows = torch.empty(
            (batch_size, num_steps, feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        # 创建帧偏移量,表示相对于起始帧的偏移量。
        offsets = torch.arange(num_steps, device=self.device)

        for motion_id, states in enumerate(self.amp_states):
            rows = torch.where(motion_ids == motion_id)[0]          # 找出使用当前动作的环境
            indices = start_frames[rows, None] + offsets[None, :]   # 计算每个环境对应的窗口帧索引，形状为 [num_envs, num_steps]。
            windows[rows] = states[indices]                         # 将对应的窗口状态写入结果张量

        return {
            "states": windows,
            "motion_ids": motion_ids,
            "start_frames": start_frames,
        }