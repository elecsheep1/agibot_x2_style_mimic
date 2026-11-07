from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

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


class FallenStateBank:
    def __init__(
        self,
        path,
        robot_joint_names,
        device,
        motion_paths,
        asset_root,
    ):
        data = load_bank(path, joint_names=robot_joint_names)
        meta = data["metadata"]
        report = meta.get("collection_report", {})

        if not (
            report.get("complete") is True
            and report.get("training_ready") is True
        ):
            raise ValueError("状态库未完成或尚未满足训练覆盖要求")
        if asset_hashes(asset_root) != meta.get("asset_hashes"):
            raise ValueError("当前机器人资产与状态库生成资产不一致")

        reference_files = meta.get("reference_files")
        reference_hashes = meta.get("reference_sha256")

        if (
            not isinstance(reference_files, list)
            or not isinstance(reference_hashes, list)
            or len(reference_files) != len(motion_paths)
            or len(reference_hashes) != len(motion_paths)
        ):
            raise ValueError("状态库参考动作元数据不完整")
        for motion_id, path in enumerate(motion_paths):
            path = Path(path)
            if path.name != Path(reference_files[motion_id]).name:
                raise ValueError(f"动作 {motion_id} 的参考顺序或文件名与状态库不一致")
            if sha256(path) != reference_hashes[motion_id]:
                raise ValueError(f"动作 {motion_id} 的参考文件内容与状态库不一致")
        self.device = torch.device(device)
        self.data = data

        train = data["split"] == TRAIN
        self.near_rows = {}
        for motion_id in range(len(motion_paths)):
            rows = np.flatnonzero(
                train
                & (data["source_type"] == SOURCE_NEAR)
                & (data["motion_id"] == motion_id)
            )
            self.near_rows[motion_id] = torch.as_tensor(
                rows, device=self.device, dtype=torch.long
            )
        self.fall_rows = torch.as_tensor(
            np.flatnonzero(train & (data["source_type"] == SOURCE_FALL)),
            device=self.device,
            dtype=torch.long,
        )
        if self.fall_rows.numel() == 0:
            raise ValueError("训练分区没有仿真摔倒状态")
        self.tensors = {
            name: torch.as_tensor(
                data[name],
                device=self.device,
                dtype=(
                    torch.long
                    if name in ("source_type", "motion_id")
                    else torch.float32
                ),
            )
            for name in (
                "root_pose",
                "root_velocity",
                "joint_pos",
                "joint_vel",
                "source_type",
                "motion_id",
            )
        }

    def sample(self, motion_ids, source_types):
        if (
            motion_ids.shape != source_types.shape
            or motion_ids.ndim != 1
            or motion_ids.device != self.device
            or source_types.device != self.device
        ):
            raise ValueError("状态库抽样参数形状或设备不一致")
        if not bool(((source_types == 0) | (source_types == 1)).all()):
            raise ValueError("source_types 只能为 0 或 1")

        index = torch.empty_like(motion_ids, dtype=torch.long)

        near = source_types == SOURCE_NEAR
        for motion_id in motion_ids[near].unique().tolist():
            selected = near & (motion_ids == motion_id)
            pool = self.near_rows.get(motion_id)
            if pool is None or pool.numel() == 0:
                raise ValueError(f"动作 {motion_id} 没有近邻训练状态")
            draw = torch.randint(
                pool.numel(),
                (int(selected.sum().item()),),
                device=self.device,
            )
            index[selected] = pool[draw]

        fall = source_types == SOURCE_FALL
        if bool(fall.any()):
            draw = torch.randint(
                self.fall_rows.numel(),
                (int(fall.sum().item()),),
                device=self.device,
            )
            index[fall] = self.fall_rows[draw]

        result = {
            name: values[index].clone()
            for name, values in self.tensors.items()
        }
        result["bank_row"] = index
        return result