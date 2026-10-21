"""X2 参考数据约定；只依赖 NumPy，可在 gmr 环境离线使用。"""

import ast
import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
ENV_CFG_FILE = (
    PROJECT_DIR / "source/x2_style_mimic/x2_style_mimic/tasks/manager_based"
    / "x2_style_mimic/x2_style_mimic_env_cfg.py"
)


def policy_joint_names():
    """读取已有策略顺序，不导入会注册 Isaac Lab 环境的包。"""
    for node in ast.parse(ENV_CFG_FILE.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "POLICY_JOINT_NAMES" for target in node.targets
        ):
            names = ast.literal_eval(node.value)
            if len(names) != 29 or len(set(names)) != len(names):
                raise ValueError("策略关节列表必须含 29 个唯一名称")
            return names
    raise ValueError("环境文件中没有找到 POLICY_JOINT_NAMES")


def joint_indices(source_names, target_names):
    """严格按名称重排，不依赖两个模拟器的树遍历顺序。"""
    source_names, target_names = list(source_names), list(target_names)
    if len(set(source_names)) != len(source_names) or len(set(target_names)) != len(target_names):
        raise ValueError("关节名称存在重复")
    missing = set(target_names) - set(source_names)
    if missing:
        raise ValueError(f"参考缺少关节：{sorted(missing)}")
    return np.array([source_names.index(name) for name in target_names], dtype=np.int64)


def load_motion(path):
    """读取无 pickle 的 NPZ，并验证后续回放所需的基本约定。"""
    with np.load(path, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(motion["metadata_json"].item()))
    if metadata["schema_version"] != 1 or metadata["quat_order"] != "xyzw":
        raise ValueError("不支持的参考数据格式或四元数顺序")
    if metadata["root_frame"] != "pelvis" or metadata["up_axis"] != "Z":
        raise ValueError("参考根坐标系不符合当前 X2 约定")
    times = motion["time_s"]
    dt = float(metadata["dt_s"])
    if len(times) < 3 or not np.isfinite(dt) or dt <= 0 or not np.allclose(np.diff(times), dt):
        raise ValueError("参考时间必须具有固定正步长且至少三帧")
    names = motion["joint_names"].tolist()
    joint_indices(names, names)
    shapes = {
        "root_pos_w": (len(times), 3), "root_quat_xyzw": (len(times), 4),
        "root_lin_vel_w": (len(times), 3), "root_ang_vel_w": (len(times), 3),
        "root_lin_vel_b": (len(times), 3), "root_ang_vel_b": (len(times), 3),
        "projected_gravity_b": (len(times), 3),
        "joint_pos": (len(times), len(names)), "joint_vel": (len(times), len(names)),
    }
    for key, shape in shapes.items():
        if motion[key].shape != shape or not np.isfinite(motion[key]).all():
            raise ValueError(f"{key} 的形状或数值不合法")
    if not np.allclose(np.linalg.norm(motion["root_quat_xyzw"], axis=1), 1, atol=1e-6):
        raise ValueError("根四元数未归一化")
    policy_ids = joint_indices(names, motion["policy_joint_names"].tolist())
    for key in ("pos", "vel"):
        if not np.array_equal(motion[f"policy_joint_{key}"], motion[f"joint_{key}"][:, policy_ids]):
            raise ValueError(f"policy_joint_{key} 与全关节数据不一致")
    motion["metadata"] = metadata
    return motion
