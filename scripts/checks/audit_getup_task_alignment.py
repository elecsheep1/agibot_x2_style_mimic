"""静态对照 PPO 与 SuperMimic 任务配置，不启动仿真或修改正式代码。"""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "source/x2_style_mimic/x2_style_mimic"
TASK = PACKAGE / "tasks/manager_based/x2_style_mimic"


def class_node(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def assignments(node):
    result = {}
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            result[item.target.id] = item.value
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = item.value
    return result


def literal(node):
    return ast.literal_eval(node)


def audit():
    ppo = assignments(class_node(TASK / "x2_getup_env_cfg.py", "GetUpRewardsCfg"))
    imagined = assignments(class_node(PACKAGE / "algorithms/supermimic/heads.py", "GetupTaskCfg"))
    mapping = {
        "standing": "standing_weight", "action_rate": "action_rate_weight",
        "joint_pos_limits": "joint_limit_weight", "joint_acc": "joint_acc_weight",
        "out_of_bounds_failure": "boundary_weight",
    }
    weights = {}
    for name, term in ppo.items():
        if not isinstance(term, ast.Call):
            continue
        weight = next(literal(kw.value) for kw in term.keywords if kw.arg == "weight")
        key = mapping.get(name)
        candidate = literal(imagined[key]) if key else None
        if candidate is not None and weight < 0:
            candidate = -candidate
        weights[name] = {"ppo": weight, "supermimic": candidate,
                         "weight_equal": candidate == weight}
    algo = assignments(class_node(TASK / "agents/x2_amp_cfg.py", "MultiMotionAMPAlgorithmCfg"))
    amp = algo["amp_cfg"]
    amp_weights = {literal(key): value for key, value in zip(amp.keys, amp.values)}
    weights["style"] = {
        "ppo": literal(amp_weights["style_reward_weight"]),
        "supermimic": literal(imagined["style_weight"]),
    }
    weights["style"]["weight_equal"] = weights["style"]["ppo"] == weights["style"]["supermimic"]
    env = assignments(class_node(TASK / "x2_getup_env_cfg.py", "X2MultiMotionGetUpEnvCfg"))
    assist = {name: literal(value) for name, value in env.items() if name.startswith("assist_")}
    return {
        "scope": "源码默认配置静态审计；不代表历史运行配置或动力学验收",
        "weights": weights,
        "ppo_assist": assist,
        "semantic_differences": [
            "SuperMimic standing_quality 缺少双脚支撑、手部卸载与 limb_quality。",
            "头部接地与自碰撞惩罚缺失；运动学几何不能替代真实接触力。",
            "PPO 越界为硬事件，SuperMimic 使用 sigmoid 软越界代价。",
            "PPO 关节加速度来自仿真，SuperMimic 为控制周期速度差分。",
            "PPO AMP 有奖励门控和策略窗口采样门控；仅对齐权重不等于对齐目标。",
            "SuperMimic 当前环境无辅助课程；启用前需将辅助力及撤力锁存状态纳入模型条件。",
            "出生分布不同：PPO 五分支，SuperMimic 首帧/RSI 各 50%。",
        ],
        "fully_aligned": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
