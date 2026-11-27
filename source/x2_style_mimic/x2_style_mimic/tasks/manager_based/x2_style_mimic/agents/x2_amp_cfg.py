"""新版多动作、五帧 AMP 训练配置。"""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import PPORunnerCfg
from ..x2_getup_env_cfg import AMP_MOTION_MANIFEST
from ..x2_style_mimic_env_cfg import POLICY_JOINT_NAMES

@configclass
class MultiMotionAMPAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = (
        "x2_style_mimic.algorithms.amp."
        "ppo_window:AMPWindowPPO"
    )

    amp_cfg: dict = {
        "manifest_file": str(AMP_MOTION_MANIFEST),

        # 完整关节集合用于重置数据检查。
        # MotionDataset 内部按名称映射，不依赖这里的排列。
        "robot_joint_names": list(POLICY_JOINT_NAMES) + [
            "head_yaw_joint",
            "head_pitch_joint",
        ],
        "joint_names": list(POLICY_JOINT_NAMES),

        "step_dt": 0.02,
        "num_steps": 5,
        "style_reward_mapping": "softplus",
        # 动态课程关闭时使用；启用后实际权重由课程维护。
        "style_reward_weight": 15.0,

        "weight_curriculum": {
            "enabled": False,
            "min_weight": 5.0,
            "max_weight": 20.0,
            "increment": 0.5,
            "check_interval": 100,
            "window_size": 2000,
            "min_episodes": 500,
            "raise_success_rate": 0.80,
            "raise_stable_rate": 0.75,
            "lower_success_rate": 0.60,
            "lower_stable_rate": 0.55,
        },

        # 环境通过 extras 提供门控；默认关闭，保持原版 AMP 行为。
        "reward_gate_enabled": True,
        "policy_sample_gate_enabled": True,

        "learning_rate": 5e-5,
        "gradient_penalty": 15.0,
        "trunk_weight_decay": 1.0e-3,
        "head_weight_decay": 1.0e-1,
        "normalization_clip": 10.0,

        "buffer_capacity": 500_000,
        "updates_per_iteration": 5,
        "batch_size": 4096,
        # 按本轮实际入池比例缩减判别器更新次数。
        "dynamic_updates_enabled": True,
    }


@configclass
class X2MultiMotionAMPRunnerCfg(PPORunnerCfg):
    experiment_name = "x2_prone_multimotion"
    run_name = "multimotion_v1"
    resume = False

    algorithm: MultiMotionAMPAlgorithmCfg = (
        MultiMotionAMPAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.002,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )
