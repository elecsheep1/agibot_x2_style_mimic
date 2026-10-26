"""新版多动作、五帧 AMP 训练配置。"""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import PPORunnerCfg
from ..x2_getup_env_cfg import MOTION_MANIFEST
from ..x2_style_mimic_env_cfg import POLICY_JOINT_NAMES

@configclass
class MultiMotionAMPAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = (
        "x2_style_mimic.algorithms.amp."
        "ppo_window:AMPWindowPPO"
    )

    amp_cfg: dict = {
        "manifest_file": str(MOTION_MANIFEST),

        # 完整关节集合用于重置数据检查。
        # MotionDataset 内部按名称映射，不依赖这里的排列。
        "robot_joint_names": list(POLICY_JOINT_NAMES) + [
            "head_yaw_joint",
            "head_pitch_joint",
        ],
        "joint_names": list(POLICY_JOINT_NAMES),

        "step_dt": 0.02,
        "num_steps": 5,
        "style_reward_weight": 30.0,

        # 环境通过 extras 提供门控；默认关闭，保持原版 AMP 行为。
        "reward_gate_enabled": False,
        "policy_sample_gate_enabled": False,

        "learning_rate": 1e-4,
        "gradient_penalty": 10.0,

        "buffer_capacity": 100_000,
        "updates_per_iteration": 4,
        "batch_size": 256,
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
            entropy_coef=0.001,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )
