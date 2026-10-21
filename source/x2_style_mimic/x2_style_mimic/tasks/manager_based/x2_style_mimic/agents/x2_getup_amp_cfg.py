"""X2 起身任务的 AMP＋PPO 起始配置。"""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from .x2_getup_ppo_cfg import X2GetUpPPORunnerCfg
from ..x2_getup_env_cfg import REFERENCE_FILE
from ..x2_style_mimic_env_cfg import POLICY_JOINT_NAMES


@configclass
class X2AMPAlgorithmCfg(RslRlPpoAlgorithmCfg):

    class_name = (
        "x2_style_mimic.algorithms.amp.ppo:AMPPPO"
    )

    amp_cfg: dict = {
        "motion_file": str(REFERENCE_FILE),
        "joint_names": list(POLICY_JOINT_NAMES),

        # 50 Hz 控制，与参考数据一致。
        "step_dt": 0.02,

        # 实际加入的风格奖励为：
        # weight × step_dt × raw_style_reward。
        "style_reward_weight": 5.0,

        # 以下学习率专用于判别器。
        "learning_rate": 1e-4,
        "gradient_penalty": 10.0,

        "buffer_capacity": 100_000,
        "updates_per_iteration": 4,     # 每轮更新判别器多少次
        "batch_size": 256,              # 每次判别器更新抽多少样本
    }


@configclass
class X2GetUpAMPRunnerCfg(X2GetUpPPORunnerCfg):

    experiment_name = "x2_getup_amp_v2"

    algorithm: X2AMPAlgorithmCfg = X2AMPAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,

        # PPO 的学习率，与判别器学习率分开。
        learning_rate=3.0e-4,
        schedule="adaptive",

        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
