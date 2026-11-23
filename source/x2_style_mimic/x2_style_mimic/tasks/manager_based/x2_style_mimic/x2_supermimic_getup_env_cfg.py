"""SuperMimic 倒地起身任务配置。"""

from dataclasses import asdict

from isaaclab.utils.configclass import configclass

from x2_style_mimic.algorithms.supermimic.heads import (
    GetupTaskCfg,
)

from .x2_supermimic_env_cfg import (
    X2SuperMimicEnvCfg,
)


@configclass
class X2SuperMimicGetupEnvCfg(X2SuperMimicEnvCfg):
    """真实环境继承原起身任务；这里声明想象任务参数。"""

    getup_task: dict = asdict(
        GetupTaskCfg(
            # 控制周期。
            step_dt=0.02,

            # 正向奖励权重。
            standing_weight=10.0,
            style_weight=10.0,

            # 惩罚系数使用正数，奖励函数内部取负。
            action_rate_weight=0.005,
            joint_limit_weight=0.5,
            joint_acc_weight=1e-7,
            boundary_weight=5.0,

            # 关节软限位比例。
            soft_limit_fraction=0.9,

            # 越界距离与可微过渡宽度，单位米。
            max_distance=2.0,
            boundary_transition=0.1,

            contact_distance=0.005,
            contact_transition=0.005,
            head_contact_weight=2.0,
            head_contact_distance=0.0,
            head_contact_transition=0.005,
        )
    )