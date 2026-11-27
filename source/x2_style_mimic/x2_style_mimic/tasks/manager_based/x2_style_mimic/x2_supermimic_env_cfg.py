"""SuperMimic 复用原 PPO 起身环境配置。"""

from isaaclab.utils.configclass import configclass

from .x2_getup_env_cfg import (
    X2MultiMotionGetUpEnvCfg,
)


@configclass
class X2SuperMimicEnvCfg(
    X2MultiMotionGetUpEnvCfg
):
    """无辅助力、无外部速度推扰的 SuperMimic 起身环境。"""

    def __post_init__(self):
        super().__post_init__()

        # 不施加训练中的随机速度跳变。
        self.events.standing_perturbation = None

        # 想象任务未使用一次性成功奖励，真实环境也关闭该项。
        self.rewards.success_bonus = None

        # 不使用自碰撞奖励和专用自接触传感器。
        # 机器人资产中的物理自碰撞继续保留。
        self.rewards.self_contact = None
        self.scene.self_contact = None

        # 覆盖继承配置，避免辅助课程被其他入口意外启用。
        self.assist_force_levels = (0.0,)
        self.assist_initial_stage = 0
        self.assist_zero_probability = 1.0
        self.assist_curriculum_enabled = False
