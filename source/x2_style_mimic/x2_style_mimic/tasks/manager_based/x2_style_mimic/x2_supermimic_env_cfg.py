"""SuperMimic 复用原 PPO 起身环境配置。"""

from isaaclab.utils.configclass import configclass

from .x2_getup_env_cfg import (
    X2MultiMotionGetUpEnvCfg,
)


@configclass
class X2SuperMimicEnvCfg(
    X2MultiMotionGetUpEnvCfg
):
    """继承原起身任务，不另行覆盖任务设置。

    保留原配置的：
    - 机器人与接触传感器
    - 动作处理
    - 全部真实任务奖励
    - 出生分布和状态库
    - 物理随机化与站立推扰
    - 观测噪声
    - 辅助课程
    - 终止条件与成功判据

    环境数量、设备和随机种子由训练入口设置。
    """

    pass