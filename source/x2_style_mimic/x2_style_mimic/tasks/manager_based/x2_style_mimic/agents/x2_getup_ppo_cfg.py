"""X2 起身任务的 PPO 起始配置。"""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class X2GetUpPPORunnerCfg(PPORunnerCfg):

    experiment_name = "x2_getup_v0"