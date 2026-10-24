# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-X2-Style-Mimic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x2_style_mimic_env_cfg:X2StyleMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="X2-Prone-MultiMotion-v0",
    entry_point=f"{__name__}.x2_getup_env:X2MultiMotionGetUpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x2_getup_env_cfg:X2MultiMotionGetUpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.x2_amp_cfg:X2MultiMotionAMPRunnerCfg",
        "amp_cfg_entry_point": f"{agents.__name__}.x2_amp_cfg:X2MultiMotionAMPRunnerCfg",
    },
)