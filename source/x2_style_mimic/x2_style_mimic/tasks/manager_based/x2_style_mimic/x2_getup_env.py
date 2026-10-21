"""为 AMP 保存自动重置前的终止状态。"""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .mdp.amp import (
    AMP_KEY_BODY_NAMES,
    AMP_STATE_DIM,
    amp_observation,
)
from .x2_style_mimic_env_cfg import POLICY_JOINT_NAMES

class X2AMPGetUpEnv(ManagerBasedRLEnv):
    """保留普通环境返回值，额外返回 AMP 的真实步后状态。"""

    def step(self, action):
        if not hasattr(self, "_amp_asset_cfg"):
            self._amp_asset_cfg = SceneEntityCfg(
                "robot",
                joint_names=POLICY_JOINT_NAMES,
                body_names=list(AMP_KEY_BODY_NAMES),
                preserve_order=True,
            )
            self._amp_asset_cfg.resolve(self.scene)     # 把名称解析成当前场景对象中的实际索引。

            self._amp_terminal_state = torch.empty(
                self.num_envs,
                AMP_STATE_DIM,
                device=self.device,
            )

            self._amp_terminal_mask = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )

        self._amp_terminal_mask.zero_()                 # 每个环境步开始，清空本步捕获标志
        self._amp_capture_active = True

        try:
            result = super().step(action)
        finally:
            self._amp_capture_active = False            # 关闭捕获开关。这样如果父类执行中出错，也不会让环境一直残留

        observations, rewards, terminated, truncated, extras = result

        done = terminated | truncated           # 合并两种结束标志

        if not torch.equal(
            done,
            self._amp_terminal_mask,
        ):
            raise RuntimeError("AMP 终止状态捕获与环境重置不一致")

        # 普通环境使用步后观测。
        # 已终止的环境使用重置前保存的真实终止状态。
        following = observations["amp"].clone()
        following[done] = self._amp_terminal_state[done]

        extras["amp_next_state"] = following
        extras["amp_terminal_mask"] = (self._amp_terminal_mask.clone())

        return (
            observations,
            rewards,
            terminated,
            truncated,
            extras,
        )

    def _reset_idx(self, env_ids):
        # 只有动作执行过程中触发的自动重置才需要捕获。
        # 初始化和手动 reset 不属于动作转移。
        if getattr(self, "_amp_capture_active", False):
            state = amp_observation(
                self,
                self._amp_asset_cfg,
            )

            self._amp_terminal_state[env_ids] = state[env_ids]
            self._amp_terminal_mask[env_ids] = True

        # 捕获完成后，执行原有重置流程。
        super()._reset_idx(env_ids)