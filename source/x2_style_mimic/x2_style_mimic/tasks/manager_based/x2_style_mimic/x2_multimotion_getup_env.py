"""新版多动作环境：保留 AMP 终止状态，并输出独立成功统计。"""

import torch

from .x2_getup_env import X2AMPGetUpEnv
from .mdp.getup_assist import GetUpAssist

class X2MultiMotionGetUpEnv(X2AMPGetUpEnv):
    def step(self, action):
        # 必须在 step 前保存：自动重置会换成下一回合的动作编号。
        motion_ids = (
            self._getup_reference_motion_ids.clone()
        )

        self._getup_assist.apply()
        result = super().step(action)
        observations, rewards, terminated, truncated, extras = result

        # 计时器在终止判断阶段保存了重置前的快照。
        snapshot = self._getup_progress_snapshot
        extras["getup_progress"] = {
            key: value.clone()
            for key, value in snapshot.items()
        }
        extras["getup_progress"]["motion_ids"] = motion_ids

        return (
            observations,
            rewards,
            terminated,
            truncated,
            extras,
        )

    def _reset_idx(self, env_ids):
        episode_stats = None

        # 初始化或手动 reset 不算已完成的训练回合。
        # 只在真实 step 引起自动重置时统计。
        if (
            getattr(self, "_amp_capture_active", False)
            and len(env_ids) > 0
            and hasattr(self, "_getup_progress_snapshot")
        ):
            snapshot = self._getup_progress_snapshot

            episode_stats = {
                key: value[env_ids].clone()
                for key, value in snapshot.items()
            }
            episode_stats["motion_ids"] = (
                self._getup_reference_motion_ids[
                    env_ids
                ].clone()
            )

        if episode_stats is not None:
            self._getup_assist.finish_episodes(
                env_ids,
                episode_stats["motion_ids"],
            )

        # 父类先捕获 AMP 终止状态，再执行实际重置。
        super()._reset_idx(env_ids)

        # 初始化时也会执行 reset，因此在这里延迟创建。
        if not hasattr(self, "_getup_assist"):
            with torch.inference_mode(False):
                self._getup_assist = GetUpAssist(self)

        self._getup_assist.reset(env_ids)

        if episode_stats is None:
            return

        # 父类重置会重建 extras["log"]，所以在它之后写入。
        log = self.extras.setdefault("log", {})
        log["Assist/stage"] = float(
            self._getup_assist.stage
        )
        log["Assist/force_cap_n"] = float(
            self._getup_assist.levels[
                self._getup_assist.stage
            ]
        )

        if "post_assist_succeeded" in episode_stats:
            post_success = episode_stats[
                "post_assist_succeeded"
            ]
            log["Assist/post_withdraw_success_rate"] = (
                post_success.float().mean().item()
            )
            log["Assist/max_post_withdraw_hold"] = (
                episode_stats[
                    "max_post_assist_hold"
                ].mean().item()
            )

            zero = (
                episode_stats["assist_episode_force_n"] == 0
            )
            if bool(zero.any()):
                log["Assist/zero_force_success_rate"] = (
                    post_success[zero].float().mean().item()
                )
        succeeded = episode_stats["succeeded"]
        first_time = episode_stats["first_success_time"]
        max_hold = episode_stats["max_hold_seconds"]
        final_hold = episode_stats["hold_seconds"]
        motion_ids = episode_stats["motion_ids"]

        log["GetUp/success_rate"] = (
            succeeded.float().mean().item()
        )
        log["GetUp/max_hold_seconds"] = (
            max_hold.mean().item()
        )
        log["GetUp/final_hold_seconds"] = (
            final_hold.mean().item()
        )

        # 只对成功回合计算首次成功时间。
        # 本批没有成功回合时不写该指标。
        if bool(succeeded.any()):
            log["GetUp/first_success_seconds"] = (
                first_time[succeeded].mean().item()
            )

        # 按参考动作分别统计，避免只学会其中某一条。
        for motion_id in range(
            len(self._getup_reference_motion_paths)
        ):
            selected = motion_ids == motion_id

            if not bool(selected.any()):
                continue

            prefix = f"GetUp/motion_{motion_id}"
            log[f"{prefix}/success_rate"] = (
                succeeded[selected].float().mean().item()
            )
            log[f"{prefix}/max_hold_seconds"] = (
                max_hold[selected].mean().item()
            )