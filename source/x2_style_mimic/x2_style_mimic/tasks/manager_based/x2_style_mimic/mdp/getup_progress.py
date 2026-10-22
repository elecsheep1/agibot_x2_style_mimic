"""只记录起身成功，不因成功结束回合。"""

import torch

from .getup import GetUpSuccess

class GetUpProgress(GetUpSuccess):
    """起身进度判定器。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.succeeded = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.first_success_time = torch.full(
            (env.num_envs,),
            -1.0,
            device=self.device,
        )
        self.max_hold_seconds = torch.zeros(
            env.num_envs,
            device=self.device,
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids

        super().reset(env_ids)
        self.succeeded[ids] = False
        self.first_success_time[ids] = -1.0
        self.max_hold_seconds[ids] = 0.0

    def __call__(
        self,
        env,
        minimum_height,
        maximum_tilt,
        maximum_linear_speed,
        maximum_angular_speed,
        maximum_knee_angle,
        hold_time,
        force_threshold,
        asset_cfg,
        ground_sensor_cfgs,
        maximum_torso_tilt=0.55,
        torso_cfg=None,
    ):
        # 原计时器负责检查条件和连续保持时长。
        reached = super().__call__(
            env=env,
            minimum_height=minimum_height,
            maximum_tilt=maximum_tilt,
            maximum_linear_speed=maximum_linear_speed,
            maximum_angular_speed=maximum_angular_speed,
            maximum_knee_angle=maximum_knee_angle,
            hold_time=hold_time,
            force_threshold=force_threshold,
            asset_cfg=asset_cfg,
            ground_sensor_cfgs=ground_sensor_cfgs,
            maximum_torso_tilt=maximum_torso_tilt,
            torso_cfg=torso_cfg,
        )

        # 首次成功时记录时间；之后不重复计数。
        first_success = reached & ~self.succeeded
        elapsed = env.episode_length_buf * env.step_dt

        self.first_success_time[first_success] = (
            elapsed[first_success]
        )
        self.succeeded |= reached

        current_hold = self._hold_steps * env.step_dt
        self.max_hold_seconds.copy_(
            torch.maximum(
                self.max_hold_seconds,
                current_hold,
            )
        )

        # 保存本步快照，包括随后可能自动重置的环境。
        env._getup_progress_snapshot = {
            "succeeded": self.succeeded.clone(),
            "first_success_time": (
                self.first_success_time.clone()
            ),
            "hold_seconds": current_hold.clone(),
            "max_hold_seconds": (
                self.max_hold_seconds.clone()
            ),
        }

        assist = getattr(env, "_getup_assist", None)
        if assist is not None:
            assist.observe_stability(
                current_hold=current_hold,
                required_hold=hold_time,
            )
            env._getup_progress_snapshot.update({
                "post_assist_succeeded": (
                    assist.post_assist_succeeded.clone()
                ),
                "max_post_assist_hold": (
                    assist.max_post_assist_hold.clone()
                ),
                "assist_episode_force_n": (
                    assist.episode_force.clone()
                ),
                "assist_applied_force_n": (
                    assist.applied_force.clone()
                ),
                "assist_episode_stage": (
                    assist.episode_stage.clone()
                ),
            })


        # 关键：统计成功，但绝不因它触发终止。
        return torch.zeros_like(reached)
