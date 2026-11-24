from __future__ import annotations

import math

import torch
import warp as wp

from isaaclab.sensors import SensorBase
from .rewards import (
    self_contact_pressure_cost,
    self_contact_control_cost,
)

GROUND_CONTACT_SENSOR_NAMES = (
    "head_yaw_ground_contact",
    "head_pitch_ground_contact",
    "left_foot_ground_contact",
    "right_foot_ground_contact",
    "left_hand_ground_contact",
    "right_hand_ground_contact",
)


class SelfContactSensor(SensorBase):
    """同环境机器人自接触监测，不包含地面"""

    def __init__(self, cfg):
        self._window_open = False
        self._view = None
        super().__init__(cfg)

    def _initialize_impl(self):
        super()._initialize_impl()

        from pxr import Usd, UsdPhysics
        from isaaclab_physx.physics import PhysxManager

        c = self.cfg
        if (
            not math.isfinite(c.force_threshold)
            or not math.isfinite(c.duration_start)
            or c.force_threshold < 0.0
            or c.duration_start < 0.0
            or c.update_period != 0.0
        ):
            raise ValueError("自接触参数无效，update_period 必须为 0")

        pressure_values = (
            c.pressure_force_threshold,
            c.pressure_release_threshold,
            c.pressure_release_rate,
            c.pressure_force_scale,
            c.pressure_grace_seconds,
            c.pressure_ramp_seconds,
            c.impact_force_threshold,
            c.impact_force_scale,
            c.impact_weight,
        )
        if (
            not all(math.isfinite(value) for value in pressure_values)
            or not (
                0.0 <= c.pressure_release_threshold
                < c.pressure_force_threshold
            )
            or c.pressure_release_rate <= 0.0
            or c.pressure_force_scale <= 0.0
            or c.pressure_grace_seconds < 0.0
            or c.pressure_ramp_seconds <= 0.0
            or c.impact_force_threshold < c.pressure_force_threshold
            or c.impact_force_scale <= 0.0
            or c.impact_weight < 0.0
        ):
            raise ValueError("自接触奖励参数无效")
        # 从实际 stage 中解析 env_0 的刚体路径。
        root_path = c.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        root_path = root_path.replace("env_.*", "env_0")
        root = self.stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise ValueError(f"无法找到机器人：{root_path}")

        paths = sorted(
            str(prim.GetPath())
            for prim in Usd.PrimRange(
                root, Usd.TraverseInstanceProxies()
            )
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )

        if len(paths) < 2 or "/env_0/" not in root_path:
            raise ValueError("未找到有效的多刚体机器人")

        self.body_names = [path.rsplit("/", 1)[-1] for path in paths]
        if len(set(self.body_names)) != len(paths):
            raise ValueError("刚体名称重复，无法建立明确配对")

        self._body_count = len(paths)

        sources = []
        filters = []

        for env_id in range(self._num_envs):
            current_paths = [
                path.replace(
                    "/env_0/", f"/env_{env_id}/", 1
                )
                for path in paths
            ]

            for path in current_paths:
                prim = self.stage.GetPrimAtPath(path)
                if (
                    not prim.IsValid()
                    or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
                ):
                    raise ValueError(f"刚体路径无效：{path}")

            for body_id, path in enumerate(current_paths):
                sources.append(path)

                # 只包含同环境的其他机器人刚体。
                # 不包含自身、地面和其他环境。
                filters.append([
                    other_path
                    for other_id, other_path
                    in enumerate(current_paths)
                    if other_id != body_id
                ])

        capacity = max(
            c.contact_capacity_min,
            self._num_envs
            * self._body_count
            * c.contacts_per_sensor,
        )

        self._view = (
            PhysxManager.get_physics_sim_view()
            .create_rigid_contact_view(
                sources,
                filter_patterns=filters,
                max_contact_data_count=capacity,
            )
        )

        if (
            self._view.sensor_count
            != self._num_envs * self._body_count
            or self._view.filter_count
            != self._body_count - 1
        ):
            raise RuntimeError("自接触矩阵维度与路径映射不一致")

        # 每个无序身体配对只保留一次：i < j。
        with torch.inference_mode(False):
            self._i, self._j = torch.triu_indices(
                self._body_count,
                self._body_count,
                offset=1,
                device=self.device,
            )

            shape = (self._num_envs, self._i.numel())

            self.duration = torch.zeros(
                shape, device=self.device
            )
            self.peak_force = torch.zeros_like(self.duration)
            self.strong_duration = torch.zeros_like(self.duration)
            # strong_duration 表示可缓慢释放的接触记忆，
            # 不再表示严格连续的强接触时长。
            self._strong_active = torch.zeros_like(
                self.duration, dtype=torch.bool
            )
            self.pressure_cost_sum = torch.zeros(
                self._num_envs, device=self.device
            )

            self._persistent = torch.zeros(
                shape,
                dtype=torch.bool,
                device=self.device,
            )
            self._max_duration = torch.zeros(
                self._num_envs, device=self.device
            )
            self._updates = torch.zeros(
                self._num_envs,
                dtype=torch.long,
                device=self.device,
            )

        self._window_open = False

    @property
    def data(self):
        # 纯读取，不额外制造一次物理采样。
        return self.snapshot()

    def begin_control_window(self):
        if not self.is_initialized or self._view is None:
            raise RuntimeError("自接触传感器尚未初始化")

        if self._window_open:
            raise RuntimeError("上一个自接触控制周期尚未结束")

        self.peak_force.zero_()
        self._persistent.zero_()
        self._max_duration.zero_()
        self._updates.zero_()
        self.pressure_cost_sum.zero_()

        # 两种接触计时均跨控制周期保留。
        self._window_open = True

    def end_control_window(self):
        self._window_open = False

    def update(self, dt, force_recompute=False):
        if not self.is_initialized:
            return

        # 禁止把一次控制周期末采样当作多个物理子步。
        if self._window_open and not math.isclose(
            float(dt),
            self._sim_physics_dt,
            rel_tol=1e-5,
            abs_tol=1e-8,
        ):
            raise RuntimeError(
                "自接触监测要求按 physics_dt 更新；"
                "当前步进方式隐藏了物理子步"
            )

        # 只强制更新本传感器，不改变其他传感器的懒更新。
        super().update(dt, force_recompute=True)

    def _update_buffers_impl(self, env_mask):
        if not self._window_open:
            return

        force = wp.to_torch(
            self._view.get_contact_force_matrix(
                self._sim_physics_dt
            )
        ).reshape(
            self._num_envs,
            self._body_count,
            self._body_count - 1,
            3,
        )

        # 每个源的过滤列排除了源自身。
        # 对 i < j：
        #   i -> j 对应第 j-1 列；
        #   j -> i 对应第 i 列。
        # 合并双向数据，避免 A-B/B-A 重复处罚。
        pair_force = torch.maximum(
            force[:, self._i, self._j - 1].norm(dim=-1),
            force[:, self._j, self._i].norm(dim=-1),
        )

        if not bool(torch.isfinite(pair_force).all()):
            raise RuntimeError("自接触力出现非有限值")

        if self.cfg.validate_contact_buffer:
            _, _, _, separations, counts, _ = (
                self._view.get_contact_data(
                    self._sim_physics_dt
                )
            )
            used = int(wp.to_torch(counts).sum())
            available = len(
                wp.to_torch(separations).flatten()
            )
            if used >= available:
                raise RuntimeError(
                    "自接触缓冲达到容量，请增大配置容量"
                )

        mask = wp.to_torch(env_mask).bool()
        pair_mask = mask[:, None]
        c = self.cfg

        # 滞回：超过上阈值开启，低于下阈值关闭，
        # 中间区域保留上一物理子步的状态。
        next_active = torch.where(
            pair_force > c.pressure_force_threshold,
            torch.ones_like(self._strong_active),
            torch.where(
                pair_force < c.pressure_release_threshold,
                torch.zeros_like(self._strong_active),
                self._strong_active,
            ),
        )
        self._strong_active.copy_(
            torch.where(
                pair_mask, next_active, self._strong_active
            )
        )

        # 接触记忆：开启时累积，关闭后缓慢减少。
        memory_cap = (
            c.pressure_grace_seconds
            + c.pressure_ramp_seconds
        )
        next_memory = torch.where(
            self._strong_active,
            self.strong_duration + self._sim_physics_dt,
            (
                self.strong_duration
                - c.pressure_release_rate * self._sim_physics_dt
            ).clamp_min(0.0),
        ).clamp_max(memory_cap)

        self.strong_duration.copy_(
            torch.where(
                pair_mask, next_memory, self.strong_duration
            )
        )

        cost = self_contact_pressure_cost(
            pair_force=pair_force,
            strong_duration=self.strong_duration,
            force_threshold=c.pressure_force_threshold,
            force_scale=c.pressure_force_scale,
            grace_seconds=c.pressure_grace_seconds,
            ramp_seconds=c.pressure_ramp_seconds,
        )
        self.pressure_cost_sum.add_(
            torch.where(mask, cost, torch.zeros_like(cost))
        )

        # 下方原有的低阈值诊断仍然按 10 N 统计，
        # 与上面的奖励计时分开。
        active = pair_force > c.force_threshold

        duration = torch.where(
            active,
            self.duration + self._sim_physics_dt,
            torch.zeros_like(self.duration),
        )
        self.duration.copy_(
            torch.where(
                mask[:, None], duration, self.duration
            )
        )

        self.peak_force.copy_(
            torch.where(
                mask[:, None],
                torch.maximum(self.peak_force, pair_force),
                self.peak_force,
            )
        )

        persistent = (
            active
            & (duration > self.cfg.duration_start)
        )
        self._persistent |= persistent & mask[:, None]

        self._max_duration.copy_(
            torch.where(
                mask,
                torch.maximum(
                    self._max_duration,
                    duration.amax(dim=-1),
                ),
                self._max_duration,
            )
        )

        self._updates.add_(mask.long())

    @torch.no_grad()
    def export_task_memory(self, env_ids=None):
        """读取逐配对接触记忆，不更新采样或计时。"""
        if not self.is_initialized:
            raise RuntimeError("自接触传感器尚未初始化，无法读取任务记忆")

        ids = (
            slice(None)
            if env_ids is None
            else env_ids
        )

        return {
            "self_contact_strong_duration": (
                self.strong_duration[ids]
                .detach()
                .clone()
            ),
            "self_contact_strong_active": (
                self._strong_active[ids]
                .detach()
                .clone()
            ),
        }


    def snapshot(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids

        metrics = {
            "pair_count": (
                self.peak_force > self.cfg.force_threshold
            ).sum(dim=-1).float(),
            "peak_pair_force_n": (
                self.peak_force.amax(dim=-1)
            ),
            "persistent_pair_count": (
                self._persistent.sum(dim=-1).float()
            ),
            "max_duration_s": self._max_duration,
            # 与当前实际奖励一致。
            "penalty_raw": self_contact_control_cost(self),
            "strong_contact_memory_s": (
                self.strong_duration.amax(dim=-1)
            ),
            "updates_per_control_step": (
                self._updates.float()
            ),
        }

        return {
            name: value[ids].clone()
            for name, value in metrics.items()
        }

    def reset(self, env_ids=None, env_mask=None):
        if not self.is_initialized:
            return

        mask = self._resolve_indices_and_mask(
            env_ids, env_mask
        )
        super().reset(env_mask=mask)

        ids = wp.to_torch(mask).bool()

        for buffer in (
            self.duration,
            self.peak_force,
            self._persistent,
            self._max_duration,
            self._updates,
            self.strong_duration,
            self._strong_active,
            self.pressure_cost_sum,
        ):
            buffer[ids] = 0

    def _invalidate_initialize_callback(self, event):
        self._window_open = False
        self._view = None
        super()._invalidate_initialize_callback(event)


@torch.no_grad()
def reset_ground_contact_sensors(
    env,
    env_ids,
):
    """重置指定环境的地面接触数据及过滤接触历史。"""
    env_ids = torch.as_tensor(
        env_ids,
        dtype=torch.long,
        device=env.device,
    )

    if env_ids.numel() == 0:
        return

    for name in GROUND_CONTACT_SENSOR_NAMES:
        sensor = env.scene.sensors[name]

        # 先执行传感器正常的重置流程。
        sensor.reset(env_ids)

        # 读取 data 可能触发懒更新。
        history = (
            sensor.data.force_matrix_w_history
        )

        if history is None:
            raise RuntimeError(
                f"{name} 缺少过滤地面接触力历史"
            )

        forces = (
            history.torch
            if hasattr(history, "torch")
            else history
        )

        if (
            forces.ndim != 5
            or forces.shape[0] != env.num_envs
            or forces.shape[2:] != (1, 1, 3)
        ):
            raise RuntimeError(
                f"{name} 的过滤接触历史形状不正确："
                f"{tuple(forces.shape)}"
            )

        # 当前 PhysX 后端 reset 遗漏了这个缓冲。
        # 使用索引赋值，确保修改原始缓冲。
        forces[env_ids] = 0.0

        # 上面读取 data 可能重新填入旧物理帧的当前力。
        # 再重置当前缓冲和时间戳；过滤历史保持为零。
        sensor.reset(env_ids)