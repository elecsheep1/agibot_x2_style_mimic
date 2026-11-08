from __future__ import annotations

import math

import torch
import warp as wp

from isaaclab.sensors import SensorBase


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
            or not isinstance(c.history_length, int)
            or isinstance(c.history_length, bool)
            or c.history_length < 1
            or c.update_period != 0.0
        ):
            raise ValueError("自接触参数无效，update_period 必须为 0")
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

            self._persistent = torch.zeros(
                shape,
                dtype=torch.bool,
                device=self.device,
            )
            self._max_duration = torch.zeros(
                self._num_envs, device=self.device
            )
            # 每个物理采样保存所有自身配对的最大力模长。
            # 先取最大值再阈值判定，等价于逐配对判定后 any。
            self.force_history = torch.zeros(
                (self._num_envs, c.history_length), device=self.device
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

        # duration 和最近物理采样历史跨控制周期保留。
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

        active = pair_force > self.cfg.force_threshold

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

        history = torch.cat(
            (pair_force.amax(dim=-1, keepdim=True), self.force_history[:, :-1]),
            dim=-1,
        )
        self.force_history.copy_(
            torch.where(mask[:, None], history, self.force_history)
        )
        self._updates.add_(mask.long())

    def snapshot(self, env_ids=None):
        from .rewards import self_contact_history_count

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
            # 与奖励共用公式；范围 [0, history_length]。
            "penalty_raw": self_contact_history_count(
                self.force_history, self.cfg.force_threshold
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
            self.force_history,
            self._updates,
        ):
            buffer[ids] = 0

    def _invalidate_initialize_callback(self, event):
        self._window_open = False
        self._view = None
        super()._invalidate_initialize_callback(event)

