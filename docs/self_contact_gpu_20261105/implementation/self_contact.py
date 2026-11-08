"""PhysX-only self-contact monitor. No reward is enabled by this module."""
from __future__ import annotations

import math
import torch
import warp as wp
from pxr import Usd, UsdPhysics
from isaaclab.sensors import SensorBase, SensorBaseCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxManager


class SelfContactSensor(SensorBase):
    def __init__(self, cfg):
        self._window_open = False
        self._view = None
        super().__init__(cfg)

    def _initialize_impl(self):
        super()._initialize_impl()
        c = self.cfg
        values = [c.force_threshold, c.force_scale, c.duration_start,
                  c.duration_full, c.penalty_cap]
        if (not all(math.isfinite(x) for x in values)
                or c.force_threshold < 0 or c.force_scale <= 0
                or not 0 <= c.duration_start < c.duration_full
                or c.penalty_cap <= 0 or c.update_period != 0):
            raise ValueError('Invalid self-contact settings; update_period must be zero')
        root_path = c.prim_path.replace('{ENV_REGEX_NS}', '/World/envs/env_0')
        root_path = root_path.replace('env_.*', 'env_0')
        root = self.stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise ValueError(f'Cannot resolve env0 robot: {root_path}')
        paths = sorted(str(p.GetPath()) for p in
                       Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                       if p.HasAPI(UsdPhysics.RigidBodyAPI))
        if len(paths) < 2 or '/env_0/' not in root_path:
            raise ValueError('Expected multiple rigid bodies under env_0')
        self.body_names = [p.rsplit('/', 1)[-1] for p in paths]
        if len(set(self.body_names)) != len(paths):
            raise ValueError('Rigid body names must be unique')
        self._b = len(paths)
        sources, filters = [], []
        for e in range(self._num_envs):
            current = [p.replace('/env_0/', f'/env_{e}/', 1) for p in paths]
            for p in current:
                prim = self.stage.GetPrimAtPath(p)
                if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError(f'Invalid rigid body path: {p}')
            for i, p in enumerate(current):
                sources.append(p)
                filters.append([q for j, q in enumerate(current) if j != i])
        self._capacity = max(c.contact_capacity_min,
                             self._num_envs * self._b * c.contacts_per_sensor)
        self._view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
            sources, filter_patterns=filters,
            max_contact_data_count=self._capacity,
        )
        if (self._view.sensor_count != self._num_envs * self._b
                or self._view.filter_count != self._b - 1):
            raise RuntimeError('Unexpected self-contact matrix dimensions')
        with torch.inference_mode(False):
            self._i, self._j = torch.triu_indices(
                self._b, self._b, offset=1, device=self.device)
            shape = (self._num_envs, self._i.numel())
            self.duration = torch.zeros(shape, device=self.device)
            self.peak_force = torch.zeros_like(self.duration)
            self._persistent = torch.zeros(shape, dtype=torch.bool, device=self.device)
            self._max_duration = torch.zeros(self._num_envs, device=self.device)
            self._penalty_sum = torch.zeros_like(self._max_duration)
            self._updates = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
        self._window_open = False

    @property
    def data(self):
        # Pure read: accessing data must not create another physics sample.
        return self.snapshot()

    def begin_control_window(self):
        if not self.is_initialized or self._view is None:
            raise RuntimeError('Self-contact sensor is not initialized')
        if self._window_open:
            raise RuntimeError('Previous self-contact control window is still open')
        self.peak_force.zero_()
        self._persistent.zero_()
        self._max_duration.zero_()
        self._penalty_sum.zero_()
        self._updates.zero_()
        # duration intentionally persists across control windows.
        self._window_open = True

    def end_control_window(self):
        self._window_open = False

    def update(self, dt, force_recompute=False):
        if not self.is_initialized:
            return
        if self._window_open and not math.isclose(
                float(dt), self._sim_physics_dt, rel_tol=1e-5, abs_tol=1e-8):
            raise RuntimeError('Expected scene update at physics_dt; decimation path hides substeps')
        # Force only this sensor, leaving other lazy sensors unchanged.
        super().update(dt, force_recompute=True)

    def _update_buffers_impl(self, env_mask):
        if not self._window_open:
            return
        force = wp.to_torch(self._view.get_contact_force_matrix(
            self._sim_physics_dt)).reshape(self._num_envs, self._b, self._b - 1, 3)
        # i<j: column j-1 in source i; column i in source j.
        # Merge both directions rather than counting them twice.
        pair_force = torch.maximum(
            force[:, self._i, self._j - 1].norm(dim=-1),
            force[:, self._j, self._i].norm(dim=-1),
        )
        if not bool(torch.isfinite(pair_force).all()):
            raise RuntimeError('Nonfinite self-contact force')
        if self.cfg.validate_contact_buffer:
            _, _, _, separations, counts, _ = self._view.get_contact_data(self._sim_physics_dt)
            if int(wp.to_torch(counts).sum()) >= len(wp.to_torch(separations).flatten()):
                raise RuntimeError('Self-contact buffer reached capacity; increase capacity')
        mask = wp.to_torch(env_mask).bool()
        active = pair_force > self.cfg.force_threshold
        duration = torch.where(active, self.duration + self._sim_physics_dt, 0.)
        self.duration.copy_(torch.where(mask[:, None], duration, self.duration))
        self.peak_force.copy_(torch.where(mask[:, None],
                             torch.maximum(self.peak_force, pair_force), self.peak_force))
        persistent = active & (duration > self.cfg.duration_start)
        self._persistent |= persistent & mask[:, None]
        self._max_duration.copy_(torch.where(mask,
            torch.maximum(self._max_duration, duration.amax(-1)), self._max_duration))
        strength = ((pair_force - self.cfg.force_threshold) / self.cfg.force_scale).clamp(0., 1.)
        blend = ((duration - self.cfg.duration_start)
                 / (self.cfg.duration_full - self.cfg.duration_start)).clamp(0., 1.)
        gate = blend.square() * (3. - 2. * blend)
        raw = (strength * gate).sum(-1).clamp(max=self.cfg.penalty_cap)
        self._penalty_sum.add_(torch.where(mask, raw, 0.))
        self._updates.add_(mask.long())

    def snapshot(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        metrics = dict(
            pair_count=(self.peak_force > self.cfg.force_threshold).sum(-1).float(),
            peak_pair_force_n=self.peak_force.amax(-1),
            persistent_pair_count=self._persistent.sum(-1).float(),
            max_duration_s=self._max_duration,
            penalty_raw=self._penalty_sum / self._updates.clamp_min(1),
            updates_per_control_step=self._updates.float(),
        )
        return {name: value[ids].clone() for name, value in metrics.items()}

    def reset(self, env_ids=None, env_mask=None):
        if not self.is_initialized:
            return
        mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(env_mask=mask)
        ids = wp.to_torch(mask).bool()
        for buffer in (self.duration, self.peak_force, self._persistent,
                       self._max_duration, self._penalty_sum, self._updates):
            buffer[ids] = 0

    def _invalidate_initialize_callback(self, event):
        self._window_open = False
        self._view = None
        super()._invalidate_initialize_callback(event)


@configclass
class SelfContactSensorCfg(SensorBaseCfg):
    class_type: type = SelfContactSensor
    force_threshold: float = 50.0
    force_scale: float = 150.0
    duration_start: float = 0.1
    duration_full: float = 0.3
    penalty_cap: float = 3.0
    contact_capacity_min: int = 262144
    contacts_per_sensor: int = 8
    validate_contact_buffer: bool = True
