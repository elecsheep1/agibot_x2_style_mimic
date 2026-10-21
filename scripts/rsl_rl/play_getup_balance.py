# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import warnings

warnings.warn(
    "scripts/reinforcement_learning/rsl_rl/play.py is deprecated. Use "
    "`./isaaclab.sh play --rl_library rsl_rl --task <TASK>` instead. "
    "Example: `./isaaclab.sh play --rl_library rsl_rl --task Isaac-Cartpole-v0`.",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
import contextlib
import importlib.metadata as metadata
import json
import math
import os
import sys
import time
import unicodedata

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

import x2_style_mimic.tasks  # noqa: F401
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# 回放诊断默认关闭；指定 episode 数时自动启用，并在完成后退出。
parser.add_argument("--diagnostics", action="store_true", help="Print cumulative X2 locomotion diagnostics.")
parser.add_argument("--getup-diagnostics", action="store_true", help="Diagnose get-up success gates and ground-filtered head contacts.")
parser.add_argument("--getup-control-trace", action="store_true", help="Print hip, knee and waist commands during env 0's first 2 seconds; requires get-up diagnostics.")
parser.add_argument("--getup-waist-pitch-target", type=float, default=None, help="Diagnostic intervention: replace only the waist pitch position target in radians; requires get-up diagnostics.")
parser.add_argument("--diagnostic-report", type=str, default=None, help="Optional JSON output path for get-up diagnostics.")
parser.add_argument("--eval-episodes", type=int, default=0, help="Stop after N completed episodes; 0 disables the limit.")
parser.add_argument("--eval-episodes-per-env", type=int, default=0, help="Get-up diagnostics: finish K episodes in every environment; overrides --eval-episodes.")
parser.add_argument("--diagnostic-interval", type=float, default=5.0, help="Diagnostic print interval in simulation seconds.")
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
if args_cli.eval_episodes < 0 or args_cli.diagnostic_interval <= 0:
    parser.error("--eval-episodes must be nonnegative and --diagnostic-interval must be positive")
if args_cli.eval_episodes_per_env < 0 or (args_cli.eval_episodes_per_env > 0 and not args_cli.getup_diagnostics):
    parser.error("--eval-episodes-per-env must be nonnegative and requires --getup-diagnostics")
if args_cli.getup_control_trace and not args_cli.getup_diagnostics:
    parser.error("--getup-control-trace requires --getup-diagnostics")
if args_cli.getup_waist_pitch_target is not None:
    if not args_cli.getup_diagnostics or not math.isfinite(args_cli.getup_waist_pitch_target):
        parser.error("--getup-waist-pitch-target requires --getup-diagnostics and a finite target")

if args_cli.video:
    args_cli.enable_cameras = True


# Call an external callback if requested. This gives opportunity to external code to register the environments
# The function is expected to return a list of arguments that were not consumed by the callback.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
# The remaining arguments are the arguments that were not consumed by both this scripts
# argparser and (optionally) the external callback function. Both sides of this
# intersection are pre-fold (the callback reads the user's original sys.argv), so
# preset tokens like ``physics=NAME`` compare correctly here. Fold runs after.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")


class LocomotionDiagnostics:
    """参照 Microduck：按名称解析左右脚、累计统计、排除 reset 后状态。"""

    def __init__(self, env):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_walk_params import (
            COMMAND_NAME,
            FOOT_BODY_NAMES,
            FOOT_CONTACT_FORCE_THRESHOLD,
        )

        self.env = env
        self.robot = env.scene["robot"]
        self.sensor = env.scene.sensors["contact_forces"]
        self.command_name = COMMAND_NAME
        self.threshold = FOOT_CONTACT_FORCE_THRESHOLD
        self.foot_ids, _ = self.robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)
        self.sensor_ids, _ = self.sensor.find_sensors(list(FOOT_BODY_NAMES), preserve_order=True)
        self.joint_ids, self.joint_names = self.robot.find_joints(
            ["left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint"],
            preserve_order=True,
        )
        device = env.device
        n = env.num_envs
        self.steps = 0
        self.completed = 0
        self.timeouts = 0
        self.samples = torch.zeros(2, device=device)
        self.tracking_sum = torch.zeros(5, device=device)
        self.contact_sum = torch.zeros(2, device=device)
        self.slide_speed_sum = torch.zeros(2, device=device)
        self.slide_count = torch.zeros(2, device=device)
        self.support_sum = torch.zeros(3, device=device)
        self.prev_contact = torch.zeros((n, 2), dtype=torch.bool, device=device)
        self.prev_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.swing_active = torch.zeros((n, 2), dtype=torch.bool, device=device)
        self.swing_duration = torch.zeros((n, 2), device=device)
        self.swing_sum = torch.zeros(2, device=device)
        self.swing_count = torch.zeros(2, device=device)
        self.joint_min = torch.full((4,), float("inf"), device=device)
        self.joint_max = torch.full((4,), -float("inf"), device=device)
        rigid_paths = list(self.sensor.body_physx_view.prim_paths)
        contact_paths = list(self.sensor.contact_view.sensor_paths)
        print(f"[DEBUG] Robot foot indices: {self.foot_ids}; sensor indices: {self.sensor_ids}")
        print(f"[DEBUG] Contact/body paths identical: {contact_paths == rigid_paths}")
        if contact_paths != rigid_paths:
            raise RuntimeError("Contact/body view paths differ; cannot trust foot diagnostics.")
        print("[DEBUG] Moving samples: command XY norm > 0.1 m/s; slip threshold: 0.02 m/s")

    def update(self, command, dones):
        self.steps += 1
        done = dones.bool().reshape(-1)
        self.completed += int(done.sum().item())
        self.timeouts += int((done & self.env.termination_manager.time_outs).sum().item())
        # env.step 已自动重置 done 环境；同时排除本步发生指令重采样的状态。
        current_command = self.env.command_manager.get_command(self.command_name)
        valid = ~done & torch.isclose(command, current_command, atol=1e-6, rtol=0).all(dim=-1)
        moving = valid & (command[:, :2].norm(dim=-1) > 0.1)
        self.samples += torch.stack((valid.sum(), moving.sum()))
        velocity = self.robot.data.root_lin_vel_b.torch
        yaw_rate = self.robot.data.root_ang_vel_b.torch[:, 2]
        values = torch.stack((command[:, 0], velocity[:, 0], velocity[:, 1],
                              (velocity[:, :2] - command[:, :2]).norm(dim=-1),
                              (yaw_rate - command[:, 2]).abs()), dim=-1)
        self.tracking_sum += torch.where(moving[:, None], values, 0.0).sum(dim=0)

        # 支撑/步态转换使用当前力，避免 3 帧历史延长接触而扭曲腾空时间。
        # 脚部速度是刚体原点速度代理，不能当作精确接触点滑移速度。
        force = self.sensor.data.net_forces_w.torch[:, self.sensor_ids]
        contact = force.norm(dim=-1) > self.threshold
        foot_speed = self.robot.data.body_lin_vel_w.torch[:, self.foot_ids, :2].norm(dim=-1)
        moving_contact = contact & moving[:, None]
        self.contact_sum += moving_contact.sum(dim=0)
        self.slide_speed_sum += torch.where(moving_contact, foot_speed, 0.0).sum(dim=0)
        self.slide_count += (moving_contact & (foot_speed > 0.02)).sum(dim=0)
        support_count = contact.sum(dim=-1)
        self.support_sum += torch.stack([
            (moving & (support_count == count)).sum() for count in (1, 2, 0)
        ])

        # 只接受观测到起飞和落地、全程为移动指令的完整摆动；不跨 reset 拼接。
        eligible = moving & self.prev_valid
        self.swing_active &= eligible[:, None]
        liftoff = self.prev_contact & ~contact & eligible[:, None]
        self.swing_active |= liftoff
        self.swing_duration = torch.where(liftoff, 0.0, self.swing_duration)
        touchdown = ~self.prev_contact & contact & self.swing_active
        self.swing_sum += torch.where(touchdown, self.swing_duration, 0.0).sum(dim=0)
        self.swing_count += touchdown.sum(dim=0)
        self.swing_active &= ~contact
        self.swing_duration += self.swing_active * self.env.step_dt
        self.prev_contact.copy_(contact)
        self.prev_valid.copy_(moving)
        q = self.robot.data.joint_pos.torch[:, self.joint_ids]
        self.joint_min = torch.minimum(self.joint_min, torch.where(moving[:, None], q, float("inf")).amin(dim=0))
        self.joint_max = torch.maximum(self.joint_max, torch.where(moving[:, None], q, -float("inf")).amax(dim=0))

    def report(self):
        valid, moving = self.samples.cpu().tolist()
        print(f"\n[Evaluation diagnostics] simulation={self.steps * self.env.step_dt:.1f}s; "
              f"episodes={self.completed}; timeouts={self.timeouts}; failures={self.completed - self.timeouts}")
        print(f"Valid env-steps: {int(valid)}; moving env-steps: {int(moving)}")
        if moving == 0:
            print("No moving samples yet; use a nonzero velocity command for gait evaluation.")
            return
        command, vx, vy, xy_error, yaw_error = (self.tracking_sum / moving).cpu().tolist()
        print(f"[Velocity tracking] command vx={command:.3f}; actual vx={vx:.3f}; vy={vy:.3f} m/s; "
              f"mean XY error={xy_error:.3f} m/s; yaw-rate error={yaw_error:.3f} rad/s")
        single, double, flight = (self.support_sum / moving).cpu().tolist()
        print(f"[Foot contact] single={single:.1%}; double={double:.1%}; flight={flight:.1%}")
        for index, side in enumerate(("Left", "Right")):
            count = self.contact_sum[index].item()
            swings = int(self.swing_count[index].item())
            slip = (f"mean speed={self.slide_speed_sum[index].item() / count:.3f} m/s; "
                    f"ratio >0.02 m/s={self.slide_count[index].item() / count:.1%}") if count else "no contact samples"
            air_time = f"{self.swing_sum[index].item() / swings:.3f} s" if swings else "N/A"
            print(f"{side}: contact={count / moving:.1%}; {slip}; complete swings={swings}; mean air time={air_time}")
        print("[Joint excursion] moving-sample max-minus-min, radians (not per-stride amplitude)")
        for name, excursion in zip(self.joint_names, (self.joint_max - self.joint_min).cpu().tolist()):
            print(f"  {name}: {excursion:.3f}")


def configure_getup_contact_probes(env_cfg):
    """Only diagnostic playback adds one ground-filtered sensor per selected body."""
    from pxr import Usd, UsdPhysics
    from isaaclab.sensors import ContactSensorCfg

    configured = list(env_cfg.rewards.head_contact.params.get("ground_sensor_cfgs", ()))
    configured += list(env_cfg.terminations.success.params.get("ground_sensor_cfgs", ()))
    if len(configured) == 4:
        print("[GETUP] Reusing four ground-filtered training sensors")
        return [entity.name for entity in configured]

    stage = Usd.Stage.Open(env_cfg.scene.robot.spawn.usd_path)
    root = stage.GetDefaultPrim()
    names = ("head_yaw_link", "head_pitch_link", "left_ankle_roll_link", "right_ankle_roll_link")
    found = {}
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            found[prim.GetName()] = prim.GetPath().MakeRelativePath(root.GetPath()).pathString
    if not set(names).issubset(found):
        raise RuntimeError(f"Missing diagnostic rigid bodies: {set(names) - set(found)}")
    ground_stage = Usd.Stage.Open(env_cfg.scene.ground.spawn.usd_path)
    ground_root = ground_stage.GetDefaultPrim()
    planes = [prim for prim in ground_stage.Traverse() if prim.GetTypeName() == "Plane"]
    if len(planes) != 1 or not ground_root:
        raise RuntimeError("Expected a single collision plane in the diagnostic ground asset")
    ground_path = env_cfg.scene.ground.prim_path + "/" + planes[0].GetPath().MakeRelativePath(ground_root.GetPath()).pathString
    print(f"[GETUP] Ground collision filter: {ground_path}")
    for name in names:
        setattr(env_cfg.scene, f"getup_probe_{name}", ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + found[name],
            filter_prim_paths_expr=[ground_path],
            history_length=env_cfg.scene.contact_forces.history_length,
            update_period=0.0,
        ))
    return [f"getup_probe_{name}" for name in names]


def format_getup_diagnostic(summary, reset_checks, output_path):
    """Format diagnostic data for a terminal without changing evaluation statistics."""
    def pad(value, width):
        value = str(value)
        size = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in value)
        return value + " " * max(0, width - size)

    def percent(value):
        return "暂无" if value is None else f"{value:.2f}%"

    labels = {
        "height": "根高度", "tilt": "骨盆倾角", "linear_speed": "身体线速度",
        "angular_speed": "身体角速度", "knees": "左右膝角", "feet_contact": "双脚接地",
        "torso_tilt": "躯干倾角",
    }
    p = summary["thresholds"]
    limits = {
        "height": f"> {p['minimum_height']:.2f} m",
        "tilt": f"< {math.degrees(p['maximum_tilt']):.1f}°",
        "linear_speed": f"< {p['maximum_linear_speed']:.2f} m/s",
        "angular_speed": f"< {p['maximum_angular_speed']:.2f} rad/s",
        "knees": f"每侧绝对值 < {p['maximum_knee_angle']:.2f} rad",
        "feet_contact": f"每脚地面接触力 > {p['force_threshold']:.1f} N",
        "torso_tilt": f"< {math.degrees(p['maximum_torso_tilt']):.1f}°" if "maximum_torso_tilt" in p else "未配置，详见 JSON",
    }
    balanced = summary["balanced_evaluation"]
    use_quota = balanced["episodes_per_env"] > 0
    complete = balanced["complete"] if use_quota else None
    title = "均衡验收已完成" if complete else ("均衡验收进行中" if use_quota else "累计回合诊断")
    line = "─" * 76
    lines = ["", line, f"[GETUP 诊断] {title}  |  仿真时间 {summary['simulation_seconds']:.2f} s", line]
    lines.append(
        f"出生范围：参考进度 {summary['reset_start_phase']:.2f}～{summary['reset_end_phase']:.2f}"
        f"  |  随机参考帧概率 {summary['reset_random_frame_probability']:.0%}"
    )
    if summary.get("waist_pitch_target_override_rad") is not None:
        lines.append(f"对照实验：腰部俯仰目标覆盖为 {summary['waist_pitch_target_override_rad']:.3f} rad；结果不代表原策略。")
    if use_quota:
        suffix = "最终值" if complete else "暂计值，尚不能作为最终成功率"
        lines.extend([
            f"均衡成功率：{percent(balanced['success_percent'])}（{suffix}）",
            f"采样进度：{balanced['completed']}/{balanced['target_episodes']} 回合"
            f"，每个环境取前 {balanced['episodes_per_env']} 回合",
            f"结束原因：成功 {balanced['successes']}  |  超时 {balanced['timeouts']}  |  越界 {balanced['out_of_bounds']}",
        ])
    else:
        lines.append(
            f"累计成功：{summary['successes']}/{summary['completed']}"
            "（未启用等量验收，快速成功环境可能抬高比例）"
        )
    low, high = summary["completed_episodes_per_env_range"]
    lines.append(
        f"覆盖情况：{summary['envs_without_completed_episode']} 个环境尚未完成首回合"
        f"；各环境累计完成 {low}～{high} 回合"
    )
    if use_quota:
        lines.append(
            f"运行总计：{summary['completed']} 回合，仅供参考；上述均衡统计不重复计入快环境的额外回合"
        )
    lines.extend([
        "", "条件统计（占全部环境采样步的比例，不是回合成功率）",
        pad("条件", 16) + pad("成功要求", 34) + pad("达标占比", 12) + "仅此项失败",
    ])
    for name, label in labels.items():
        if name in summary["pass_rates"]:
            lines.append(
                pad(label, 16) + pad(limits[name], 34)
                + pad(f"{summary['pass_rates'][name]:.2%}", 12)
                + f"{summary['sole_blocker_rates'][name]:.2%}"
            )
    lines.extend([
        "“仅此项失败”指其他条件均达标、只有该项未达标的采样占比。",
        f"全部条件同时达标：{summary['all_conditions_rate']:.2%}"
        f"  |  最长连续达标：{summary['max_hold_seconds']:.2f}/{p['hold_time']:.2f} s",
        "", "接触与高度",
        f"根高度范围：{summary['root_height_range'][0]:.3f}～{summary['root_height_range'][1]:.3f} m",
        f"头部地面接触占比：{summary['raised_head_ground_contact_rate']:.2%}（头部抬起的采样中）"
        f"  |  双脚接地比例：{summary['raised_both_feet_ground_rate']:.2%}（同一范围）",
        f"头部地面接触力峰值：{max(summary['ground_force_max'][:2]):.1f} N（全程瞬时峰值）",
    ])
    if reset_checks:
        gravity_error = max(row["gravity_cache_max_error"] for row in reset_checks)
        target_errors = [row["previous_target_max_error"] for row in reset_checks if row["previous_target_max_error"] is not None]
        action_error = max(row["reset_action_max"] for row in reset_checks)
        target_text = f"{max(target_errors):.2e} rad" if target_errors else "未检查"
        normal = gravity_error <= 1e-5 and action_error <= 1e-6 and (not target_errors or max(target_errors) <= 1e-6)
        lines.extend([
            "", f"重置一致性：{'已检查项目正常' if normal else '发现不一致，请检查 JSON 的 reset_checks'}",
            f"重力方向最大误差 {gravity_error:.2e}  |  限速目标误差 {target_text}  |  残留动作 {action_error:.2e}",
        ])
    latest = summary.get("latest_env0", {})
    if latest:
        failed = [labels.get(name, name) for name, passed in latest["pass_conditions"].items() if not passed]
        lines.extend([
            "", "环境 0 最近样本（可能属于新回合，不代表整体结果）",
            f"参考帧 {latest['reference_frame']}  |  根高度 {latest['height']:.3f} m"
            f"  |  骨盆 {latest['tilt_deg']:.1f}°  |  躯干 {latest['torso_tilt_deg']:.1f}°",
            f"线速度 {latest['linear_speed']:.3f} m/s  |  角速度 {latest['angular_speed']:.3f} rad/s"
            f"  |  连续达标 {latest['hold_seconds']:.2f} s",
            "当前未达标：" + ("、".join(failed) if failed else "无"),
        ])
    lines.extend(["", f"完整数据：{output_path}", "反馈时请复制本段 [GETUP 诊断]；无需复制启动警告。", line])
    return "\n".join(lines)


class GetUpDiagnostics:
    """Sample after reward computation, before automatic resets erase terminal data."""

    gate_names = ("height", "tilt", "linear_speed", "angular_speed", "knees", "feet_contact", "torso_tilt")

    def __init__(self, env, output_path, probe_names=None):
        self.env = env
        self.robot = env.scene["robot"]
        self.output_path = output_path
        self.success_cfg = env.termination_manager.get_term_cfg("success")
        self.params = self.success_cfg.params
        self.head_cfg = env.reward_manager.get_term_cfg("head_contact").params
        self.sensor = env.scene.sensors["contact_forces"]
        self.head_sensor = self.sensor
        self.body_names = ("head_yaw_link", "head_pitch_link", "left_ankle_roll_link", "right_ankle_roll_link")
        self.body_ids, _ = self.robot.find_bodies(list(self.body_names), preserve_order=True)
        self.head_sensor_ids, _ = self.sensor.find_sensors(list(self.body_names[:2]), preserve_order=True)
        self.foot_sensor_ids, _ = self.sensor.find_sensors(list(self.body_names[2:]), preserve_order=True)
        probe_names = probe_names or [f"getup_probe_{name}" for name in self.body_names]
        self.probes = [env.scene.sensors[name] for name in probe_names]
        for name, sensor in [("training", self.sensor), *zip(self.body_names, self.probes)]:
            rigid = list(sensor.body_physx_view.prim_paths)
            contact = list(sensor.contact_view.sensor_paths)
            if rigid != contact:
                raise RuntimeError(f"Contact/body path mismatch in {name}; diagnostic data cannot be trusted")
        print("[GETUP 检查] 接触路径匹配正常；双脚和头部使用地面过滤信号。", flush=True)
        self.steps = self.completed = self.successes = self.timeouts = self.out_of_bounds = 0
        self.completed_per_env = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.episode_quota = args_cli.eval_episodes_per_env
        self.balanced_successes = self.balanced_timeouts = self.balanced_out_of_bounds = 0
        self.samples = self.all_pass = self.raised_samples = self.head_contact_samples = 0
        self.pass_counts = torch.zeros(len(self.gate_names), device=env.device, dtype=torch.long)
        self.only_fail_counts = torch.zeros_like(self.pass_counts)
        self.hold = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.max_hold = 0
        self.head_force_sum = torch.zeros(2, device=env.device)
        self.head_force_max = torch.zeros(2, device=env.device)
        self.ground_force_sum = torch.zeros(4, device=env.device)
        self.ground_force_max = torch.zeros(4, device=env.device)
        # Exact path lists avoid recursive name matches in this nested USD hierarchy.
        self.partner_names = list(self.sensor.body_names)
        all_paths = list(self.sensor.body_physx_view.prim_paths)
        self.partner_views = []
        for sensor in self.probes:
            paths = list(sensor.body_physx_view.prim_paths)
            filters = []
            for path in paths:
                prefix = path.split("/Robot/", 1)[0] + "/Robot/"
                by_name = {p.rsplit("/", 1)[-1]: p for p in all_paths if p.startswith(prefix)}
                filters.append([by_name[name] for name in self.partner_names])
            view = self.sensor._physics_sim_view.create_rigid_contact_view(paths, filter_patterns=filters)
            if view.sensor_count != env.num_envs or view.filter_count != len(self.partner_names):
                raise RuntimeError("Unexpected exact-path contact view layout")
            self.partner_views.append(view)
        self.partner_force_max = [torch.zeros(len(self.partner_names), device=env.device) for _ in self.probes]
        self.probe_head_net_max = torch.zeros(2, device=env.device)
        self.height_min = float("inf")
        self.height_max = -float("inf")
        self.raised_head_net = self.raised_head_ground = self.raised_feet_ground = 0
        self.records = []
        self.control_trace = []
        self.control_trace_printed = False
        self.latest = {}
        self.reported_steps = -1
        self._original_compute = env.reward_manager.compute
        self._original_reset_idx = env._reset_idx
        self.reset_records = []

        def compute_with_diagnostics(dt):
            reward = self._original_compute(dt)
            self.update()
            return reward

        env.reward_manager.compute = compute_with_diagnostics

        def reset_with_diagnostics(env_ids):
            self._original_reset_idx(env_ids)
            self._record_reset(env_ids)

        env._reset_idx = reset_with_diagnostics
        self._record_reset(torch.arange(env.num_envs, device=env.device), initial=True)

    def _record_reset(self, env_ids, initial=False):
        """Compare reported observations with the root pose after reset; do not repair caches."""
        ids = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        r = self.robot
        quat = r.data.root_link_pose_w.torch[ids, 3:7]
        gravity = torch.zeros_like(quat[:, :3])
        gravity[:, 2] = -1.0
        inverse_xyz = -quat[:, :3]
        cross = 2.0 * torch.cross(inverse_xyz, gravity, dim=-1)
        expected = gravity + quat[:, 3:4] * cross + torch.cross(inverse_xyz, cross, dim=-1)
        reported = r.data.projected_gravity_b.torch[ids]
        term = self.env.action_manager.get_term("joint_pos")
        position = r.data.joint_pos.torch[:, term._joint_ids][ids]
        previous = getattr(term, "_previous_target", None)
        record = dict(
            initial=initial, diagnostic_step=self.steps, reset_count=int(ids.numel()),
            first_env_id=int(ids[0]),
            first_reference_frame=int(self.env._getup_reference_frame_ids[ids[0]]),
            gravity_cache_max_error=float((reported - expected).abs().max()),
            first_reported_gravity=reported[0].tolist(), first_expected_gravity=expected[0].tolist(),
            previous_target_max_error=float((previous[ids] - position).abs().max()) if previous is not None else None,
            reset_action_max=float(self.env.action_manager.action[ids].abs().max()),
            first_root_height=float(r.data.root_link_pose_w.torch[ids[0], 2] - self.env.scene.env_origins[ids[0], 2]),
        )
        self.reset_records.append(record)
        abnormal = (
            record["gravity_cache_max_error"] > 1e-5
            or record["reset_action_max"] > 1e-6
            or (record["previous_target_max_error"] is not None and record["previous_target_max_error"] > 1e-6)
        )
        if initial or (abnormal and len(self.reset_records) <= 8):
            state = "异常" if abnormal else "正常"
            print(
                f"[GETUP 重置] {state} | 参考帧 {record['first_reference_frame']}"
                f" | 重力误差 {record['gravity_cache_max_error']:.2e}"
                f" | 限速目标误差 {record['previous_target_max_error']}"
                f" | 残留动作 {record['reset_action_max']:.2e}；完整记录见 JSON。",
                flush=True,
            )

    def update(self):
        e, p, r = self.env, self.params, self.robot
        height = r.data.root_pos_w.torch[:, 2] - e.scene.env_origins[:, 2]
        cosine = (-r.data.projected_gravity_b.torch[:, 2]).clamp(-1., 1.)
        tilt = cosine.acos()
        linear = r.data.root_lin_vel_w.torch.norm(dim=-1)
        angular = r.data.root_ang_vel_w.torch.norm(dim=-1)
        knees = r.data.joint_pos.torch[:, p["asset_cfg"].joint_ids].abs()
        feet_net_force = self.sensor.data.net_forces_w_history.torch[:, :, self.foot_sensor_ids].norm(dim=-1).amax(dim=1)
        if "ground_sensor_cfgs" in p:
            feet_force = torch.stack([
                e.scene.sensors[cfg.name].data.force_matrix_w_history.torch.norm(dim=-1).amax(dim=(1, 2, 3))
                for cfg in p["ground_sensor_cfgs"]
            ], dim=-1)
        else:
            feet_force = feet_net_force
        torso_valid = torch.ones_like(height, dtype=torch.bool)
        torso_tilt = torch.zeros_like(height)
        if p.get("torso_cfg") is not None:
            torso_quat = r.data.body_link_pose_w.torch[:, p["torso_cfg"].body_ids, 3:7]
            torso_cosine = 1.0 - 2.0 * (torso_quat[..., 0].square() + torso_quat[..., 1].square())
            torso_tilt = torso_cosine.clamp(-1.0, 1.0).acos().amax(dim=-1)
            torso_valid = (torso_cosine > math.cos(p["maximum_torso_tilt"])).all(dim=-1)
        gates = torch.stack((
            height > p["minimum_height"], cosine > math.cos(p["maximum_tilt"]),
            linear < p["maximum_linear_speed"], angular < p["maximum_angular_speed"],
            knees.amax(dim=-1) < p["maximum_knee_angle"],
            (feet_force > p["force_threshold"]).all(dim=-1),
            torso_valid,
        ), dim=-1)
        all_pass = gates.all(dim=-1)
        self.hold = torch.where(all_pass, self.hold + 1, 0)
        self.max_hold = max(self.max_hold, int(self.hold.max()))
        head = self.head_sensor.data.net_forces_w_history.torch[
            :, :, self.head_sensor_ids
        ].norm(dim=-1).amax(dim=1)
        ground_forces = []
        for index, sensor in enumerate(self.probes):
            matrix = sensor.data.force_matrix_w_history
            if matrix is None or sensor.contact_view.filter_count == 0:
                raise RuntimeError("Ground filter matched no shapes; cannot interpret ground contact")
            # [env, history, one sensor body, filter bodies, xyz]
            if sensor.contact_view.filter_count != len(sensor.cfg.filter_prim_paths_expr):
                raise RuntimeError("Unexpected partner-filter grouping; cannot label contact partners")
            force = matrix.torch[..., 0, :].norm(dim=-1).amax(dim=1).squeeze(-1)
            ground_forces.append(force)
        ground = torch.stack(ground_forces, dim=-1)
        import warp as wp
        exact_net = []
        for index, view in enumerate(self.partner_views):
            matrix = wp.to_torch(view.get_contact_force_matrix(dt=e.physics_dt))
            partner_force = matrix.norm(dim=-1).amax(dim=0)
            self.partner_force_max[index] = torch.maximum(self.partner_force_max[index], partner_force)
            if index < 2:
                exact_net.append(wp.to_torch(view.get_net_contact_forces(dt=e.physics_dt)).norm(dim=-1))
        probe_head_net = torch.stack(exact_net, dim=-1)
        self.probe_head_net_max = torch.maximum(self.probe_head_net_max, probe_head_net.amax(dim=0))
        self.height_min = min(self.height_min, float(height.min()))
        self.height_max = max(self.height_max, float(height.max()))
        head_z = r.data.body_link_pose_w.torch[:, self.body_ids[:2], 2] - e.scene.env_origins[:, None, 2]
        raised = (head_z > .30).all(dim=-1)
        threshold = self.head_cfg["force_threshold"]
        head_contact = (head > threshold).any(dim=-1)
        head_ground = (ground[:, :2] > threshold).any(dim=-1)
        self.steps += 1
        self.samples += e.num_envs
        self.pass_counts += gates.sum(dim=0)
        self.only_fail_counts += ((~gates) & ((~gates).sum(dim=-1, keepdim=True) == 1)).sum(dim=0)
        self.all_pass += int(all_pass.sum())
        self.head_contact_samples += int(head_contact.sum())
        self.raised_samples += int(raised.sum())
        self.raised_head_net += int((raised & head_contact).sum())
        self.raised_head_ground += int((raised & head_ground).sum())
        self.raised_feet_ground += int((raised & (ground[:, 2:] > p["force_threshold"]).all(dim=-1)).sum())
        self.head_force_sum += head.sum(dim=0)
        self.head_force_max = torch.maximum(self.head_force_max, head.amax(dim=0))
        self.ground_force_sum += ground.sum(dim=0)
        self.ground_force_max = torch.maximum(self.ground_force_max, ground.amax(dim=0))
        self.completed += int(e.reset_buf.sum())
        if self.episode_quota > 0:
            counted = e.reset_buf.bool() & (self.completed_per_env < self.episode_quota)
            self.balanced_successes += int((counted & e.termination_manager.get_term("success").bool()).sum())
            self.balanced_timeouts += int((counted & e.termination_manager.get_term("time_out").bool()).sum())
            self.balanced_out_of_bounds += int((counted & e.termination_manager.get_term("out_of_bounds").bool()).sum())
        self.completed_per_env += e.reset_buf.long()
        self.successes += int(e.termination_manager.get_term("success").sum())
        self.timeouts += int(e.termination_manager.get_term("time_out").sum())
        self.out_of_bounds += int(e.termination_manager.get_term("out_of_bounds").sum())
        self.latest = dict(
            height=float(height[0]), tilt_deg=float(tilt[0] * 180. / math.pi),
            torso_tilt_deg=float(torso_tilt[0] * 180. / math.pi),
            linear_speed=float(linear[0]), angular_speed=float(angular[0]),
            knee_angles=knees[0].tolist(), feet_force=feet_force[0].tolist(), feet_net_force=feet_net_force[0].tolist(),
            head_z=head_z[0].tolist(), head_net_force=head[0].tolist(),
            probe_head_net_force=probe_head_net[0].tolist(),
            ground_force=ground[0].tolist(),
            pass_conditions=dict(zip(self.gate_names, gates[0].tolist())),
            hold_seconds=float(self.hold[0]) * e.step_dt,
            reference_frame=int(e._getup_reference_frame_ids[0]),
        )
        if self.steps % 10 == 0 or bool(e.reset_buf[0]):
            self.records.append(dict(step=self.steps, episode_step=int(e.episode_length_buf[0]), **self.latest))
        episode_step = int(e.episode_length_buf[0])
        if args_cli.getup_control_trace and not any(row["episode_step"] == episode_step for row in self.control_trace):
            # Global steps identify the first episode even if it terminates early.
            if self.steps == episode_step and 0 < episode_step * e.step_dt <= 2.000001 and (episode_step == 1 or episode_step % 5 == 0):
                term = e.action_manager.get_term("joint_pos")
                probe_names = [name for name in (
                    "left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint",
                    "right_knee_joint", "waist_pitch_joint",
                ) if name in term._joint_names]
                joints = []
                # Recover the requested target before position/rate limits.
                requested = term.raw_actions * term._scale + term._offset
                before_joint_limit = requested
                if term.cfg.clip is not None:
                    before_joint_limit = requested.clamp(
                        min=term._clip[:, :, 0], max=term._clip[:, :, 1])
                def inverse_rotate(q, v):
                    xyz, w = q[:3], q[3]
                    t = 2.0 * torch.cross(xyz, v, dim=-1)
                    return v - w * t + torch.cross(xyz, t, dim=-1)
                def signed_pitch(q):
                    x, y, z, w = q.unbind(-1)
                    return torch.asin((2.0 * (w*y-z*x)).clamp(-1., 1.)) * (180. / math.pi)
                root_pose = r.data.root_link_pose_w.torch[0]
                root_q = root_pose[3:7]
                velocity_b = inverse_rotate(root_q, r.data.root_link_vel_w.torch[0, :3])
                omega_b = inverse_rotate(root_q, r.data.root_link_vel_w.torch[0, 3:6])
                torso_id = self.params['torso_cfg'].body_ids[0]
                torso_q = r.data.body_link_pose_w.torch[0, torso_id, 3:7]
                foot_positions = {}
                for foot_name in ('left_ankle_roll_link', 'right_ankle_roll_link'):
                    foot_id = r.body_names.index(foot_name)
                    relative = r.data.body_link_pose_w.torch[0, foot_id, :3] - root_pose[:3]
                    foot_positions[foot_name] = inverse_rotate(root_q, relative).tolist()
                for name in probe_names:
                    actual_id = self.robot.joint_names.index(name)
                    action_id = term._joint_names.index(name)
                    joints.append(dict(
                        name=name,
                        position=float(r.data.joint_pos.torch[0, actual_id]),
                        target=float(term.processed_actions[0, action_id]),
                        velocity=float(r.data.joint_vel.torch[0, actual_id]),
                        raw_action=float(term.raw_actions[0, action_id]),
                        requested_target=float(requested[0, action_id]),
                        lower_limit=float(r.data.joint_pos_limits.torch[0, actual_id, 0]),
                        upper_limit=float(r.data.joint_pos_limits.torch[0, actual_id, 1]),
                        position_limited_target=float(before_joint_limit[0, action_id].clamp(
                            min=r.data.joint_pos_limits.torch[0, actual_id, 0],
                            max=r.data.joint_pos_limits.torch[0, actual_id, 1])),
                    ))
                self.control_trace.append(dict(
                    episode_step=episode_step, time_s=episode_step * e.step_dt,
                    height=float(height[0]), torso_tilt_deg=float(torso_tilt[0] * 180. / math.pi),
                    linear_speed=float(linear[0]), angular_speed=float(angular[0]), joints=joints,
                    root_pitch_deg=float(signed_pitch(root_q)),
                    torso_pitch_deg=float(signed_pitch(torso_q)),
                    root_linear_velocity_b=velocity_b.tolist(),
                    root_angular_velocity_b=omega_b.tolist(),
                    feet_relative_to_root_b=foot_positions,
                    feet_ground_force=feet_force[0].tolist(),
                ))
        # Start a fresh hold counter after recording terminal states.
        self.hold[e.reset_buf] = 0

    def report(self):
        if not self.samples:
            return
        rates = (self.pass_counts / self.samples).tolist()
        balanced_completed = int(self.completed_per_env.clamp(max=self.episode_quota).sum()) if self.episode_quota else 0
        balanced = dict(
            episodes_per_env=self.episode_quota,
            target_episodes=self.env.num_envs * self.episode_quota,
            completed=balanced_completed, successes=self.balanced_successes,
            timeouts=self.balanced_timeouts, out_of_bounds=self.balanced_out_of_bounds,
            success_percent=round(100.0 * self.balanced_successes / balanced_completed, 2) if balanced_completed else None,
            complete=bool(self.episode_quota and (self.completed_per_env >= self.episode_quota).all()),
        )
        summary = dict(
            waist_pitch_target_override_rad=args_cli.getup_waist_pitch_target,
            reset_random_frame_probability=self.env.cfg.events.reset_from_reference.params["random_frame_probability"],
            reset_start_phase=self.env.cfg.events.reset_from_reference.params.get("start_phase", 0.0),
            reset_end_phase=self.env.cfg.events.reset_from_reference.params.get("end_phase", 0.95),
            feet_contact_source="ground_filtered" if "ground_sensor_cfgs" in self.params else "all_contacts",
            head_penalty_source="ground_filtered" if "ground_sensor_cfgs" in self.head_cfg else "all_contacts",
            thresholds={key: self.params[key] for key in (
                "minimum_height", "maximum_tilt", "maximum_linear_speed", "maximum_angular_speed",
                "maximum_knee_angle", "hold_time", "force_threshold", "maximum_torso_tilt",
            ) if key in self.params},
            simulation_seconds=self.steps * self.env.step_dt,
            samples=self.samples, completed=self.completed, successes=self.successes,
            timeouts=self.timeouts, out_of_bounds=self.out_of_bounds,
            success_rate=self.successes / self.completed if self.completed else None,
            envs_without_completed_episode=int((self.completed_per_env == 0).sum()),
            completed_episodes_per_env_range=[int(self.completed_per_env.min()), int(self.completed_per_env.max())],
            balanced_evaluation=balanced,
            pass_rates=dict(zip(self.gate_names, rates)),
            sole_blocker_rates=dict(zip(self.gate_names, (self.only_fail_counts / self.samples).tolist())),
            all_conditions_rate=self.all_pass / self.samples,
            max_hold_seconds=self.max_hold * self.env.step_dt,
            head_contact_rate=self.head_contact_samples / self.samples,
            head_net_force_mean=(self.head_force_sum / self.samples).tolist(),
            head_net_force_max=self.head_force_max.tolist(),
            probe_head_net_force_max=self.probe_head_net_max.tolist(),
            root_height_range=[self.height_min, self.height_max],
            head_partner_peak_forces={
                self.body_names[i]: {
                    name: float(value)
                    for name, value in zip(self.partner_names, self.partner_force_max[i])
                    if float(value) > self.head_cfg["force_threshold"]
                } for i, sensor in enumerate(self.probes[:2])
            },
            foot_partner_peak_forces={
                self.body_names[i]: {
                    name: float(value)
                    for name, value in zip(self.partner_names, self.partner_force_max[i])
                    if float(value) > self.params["force_threshold"]
                } for i in range(2, 4)
            },
            ground_force_mean=(self.ground_force_sum / self.samples).tolist(),
            ground_force_max=self.ground_force_max.tolist(),
            raised_head_samples=self.raised_samples,
            raised_head_net_contact_rate=self.raised_head_net / max(self.raised_samples, 1),
            raised_head_ground_contact_rate=self.raised_head_ground / max(self.raised_samples, 1),
            raised_both_feet_ground_rate=self.raised_feet_ground / max(self.raised_samples, 1),
            latest_env0=self.latest,
        )
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as stream:
            json.dump(dict(summary=summary, env0_samples=self.records, reset_checks=self.reset_records,
                           initial_control_trace=self.control_trace), stream, ensure_ascii=False, indent=2)
        print(format_getup_diagnostic(summary, self.reset_records, self.output_path), flush=True)
        if self.control_trace and not self.control_trace_printed:
            print("\n[GETUP 控制] 环境 0 首回合，目标为经过限位和限速后的 PD 位置目标。", flush=True)
            print("角度单位 rad，速度单位 rad/s；这不是关节力矩测量。", flush=True)
            labels = {"left_hip_pitch_joint": "左髋俯仰", "right_hip_pitch_joint": "右髋俯仰",
                      "left_knee_joint": "左膝", "right_knee_joint": "右膝", "waist_pitch_joint": "腰部俯仰"}
            print("[平衡诊断] 首回合前 2 s；机身坐标 +X 为前方、+Y 为左侧、+Z 为上方。", flush=True)
            print("俯仰为 xyzw 四元数的 ZYX 欧拉角，正值表示 +X 轴向下；大侧倾时需结合总倾角。", flush=True)
            print("脚位置是踝刚体原点相对骨盆的位置，不是足底接触点或质心投影。", flush=True)
            for row in self.control_trace:
                print(f"t={row['time_s']:.2f} s | 根高度 {row['height']:.3f} m"
                      f" | 躯干倾角 {row['torso_tilt_deg']:.1f}°"
                      f" | 线/角速度 {row['linear_speed']:.3f} m/s、{row['angular_speed']:.3f} rad/s", flush=True)
                v = row['root_linear_velocity_b']; w = row['root_angular_velocity_b']
                feet = row['feet_relative_to_root_b']
                print(f"  骨盆/躯干俯仰 {row['root_pitch_deg']:+.1f}° / {row['torso_pitch_deg']:+.1f}°"
                      f" | 机身 vx/vy/vz {v[0]:+.3f}/{v[1]:+.3f}/{v[2]:+.3f} m/s"
                      f" | 滚转/俯仰角速度 {w[0]:+.3f}/{w[1]:+.3f} rad/s", flush=True)
                print(f"  左/右踝相对骨盆 x {feet['left_ankle_roll_link'][0]:+.3f}/"
                      f"{feet['right_ankle_roll_link'][0]:+.3f} m"
                      f" | 地面力 {row['feet_ground_force']} N", flush=True)
                for joint in row["joints"]:
                    print(f"  {labels[joint['name']]}：实际 {joint['position']:+.3f}"
                          f" → 目标 {joint['target']:+.3f} | 实际速度 {joint['velocity']:+.3f}", flush=True)
                    if joint['name'] == 'waist_pitch_joint':
                        print(f"    腰原始动作 {joint['raw_action']:+.3f} → 缩放目标 {joint['requested_target']:+.3f}"
                              f" → 位置限位后 {joint['position_limited_target']:+.3f} → 最终目标 {joint['target']:+.3f} rad"
                              f" | 硬限位 [{joint['lower_limit']:+.3f}, {joint['upper_limit']:+.3f}]", flush=True)
            print("[GETUP 控制结束] 完整时序已保存到 JSON 的 initial_control_trace。", flush=True)
            self.control_trace_printed = True
        self.reported_steps = self.steps

    def close(self):
        self.env.reward_manager.compute = self._original_compute
        self.env._reset_idx = self._original_reset_idx


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        getup_diagnostics_enabled = args_cli.getup_diagnostics or (
            (args_cli.diagnostics or args_cli.eval_episodes > 0)
            and "GetUp" in task_name
        )
        getup_probe_names = None
        if getup_diagnostics_enabled:
            getup_probe_names = configure_getup_contact_probes(env_cfg)

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # configure_seed must be called after runner construction so that PyTorch deterministic settings
        # do not interfere with the runner's internal initialization.
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # export the trained policy to JIT and ONNX formats
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            # use the new export functions for rsl-rl >= 4.0.0
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None  # Not needed for rsl-rl >= 4.0.0
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT and ONNX
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt
        waist_override = None
        if args_cli.getup_waist_pitch_target is not None:
            if env.unwrapped.action_manager.active_terms != ["joint_pos"]:
                raise RuntimeError("Waist diagnostic requires a single joint_pos action term")
            term = env.unwrapped.action_manager.get_term("joint_pos")
            waist_index = term._joint_names.index("waist_pitch_joint")
            robot = env.unwrapped.scene["robot"]
            joint_index = robot.joint_names.index("waist_pitch_joint")
            limits = robot.data.joint_pos_limits.torch[:, joint_index]
            target = args_cli.getup_waist_pitch_target
            if bool(((target < limits[:, 0]) | (target > limits[:, 1])).any()):
                raise ValueError("Waist diagnostic target exceeds joint position limits")
            scale = term._scale[:, waist_index] if torch.is_tensor(term._scale) else term._scale
            offset = term._offset[:, waist_index] if torch.is_tensor(term._offset) else term._offset
            if bool(torch.as_tensor(scale).eq(0).any()):
                raise ValueError("Waist action scale cannot be zero")
            waist_override = (waist_index, (target - offset) / scale)
            print(f"[GETUP 对照] 腰部俯仰目标设为 {target:.3f} rad，仍经过原有限位、限速和 PD 控制。", flush=True)

        # reset environment
        obs = env.get_observations()
        timestep = 0
        diagnostics = None
        if getup_diagnostics_enabled:
            report_path = args_cli.diagnostic_report or os.path.join(log_dir, "diagnostics", "getup_balance_latest.json")
            diagnostics = GetUpDiagnostics(env.unwrapped, os.path.abspath(report_path), getup_probe_names)
        elif args_cli.diagnostics or args_cli.eval_episodes > 0:
            diagnostics = LocomotionDiagnostics(env.unwrapped)
        diagnostic_interval_steps = max(1, round(args_cli.diagnostic_interval / dt))
        # simulate environment
        try:
            while True:
                start_time = time.time()
                # run everything in inference mode
                with torch.inference_mode():
                    # agent stepping
                    actions = policy(obs)
                    if waist_override is not None:
                        actions = actions.clone()
                        actions[:, waist_override[0]] = waist_override[1]
                    # 保存本步实际命令，避免自动 reset 后读到下一 episode 的命令。
                    command = (
                        env.unwrapped.command_manager.get_command(diagnostics.command_name).clone()
                        if isinstance(diagnostics, LocomotionDiagnostics) else None
                    )
                    # env stepping
                    obs, _, dones, _ = env.step(actions)
                    if isinstance(diagnostics, LocomotionDiagnostics):
                        diagnostics.update(command, dones)
                    # reset recurrent states for episodes that have terminated
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)
                if diagnostics is not None:
                    evaluation_done = (
                        args_cli.eval_episodes > 0
                        and diagnostics.completed >= args_cli.eval_episodes
                    )
                    if args_cli.eval_episodes_per_env > 0:
                        evaluation_done = bool((diagnostics.completed_per_env >= args_cli.eval_episodes_per_env).all())
                    if diagnostics.steps % diagnostic_interval_steps == 0 or evaluation_done:
                        diagnostics.report()
                    if evaluation_done:
                        break
                if args_cli.video:
                    timestep += 1
                    if timestep == args_cli.video_length:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            if diagnostics is not None:
                diagnostics.report()
        finally:
            # 正常结束、Ctrl+C 和异常均释放仿真资源。
            if isinstance(diagnostics, GetUpDiagnostics):
                if diagnostics.steps != diagnostics.reported_steps:
                    diagnostics.report()
                diagnostics.close()
            env.close()


if __name__ == "__main__":
    main()
