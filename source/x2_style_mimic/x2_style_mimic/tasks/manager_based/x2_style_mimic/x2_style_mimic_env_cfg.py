# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""灵犀 X2 的模型加载、重置与关节控制检查环境。"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from x2_style_mimic.assets import X2_ULTRA_CFG
from .x2_walk_params import (
    COMMAND_NAME,
    FOOT_BODY_NAMES,
    FOOT_CONTACT_FORCE_THRESHOLD,
    )

from . import mdp

# 身体 29 个关节，不包含头部。
# 显式指定策略顺序，与部署例程中的 action_seq 对齐。
POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]


@configclass
class X2StyleMimicSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
        ),
    )

    robot: ArticulationCfg = X2_ULTRA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.9, 0.9, 0.9),
        ),
    )

@configclass
class ActionsCfg:

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=POLICY_JOINT_NAMES,
        preserve_order=True,

        scale=0.25,
        use_default_offset=True,
    )

@configclass
class CommandsCfg:

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.5),  # m/s
            lin_vel_y=(0.0, 0.0),  # m/s
            ang_vel_z=(0.0, 0.0),  # rad/s
        ),
    )

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        velocity_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": COMMAND_NAME},
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=POLICY_JOINT_NAMES,
                    preserve_order=True,
                ),
            },
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=POLICY_JOINT_NAMES,
                    preserve_order=True,
                ),
            },
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class PrivilegedCfg(ObsGroup):

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    privileged :PrivilegedCfg = PrivilegedCfg()

@configclass
class RewardsCfg:

    # 跟踪水平速度指令。
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={
            "command_name": COMMAND_NAME,
            "std": 0.25,
        },
    )

    # 跟踪航向角速度指令。
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={
            "command_name": COMMAND_NAME,
            "std": 0.5,
        },
    )

    # 抑制机身上下弹跳。
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
    )

    # 抑制机身滚转、俯仰角速度。
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    # 鼓励根刚体保持竖直。
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    # 抑制相邻策略步之间的动作突变。
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # 惩罚身体关节超出软位置限制。
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=POLICY_JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    # 首版限制腰部和上肢过度偏离默认姿态。
    # 加入 AMP 后需要重新评估，避免压制风格动作。
    upper_body_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_.*_joint",
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            ),
        },
    )

    # 根据近期接触状态，抑制支撑脚水平滑动。
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=list(FOOT_BODY_NAMES),
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=list(FOOT_BODY_NAMES),
                preserve_order=True,
            ),
            "force_threshold": FOOT_CONTACT_FORCE_THRESHOLD,
        },
    )

    # 失败终止时惩罚，正常超时不罚。
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-20.0,
    )

@configclass
class TerminationsCfg:

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    low_root_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3},
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0},    # rad，约 57°
    )

@configclass
class EventCfg:

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class X2StyleMimicEnvCfg(ManagerBasedRLEnvCfg):

    scene: X2StyleMimicSceneCfg = X2StyleMimicSceneCfg(
        num_envs=1,
        env_spacing=3.0,
    )
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = self.decimation

        self.episode_length_s = 10.0

        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.7)

