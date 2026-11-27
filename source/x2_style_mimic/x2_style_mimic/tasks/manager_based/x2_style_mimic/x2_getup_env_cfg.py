from pathlib import Path

from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.utils.noise import UniformNoiseCfg

from . import mdp
from .mdp.rewards import ActionSecondDifferencePenalty
from .mdp import getup as getup_mdp
from .mdp.amp import (
    AMP_KEY_BODY_NAMES,
    amp_observation,
    critic_body_geometry,
    critic_root_height,
)
from .mdp.actions_cfg import LimitedJointPositionActionCfg
from .mdp.getup_progress import GetUpProgress
from .x2_style_mimic_env_cfg import (
    POLICY_JOINT_NAMES,
    ObservationsCfg,
    X2StyleMimicEnvCfg,
    X2StyleMimicSceneCfg,
)
from .mdp.self_contact_cfg import SelfContactSensorCfg

PROJECT_DIR = Path(__file__).resolve().parents[6]
# 环境重置只使用原有 8 条参考，保持已验证的出生分布。
MOTION_MANIFEST = (
    PROJECT_DIR / "data" / "motions" / "prone_getup2" / "motions_birth.json"
)
# AMP 使用全部参考，新增动作的采样比例由清单中的 weight 控制。
AMP_MOTION_MANIFEST = (
    PROJECT_DIR / "data" / "motions" / "prone_getup2" / "motions.json"
)
_ROBOT_PATH = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
_TORSO_PATH = _ROBOT_PATH + "/waist_yaw_link/waist_pitch_link/torso_link"
_GROUND_PATH = "/World/ground/GroundPlane/CollisionPlane"


def _ground_sensor(body_path):
    return ContactSensorCfg(
        prim_path=body_path,
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=[_GROUND_PATH],
    )

@configclass
class GetUpSceneCfg(X2StyleMimicSceneCfg):
    head_yaw_ground_contact = _ground_sensor(
        _TORSO_PATH + "/head_yaw_link"
    )

    head_pitch_ground_contact = _ground_sensor(
        _TORSO_PATH + "/head_yaw_link/head_pitch_link"
    )

    left_foot_ground_contact = _ground_sensor(
        _ROBOT_PATH
        + "/left_hip_pitch_link/left_hip_roll_link"
        + "/left_hip_yaw_link/left_knee_link"
        + "/left_ankle_pitch_link/left_ankle_roll_link"
    )

    right_foot_ground_contact = _ground_sensor(
        _ROBOT_PATH
        + "/right_hip_pitch_link/right_hip_roll_link"
        + "/right_hip_yaw_link/right_knee_link"
        + "/right_ankle_pitch_link/right_ankle_roll_link"
    )

    left_hand_ground_contact = _ground_sensor(
        _ROBOT_PATH + "/.*left_wrist_roll_link"
    )

    right_hand_ground_contact = _ground_sensor(
        _ROBOT_PATH + "/.*right_wrist_roll_link"
    )

    self_contact = SelfContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=0.0,
        debug_vis=False,
    )

@configclass
class GetUpObservationsCfg(ObservationsCfg):

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):

        def __post_init__(self):
            super().__post_init__()
            self.velocity_command = None
            self.history_length = 5
            self.flatten_history_dim = True

            self.enable_corruption = True
            self.base_ang_vel.noise = UniformNoiseCfg(n_min=-0.05, n_max=0.05,)
            self.projected_gravity.noise = UniformNoiseCfg(n_min=-0.01, n_max=0.01,)
            self.joint_pos_rel.noise = UniformNoiseCfg(n_min=-0.01, n_max=0.01,)
            self.joint_vel_rel.noise = UniformNoiseCfg(n_min=-0.10, n_max=0.10,)
            self.last_action.noise = None

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class PrivilegedCfg(ObservationsCfg.PrivilegedCfg):

        root_height = ObsTerm(
            func=critic_root_height,
        )

        body_geometry = ObsTerm(
            func=critic_body_geometry,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=list(AMP_KEY_BODY_NAMES),
                    preserve_order=True,
                ),
            },
        )

        getup_phase = ObsTerm(
            func=getup_mdp.getup_phase_observation,
        )

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5
            self.flatten_history_dim = True

    privileged: PrivilegedCfg = PrivilegedCfg()

    @configclass
    class AmpCfg(ObsGroup):
        state = ObsTerm(
            func=amp_observation,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=POLICY_JOINT_NAMES,
                    body_names=list(AMP_KEY_BODY_NAMES),
                    preserve_order=True,
                ),
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

            # AMP 窗口由算法维护，ObservationManager 输出单帧。
            self.history_length = 0

    amp: AmpCfg = AmpCfg()

@configclass
class GetUpEventsCfg:
    reset_from_reference = EventTerm(
        func=getup_mdp.reset_from_motion_dataset,
        mode="reset",
        params={
            "manifest_file": str(MOTION_MANIFEST),
            "policy_joint_names": list(POLICY_JOINT_NAMES),
            "start_phase": 0.0,
            "end_phase": 0.0,
            "use_reference_velocity": True,
            "zero_initial_frame_velocity": True,
            "height_offset": 0.01,
            "perturb_state_bank": str(PROJECT_DIR / "data/fallen_states/x2_fallen_v2_hipfiltered/bank.npz"),
            "recovery_state_bank": str(PROJECT_DIR / "data/fallen_states/x2_pre_entrapment_v1/train_validated.npz"),
            # 首帧、近邻、RSI、仿真摔倒、动态恢复。
            "birth_probabilities": (0.30, 0.10, 0.50, 0.09, 0.01),
            "transition_frame_ranges": (
                (0, 543),  # x2_getup_139_16_umr_v1_50hz.npz
                (0, 598),  # x2_getup_139_17_umr_v1_50hz.npz
                (0, 339),  # x2_lafan_subject2_prone_getup_umr_50hz.npz
                (0, 283),  # x2_lafan_subject2_supine_roll_getup_umr_50hz.npz
                (0, 543),  # x2_getup_139_16_umr_v1_50hz_mirror.npz
                (0, 598),  # x2_getup_139_17_umr_v1_50hz_mirror.npz
                (0, 339),  # x2_lafan_subject2_prone_getup_umr_50hz_mirror.npz
                (0, 283),  # x2_lafan_subject2_supine_roll_getup_umr_50hz_mirror.npz
            ),
        },
    )

    randomize_material = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 0.05),    # 机器人碰撞材质的恢复系数
            "num_buckets": 65,                   # 预采样65组材质参数，再随机分配给碰撞形状
            "make_consistent": True,             # 保证动态摩擦系数不超过静态摩擦系数
        },
    )

    randomize_mass = EventTerm(
        func=isaac_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["base_link", "torso_link"],
            ),
            "mass_distribution_params": (0.90, 1.10),   # 质量分布范围，随机缩放质量。
            "operation": "scale",                       # 缩放质量，而不是直接设置质量。
            "distribution": "uniform",                  # 随机分布类型，均匀分布。
            "recompute_inertia": True,                  # 缩放质量后重新计算惯性张量。
        },
    )

    randomize_gains = EventTerm(
        func=isaac_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=POLICY_JOINT_NAMES,
                preserve_order=True,
            ),
            "stiffness_distribution_params": (0.90, 1.10),   # 随机缩放刚度。
            "damping_distribution_params": (0.90, 1.10),     # 随机缩放阻尼。
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    standing_perturbation = EventTerm(
        func=getup_mdp.perturb_standing_velocity,
        mode="interval",
        interval_range_s=(4.0, 7.0),
        is_global_time=False,
        params={
            "probability": 0.35,
            "min_delta_v": 0.15,
            "max_delta_v": 0.35,
            "ground_sensor_cfgs": (
                SceneEntityCfg("left_foot_ground_contact"),
                SceneEntityCfg("right_foot_ground_contact"),
            ),
        },
    )

@configclass
class GetUpRewardsCfg:
    standing = RewTerm(
        func=getup_mdp.standing_quality_reward,
        weight=10.0,
        params={
            "torso_cfg": SceneEntityCfg(
                "robot",
                body_names=["torso_link"],
                preserve_order=True,
            ),
            "ground_sensor_cfgs": (
                SceneEntityCfg("left_foot_ground_contact"),
                SceneEntityCfg("right_foot_ground_contact"),
            ),
            "wrist_sensor_cfgs": (
                SceneEntityCfg("left_hand_ground_contact"),
                SceneEntityCfg("right_hand_ground_contact"),
            ),
            "limb_joint_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_yaw_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_roll_joint",
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_yaw_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_roll_joint",
                ],
                preserve_order=True,
            ),
            # 顺序必须与上面的 joint_names 完全一致。
            "limb_joint_targets": (
                0.20,  0.15, 0.0, -0.30, 0.0, 0.0, 0.0,
                0.20, -0.15, 0.0, -0.30, 0.0, 0.0, 0.0,
            ),
            "feet_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
                preserve_order=True,
            ),
            "stance_width": 0.34,
            "toe_out_rad": 0.08726646,
            "limb_fraction": 0.20,
        },
    )

    success_bonus = RewTerm(
        func=getup_mdp.GetUpSuccessBonus,
        weight=30.0,
    )

    head_contact = RewTerm(
        func=getup_mdp.head_contact_penalty,
        weight=-2.0,
        params={
            "force_threshold": 10.0,
            "ground_sensor_cfgs": (
                SceneEntityCfg("head_yaw_ground_contact"),
                SceneEntityCfg("head_pitch_ground_contact"),
            ),
        },
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.010,
    )

    action_second_difference = RewTerm(
        func=ActionSecondDifferencePenalty,
        weight=-0.000,
    )

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=POLICY_JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=POLICY_JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    self_contact = RewTerm(
        func=mdp.self_contact_penalty,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("self_contact")},
    )

    out_of_bounds_failure = RewTerm(
        func=getup_mdp.out_of_bounds_failure,
        weight=-5.0,
    )

@configclass
class GetUpTerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    out_of_bounds = DoneTerm(
        func=getup_mdp.out_of_bounds,
        params={
            "max_distance": 2.0,
        },
    )

    # GetUpProgress 更新成功统计，但成功不会结束回合。
    progress = DoneTerm(
        func=GetUpProgress,
        params={
            "minimum_height": 0.50,
            "maximum_tilt": 0.35,
            "maximum_linear_speed": 0.20,
            "maximum_angular_speed": 0.50,
            "maximum_knee_angle": 1.50,
            "hold_time": 3.00,
            "force_threshold": 5.0,
            "maximum_torso_tilt": 0.55,
            "torso_cfg": SceneEntityCfg(
                "robot",
                body_names=["torso_link"],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_knee_joint",
                    "right_knee_joint",
                ],
                preserve_order=True,
            ),
            "ground_sensor_cfgs": (
                SceneEntityCfg("left_foot_ground_contact"),
                SceneEntityCfg("right_foot_ground_contact"),
            ),
        },
    )

    # 一段时间内高度没有明显进展的早停
    no_height_progress = DoneTerm(
        func=getup_mdp.NoHeightProgress,
        time_out=False,
        params={
            "min_progress": 0.02,
            "grace_time": 8.0,
            "patience": 4.0,
        },
    )

@configclass
class GetUpActionsCfg:
    joint_pos = LimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=POLICY_JOINT_NAMES,
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
        target_velocity_fraction=0.5,
    )


@configclass
class X2MultiMotionGetUpEnvCfg(X2StyleMimicEnvCfg):
    scene: GetUpSceneCfg = GetUpSceneCfg(
        num_envs=1,
        env_spacing=3.0,
    )
    observations: GetUpObservationsCfg = GetUpObservationsCfg()
    rewards: GetUpRewardsCfg = GetUpRewardsCfg()
    terminations: GetUpTerminationsCfg = GetUpTerminationsCfg()
    events: GetUpEventsCfg = GetUpEventsCfg()
    actions: GetUpActionsCfg = GetUpActionsCfg()

    # 从零训练使用高度衰减辅助课程，保留 30% 零辅助回合。
    assist_force_levels: tuple = (100.0, 75.0, 50.0, 25.0, 0.0)
    assist_initial_stage: int = 0
    assist_zero_probability: float = 0.30
    assist_full_below_height: float = 0.20
    assist_zero_above_height: float = 0.50
    assist_curriculum_enabled: bool = True
    assist_curriculum_window: int = 64
    assist_curriculum_threshold: float = 0.80
    assist_curriculum_final_threshold: float = 0.70

    # AMP 奖励按连续稳定时长平滑退出。
    amp_gate_fade_start_s: float = 0.10
    amp_gate_fade_end_s: float = 0.50
    # 超过该连续稳定时长的状态，不进入 AMP 策略样本窗口。
    amp_policy_sample_max_hold_s: float = 0.10

    def __post_init__(self):
        super().__post_init__()

        if not MOTION_MANIFEST.is_file():
            raise FileNotFoundError(MOTION_MANIFEST)

        if self.sim.physics is None:
            self.sim.physics = PhysxCfg(solver_type=0)
        else:
            self.sim.physics.solver_type = 0

        self.commands = None
        self.observations.policy.velocity_command = None

        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0

        self.scene.robot.spawn.rigid_props.disable_gravity = False
