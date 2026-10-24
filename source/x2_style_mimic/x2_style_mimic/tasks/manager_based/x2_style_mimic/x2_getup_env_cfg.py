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

from . import mdp
from .mdp import getup as getup_mdp
from .mdp.amp import (
    AMP_KEY_BODY_NAMES,
    amp_observation,
    body_observation,
)
from .mdp.actions_cfg import LimitedJointPositionActionCfg
from .mdp.getup_progress import GetUpProgress
from .x2_style_mimic_env_cfg import (
    POLICY_JOINT_NAMES,
    ObservationsCfg,
    X2StyleMimicEnvCfg,
    X2StyleMimicSceneCfg,
)

PROJECT_DIR = Path(__file__).resolve().parents[6]
MOTION_MANIFEST = (
    PROJECT_DIR
    / "data"
    / "motions"
    / "prone_getup"
    / "motions.json"
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

def _relative_body_terms():
    support = ObsTerm(
        func=body_observation,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "torso_link",
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
                preserve_order=True,
            ),
        },
    )

    hands = ObsTerm(
        func=body_observation,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_wrist_roll_link",
                    "right_wrist_roll_link",
                ],
                preserve_order=True,
            ),
            "wrist_only": True,
        },
    )

    return support, hands

@configclass
class GetUpObservationsCfg(ObservationsCfg):

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        relative_support, relative_hands = _relative_body_terms()

        def __post_init__(self):
            super().__post_init__()
            self.velocity_command = None
            self.history_length = 5
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class PrivilegedCfg(ObservationsCfg.PrivilegedCfg):
        relative_support, relative_hands = _relative_body_terms()

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
            "initial_frame_probability": 0.5,
            "transition_frame_ranges": (
                (170, 225),  # x2_getup_v0
                (285, 350),  # 139_16
                (375, 500),  # 139_17
                (195, 275),  # 140_02
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
        },
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
        weight=-0.005,
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

    # 保留现有环境所需字段，默认使用零辅助力。
    assist_force_levels: tuple = (0.0,)
    assist_initial_stage: int = 0
    assist_zero_probability: float = 1.0
    assist_full_until_s: float = 2.0
    assist_zero_after_s: float = 5.0
    assist_curriculum_enabled: bool = False
    assist_curriculum_window: int = 64
    assist_curriculum_threshold: float = 0.80

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
