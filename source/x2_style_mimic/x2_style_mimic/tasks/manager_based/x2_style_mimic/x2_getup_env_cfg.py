from pathlib import Path

from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import mdp
from .mdp import getup as getup_mdp
from .mdp.amp import AMP_KEY_BODY_NAMES, amp_observation
from .mdp.actions_cfg import LimitedJointPositionActionCfg
from .x2_style_mimic_env_cfg import (
    POLICY_JOINT_NAMES,
    ObservationsCfg,
    X2StyleMimicEnvCfg,
    X2StyleMimicSceneCfg,
)

PROJECT_DIR = Path(__file__).resolve().parents[6]

_ROBOT_PATH = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
_TORSO_PATH = (
    _ROBOT_PATH
    + "/waist_yaw_link/waist_pitch_link/torso_link"
)
_GROUND_PATH = "/World/ground/GroundPlane/CollisionPlane"

REFERENCE_FILE = (
    PROJECT_DIR
    / "data"
    / "motions"
    / "x2_getup_v0_50hz.npz"
)

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


@configclass
class GetUpObservationsCfg(ObservationsCfg):
    """保留策略和特权观测，额外添加 AMP 状态。"""

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

    amp : AmpCfg = AmpCfg()

@configclass
class GetUpEventsCfg:

    reset_from_reference = EventTerm(
        func=getup_mdp.reset_from_reference,
        mode="reset",
        params={
            "motion_file": str(REFERENCE_FILE),
            "random_frame_probability": 0.8,
            "start_phase": 0.0,
            "end_phase": 0.95,
        },
    )

@configclass
class GetUpRewardsCfg:

    height = RewTerm(
        func=getup_mdp.height_reward,
        weight=3.0,
        params={
            "minimum_height": 0.05,
            "target_height": 0.55,
            "inverted_cutoff": -0.2,
        },
    )

    upright = RewTerm(
        func=getup_mdp.upright_reward,
        weight=1.0,
        params={
            "minimum_height": 0.15,
            "full_height": 0.50,
        },
    )

    crouch_stability = RewTerm(
        func=getup_mdp.crouch_stability_reward,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_knee_joint",
                    "right_knee_joint",
                ],
                preserve_order=True,
            ),
            "torso_cfg": SceneEntityCfg(
                "robot",
                body_names=["torso_link"],
            ),
            "ground_sensor_cfgs": (
                SceneEntityCfg("left_foot_ground_contact"),
                SceneEntityCfg("right_foot_ground_contact"),
            ),
        },
    )

    rise_speed = RewTerm(
        func=getup_mdp.rise_speed_penalty,
        weight=-1.0,
    )

    head_contact = RewTerm(
        func=getup_mdp.head_contact_penalty,
        weight=-2.0,
        params={
            "force_threshold": 10.0,  # N
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

    success = RewTerm(
        func=getup_mdp.success_reward,
        weight=500.0,
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

    success = DoneTerm(
        func=getup_mdp.GetUpSuccess,
        params={
            "minimum_height": 0.50,          # m
            "maximum_tilt": 0.35,            # rad，约 20°
            "maximum_linear_speed": 0.20,    # m/s
            "maximum_angular_speed": 0.50,   # rad/s
            "maximum_knee_angle": 1.50,      # rad，避免深蹲算成功
            "hold_time": 3.00,               # s，连续保持
            "force_threshold": 5.0,          # N
            "maximum_torso_tilt": 0.55,      # rad
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
class X2GetUpEnvCfg(X2StyleMimicEnvCfg):

    scene: GetUpSceneCfg = GetUpSceneCfg(
        num_envs=1,
        env_spacing=3.0,
    )
    observations: GetUpObservationsCfg = GetUpObservationsCfg()
    rewards: GetUpRewardsCfg = GetUpRewardsCfg()
    terminations: GetUpTerminationsCfg = GetUpTerminationsCfg()
    events: GetUpEventsCfg = GetUpEventsCfg()
    actions: GetUpActionsCfg = GetUpActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 起身任务使用 PGS 求解器。
        if self.sim.physics is None:
            self.sim.physics = PhysxCfg(solver_type=0)
        else:
            self.sim.physics.solver_type = 0

        if not REFERENCE_FILE.is_file():
            raise FileNotFoundError(
                f"The get-up reference file does not exist:{REFERENCE_FILE}"
            )

        # 起身任务暂时不接收行走速度指令。
        self.commands = None
        self.observations.policy.velocity_command = None

        # 保持 500 Hz 物理、50 Hz 控制。
        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0

        # 真正运行物理，保留重力。
        self.scene.robot.spawn.rigid_props.disable_gravity = False