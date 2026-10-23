"""多动作、五帧观测的新版起身任务。"""
from copy import deepcopy

from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from .x2_getup_env_cfg import _ground_sensor
from .mdp import getup as getup_mdp
from .x2_getup_env_cfg import PROJECT_DIR
from .x2_style_mimic_env_cfg import POLICY_JOINT_NAMES
from .mdp.getup_progress import GetUpProgress

from .x2_style_mimic_env_cfg import X2StyleMimicEnvCfg
from .x2_getup_env_cfg import (
    GetUpSceneCfg,
    GetUpObservationsCfg,
    GetUpRewardsCfg,
    GetUpTerminationsCfg,
    GetUpActionsCfg,
)

MOTION_MANIFEST = (
    PROJECT_DIR
    / "data"
    / "motions"
    / "prone_getup"
    / "motions.json"
)

@configclass
class MultiMotionObservationsCfg(GetUpObservationsCfg):
    """保留原观测内容，为策略和特权观测增加五帧历史。"""

    @configclass
    class PolicyCfg(GetUpObservationsCfg.PolicyCfg):

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class PrivilegedCfg(GetUpObservationsCfg.PrivilegedCfg):

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5
            self.flatten_history_dim = True

    privileged: PrivilegedCfg = PrivilegedCfg()

@configclass
class MultiMotionGetUpEventsCfg:
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
class MultiMotionSceneCfg(GetUpSceneCfg):
    left_hand_ground_contact = _ground_sensor(
        "{ENV_REGEX_NS}/Robot/Geometry/base_link/.*left_wrist_roll_link"
    )
    right_hand_ground_contact = _ground_sensor(
        "{ENV_REGEX_NS}/Robot/Geometry/base_link/.*right_wrist_roll_link"
    )

@configclass
class MultiMotionRewardsCfg(GetUpRewardsCfg):
    success = None

    transition = RewTerm(
        func=getup_mdp.getup_transition_reward,
        weight=3.0,
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
            "foot_sensor_cfgs": (
                SceneEntityCfg("left_foot_ground_contact"),
                SceneEntityCfg("right_foot_ground_contact"),
            ),
            "hand_sensor_cfgs": (
                SceneEntityCfg("left_hand_ground_contact"),
                SceneEntityCfg("right_hand_ground_contact"),
            ),
        },
    )

    # 多动作任务使用新的转换运动惩罚，替换原 rise_speed。
    rise_speed = RewTerm(
        func=getup_mdp.transition_motion_penalty,
        weight=-0.50,
    )

    support_transfer = RewTerm(
        func=getup_mdp.getup_support_transfer_reward,
        weight=2.0,
        params=deepcopy(transition.params),
    )

    completion = RewTerm(
        func=getup_mdp.getup_completion_reward,
        weight=6.0,
        params=deepcopy(transition.params),
    )


@configclass
class MultiMotionTerminationsCfg(GetUpTerminationsCfg):
    # 禁用原成功终止项。
    success = None

    # 复制原成功条件，作为独立统计项。
    progress = deepcopy(GetUpTerminationsCfg().success)

    def __post_init__(self):
        self.progress.func = GetUpProgress
        self.progress.params["hold_time"] = 3.0



@configclass
class X2MultiMotionGetUpEnvCfg(X2StyleMimicEnvCfg):
    scene: MultiMotionSceneCfg = MultiMotionSceneCfg(
        num_envs=1,
        env_spacing=3.0,
    )
    observations: MultiMotionObservationsCfg = MultiMotionObservationsCfg()
    rewards: MultiMotionRewardsCfg = MultiMotionRewardsCfg()
    terminations: MultiMotionTerminationsCfg = MultiMotionTerminationsCfg()
    events: MultiMotionGetUpEventsCfg = MultiMotionGetUpEventsCfg()
    actions: GetUpActionsCfg = GetUpActionsCfg()

    assist_force_levels: tuple = (
        120.0, 90.0, 60.0, 30.0, 0.0
    )
    assist_initial_stage: int = 0

    assist_zero_probability: float = 0.20
    assist_full_until_s: float = 2.0
    assist_zero_after_s: float = 5.0
    assist_curriculum_enabled: bool = True
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

        # 它由 Isaac Lab 的 ObservationManager 读取，决定生成 amp 观测时是否自动堆叠历史。
        # AMP 历史由算法单独维护，此处仍输出单帧 80 维。
        self.observations.amp.history_length = 0

        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = self.decimation

        # 最长参考约 12 秒，留出起身后的稳定保持时间。
        self.episode_length_s = 20.0
        self.scene.robot.spawn.rigid_props.disable_gravity = False
