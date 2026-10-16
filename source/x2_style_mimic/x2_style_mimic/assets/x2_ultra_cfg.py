"""灵犀 X2 Ultra v1.3.0 资产配置。"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


ASSETS_DIR = Path(__file__).resolve().parent

X2_ULTRA_USD = (
    ASSETS_DIR
    / "robots"
    / "x2_ultra"
    / "x2_ultra"
    / "x2_ultra.usda"
)

if not X2_ULTRA_USD.is_file():
    raise FileNotFoundError(f"X2 USD not found: {X2_ULTRA_USD}")

X2_ULTRA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(X2_ULTRA_USD),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,

            ".*_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            ".*_elbow_joint": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,

    actuators={
        # 髋部 6 + 膝盖 2 + 腰 yaw 1，共 9 个关节。
        "hips_knees_and_waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_.*_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
            ],
            effort_limit_sim=120.0,
            velocity_limit_sim=11.936,
            stiffness=100.0,
            damping=3.0,
        ),

        # 左右踝 pitch。
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit_sim=36.0,
            velocity_limit_sim={
                "left_ankle_pitch_joint": 13.087,
                "right_ankle_pitch_joint": 13.088,
            },
            stiffness=40.0,
            damping=2.0,
        ),

        # 左右踝 roll。
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit_sim=24.0,
            velocity_limit_sim=15.077,
            stiffness=40.0,
            damping=2.0,
        ),

        # 腰 pitch 和 roll。
        "waist_pitch_roll": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_pitch_joint",
                "waist_roll_joint",
            ],
            effort_limit_sim=48.0,
            velocity_limit_sim=13.088,
            stiffness=80.0,
            damping=3.0,
        ),

        # 双侧肩 pitch 和 roll。
        "shoulder_pitch_roll": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            effort_limit_sim=36.0,
            velocity_limit_sim=13.088,
            stiffness=30.0,
            damping=1.0,
        ),

        # 双侧肩 yaw、肘、腕 yaw。
        "shoulder_yaw_elbows_and_wrist_yaw": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=24.0,
            velocity_limit_sim=15.077,
            stiffness=20.0,
            damping=1.0,
        ),

        # 双侧腕 pitch 和 roll。
        "wrist_pitch_roll": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=4.8,
            velocity_limit_sim=4.188,
            stiffness=5.0,
            damping=0.3,
        ),

        # 头部两个关节分别使用各自的限制。
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_yaw_joint",
                "head_pitch_joint",
            ],
            effort_limit_sim={
                "head_yaw_joint": 2.6,
                "head_pitch_joint": 0.6,
            },
            velocity_limit_sim={
                "head_yaw_joint": 6.019,
                "head_pitch_joint": 6.28,
            },
            stiffness=2.0,
            damping=0.2,
        ),
    },
)