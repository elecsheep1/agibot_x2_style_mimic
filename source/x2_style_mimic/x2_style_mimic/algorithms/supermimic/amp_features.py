from torch import nn

from .state import unpack_state
from .kinematics import DifferentiableKinematics
from ...tasks.manager_based.x2_style_mimic.mdp.amp import (
    AMP_KEY_BODY_NAMES,
    build_amp_state,
)


class DifferentiableAMPFeatures(nn.Module):
    """可微 AMP 特征模块,用于计算想象轨迹的 AMP 特征。"""

    def __init__(self, spec, kinematics):
        super().__init__()
        self.spec = spec
        self.kinematics = kinematics

    @classmethod  # 把函数声明为类方法
    def from_urdf(
        cls,
        spec,
        urdf_path,
        joint_names,
        usd_path=None,
    ):
        if len(joint_names) != spec.num_joints:
            raise ValueError("joint_names 必须对应 StateSpec 的全部关节")
        kinematics = DifferentiableKinematics(
            urdf_path=urdf_path,
            joint_names=joint_names,
            body_names=AMP_KEY_BODY_NAMES,
        )
        if usd_path is not None:
            kinematics.align_joint_origins_from_usd(
                usd_path
            )
        return cls(
            spec=spec,
            kinematics=kinematics,
        )

    def forward(self, states):
        fields = unpack_state(states, self.spec)

        key_pose_w, key_velocity_w = self.kinematics(
            root_pose=fields["root_pose"],
            root_velocity=fields["root_velocity"],
            joint_position=fields["joint_position"],
            joint_velocity=fields["joint_velocity"],
        )

        return build_amp_state(
            root_pos_w=fields["root_pose"][:, :3],
            root_quat_xyzw=fields["root_pose"][:, 3:7],
            root_lin_vel_w=fields["root_velocity"][:, :3],
            root_ang_vel_w=fields["root_velocity"][:, 3:6],
            key_pose_w=key_pose_w,
            key_vel_w=key_velocity_w,
            ground_height=0.0,
        )
