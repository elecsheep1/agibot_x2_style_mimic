import torch

from isaaclab.envs import ManagerBasedRLEnv

from x2_style_mimic.algorithms.supermimic.state import (
    StateSpec,
    pack_state,
)

class X2SuperMimicEnv(ManagerBasedRLEnv):
    """只负责真实仿真和完整转移采集，不负责算法更新。"""

    @torch.no_grad()
    def read_world_state(self):
        """读取当前世界状态，返回 [B, D] 的状态向量。"""
        robot = self.scene["robot"]
        action_term = self.action_manager.get_term("joint_pos")

        root_pose = robot.data.root_link_pose_w.torch.clone()
        root_pose[:, :3] -= self.scene.env_origins
        root_velocity = robot.data.root_link_vel_w.torch

        # 与当前 LimitedJointPositionAction 的内部状态对应。
        # previous