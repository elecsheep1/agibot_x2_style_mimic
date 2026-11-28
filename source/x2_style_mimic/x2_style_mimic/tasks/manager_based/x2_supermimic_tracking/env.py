"""直接继承 ManagerBasedRLEnv，捕获自动重置前的真实终止状态。"""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from .mdp import tensor
from x2_style_mimic.algorithms.supermimic_tracking.rotations import to_wxyz, quat_apply_inverse


class X2SuperMimicTrackingEnv(ManagerBasedRLEnv):
    @torch.no_grad()
    def read_tracking_state(self):
        robot = self.scene['robot']
        pose = tensor(robot.data.root_link_pose_w)
        velocity = tensor(robot.data.root_link_vel_w)
        q = to_wxyz(pose[:,3:7])
        return torch.cat((pose[:,:3]-self.scene.env_origins,q,tensor(robot.data.joint_pos),
                          velocity[:,:3],quat_apply_inverse(q,velocity[:,3:]),tensor(robot.data.joint_vel)),-1).clone()

    @torch.no_grad()
    def step(self, action):
        if action.shape != (self.num_envs,29) or not torch.isfinite(action).all():
            raise ValueError('跟踪动作必须为有限的 [N,29] 张量')
        command = self.command_manager.get_term('motion')
        before = self.read_tracking_state()
        frame,xi = command.frame.clone(),command.xi.clone()
        previous = self.action_manager.get_term('joint_pos').raw_actions.clone()
        self._terminal_state = torch.empty_like(before)
        self._terminal_mask = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._capture_tracking = True
        # 在调用 process_actions 前保存上一条干净动作，避免正则读到本步动作。
        command.previous_action.copy_(previous)
        try:
            obs,reward,terminated,truncated,extras = super().step(action)
        finally:
            self._capture_tracking = False
        done = terminated | truncated
        if not torch.equal(done,self._terminal_mask):
            raise RuntimeError('跟踪转移的终止状态与自动重置不一致')
        next_state = self.read_tracking_state()
        next_state[done] = self._terminal_state[done]
        # 自动重置会清空 executed_raw；实际执行动作由 action term 在 process 时留存。
        executed = self.action_manager.get_term('joint_pos').last_executed.clone()
        extras['tracking_transition'] = dict(state=before,next_state=next_state,action=action.clone(),
            executed=executed,frame=frame,context=xi,reward=reward.clone(),terminated=terminated.clone(),truncated=truncated.clone())
        alive = ~done
        command.log_rewards(alive.nonzero().flatten())
        command.frame[alive] = torch.minimum(frame[alive]+1,command.library.last_frame_of(frame[alive]))
        # 步后目标是下一控制步的目标；本体历史已由 Manager 正常更新一次。
        obs['goal'] = command.command
        return obs,reward,terminated,truncated,extras

    def _reset_idx(self, env_ids):
        if getattr(self,'_capture_tracking',False):
            self._terminal_state[env_ids] = self.read_tracking_state()[env_ids]
            self._terminal_mask[env_ids] = True
            command = self.command_manager.get_term('motion')
            command.log_rewards(env_ids)
        super()._reset_idx(env_ids)
