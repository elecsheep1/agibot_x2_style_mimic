"""Diagnostic-only first-episode hip/knee target hold."""
import torch


class LegTargetHold:
    def __init__(self, env, after_seconds, clip_actions=None):
        self.env = env
        self.after_seconds = after_seconds
        self.clip_actions = clip_actions
        self.term = env.action_manager.get_term('joint_pos')
        robot = env.scene['robot']
        ids = self.term._joint_ids
        ids = list(range(robot.num_joints))[ids] if isinstance(ids, slice) else list(ids)
        self.names = ['left_hip_pitch_joint', 'right_hip_pitch_joint',
                      'left_knee_joint', 'right_knee_joint']
        self.columns = [ids.index(robot.joint_names.index(name)) for name in self.names]
        self.targets = torch.zeros_like(self.term.processed_actions)
        self.active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.first_done = self.active.clone()

    def apply(self, actions):
        if self.after_seconds is None:
            return actions
        elapsed = self.env.episode_length_buf * self.env.step_dt
        new = (~self.first_done) & (~self.active) & (elapsed + 1e-6 >= self.after_seconds)
        if bool(new.any()):
            self.targets[new] = self.term.processed_actions[new]
            print(f'[髋膝保持] 环境 {new.nonzero().flatten().tolist()} | '
                  f'目标 {self.targets[new][:, self.columns].tolist()} rad', flush=True)
        self.active |= new
        selected = self.active & (~self.first_done)
        if not bool(selected.any()):
            return actions
        scale = torch.as_tensor(self.term._scale, device=actions.device).expand_as(actions)
        offset = torch.as_tensor(self.term._offset, device=actions.device).expand_as(actions)
        if bool((scale[:, self.columns] == 0).any()):
            raise ValueError('Cannot invert zero action scale for target hold')
        raw = (self.targets[:, self.columns] - offset[:, self.columns]) / scale[:, self.columns]
        if self.clip_actions is not None and bool((raw[selected].abs() > self.clip_actions).any()):
            raise ValueError('Held targets exceed wrapper action clipping; cannot run exact control')
        result = actions.clone()
        rows = selected.nonzero().flatten()
        result[rows[:, None], self.columns] = raw[rows]
        return result

    def update(self, dones):
        self.first_done |= dones.bool().flatten()
        self.active &= ~self.first_done
