"""独立参考跟踪任务；不调用原起身奖励、辅助课程或 AMP。"""
import math
import torch
from isaaclab.managers import CommandTerm
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from x2_style_mimic.algorithms.supermimic_tracking.motion_library import MotionLibrary, X2Kinematics, TRACKED_BODIES
from x2_style_mimic.algorithms.supermimic_tracking.heads import AnalyticHeads, analytic_observation
from x2_style_mimic.algorithms.supermimic_tracking.wm_trainer import reference_at
from x2_style_mimic.algorithms.supermimic_tracking.rotations import (
    to_wxyz, to_xyzw, quat_apply, quat_apply_inverse, quat_mul, quat_from_rotvec,
)


def tensor(value):
    return value.torch if hasattr(value, 'torch') else value


class NoisyJointPositionAction(JointPositionAction):
    """观测与正则使用干净动作，世界模型监督使用实际执行的含噪动作。"""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.executed_raw = self.raw_actions.clone()
        self.last_executed = self.executed_raw.clone()

        robot = env.scene[cfg.asset_name]
        limits = tensor(
            robot.data.joint_pos_limits
        )[:, self._joint_ids]

        self.target_lower = limits[..., 0].clone()
        self.target_upper = limits[..., 1].clone()

        if self.cfg.clip is not None:
            self.target_lower = torch.maximum(
                self.target_lower,
                self._clip[..., 0],
            )
            self.target_upper = torch.minimum(
                self.target_upper,
                self._clip[..., 1],
            )

        if bool(
            (self.target_lower >= self.target_upper).any()
        ):
            raise ValueError("关节目标范围无效")

    # def process_actions(self, actions):
        # self._raw_actions.copy_(actions)
        # self.executed_raw = actions + self.cfg.noise_std * torch.randn_like(actions)
        # self.last_executed.copy_(self.executed_raw)
        # self._processed_actions = self.executed_raw * self._scale + self._offset
        # if self.cfg.clip is not None:
        #     self._processed_actions = self._processed_actions.clamp(self._clip[...,0],self._clip[...,1])
    def process_actions(self, actions):
        # 观测和动作变化正则仍然使用干净动作。
        self._raw_actions.copy_(actions)

        noisy_action = (
            actions
            + self.cfg.noise_std
            * torch.randn_like(actions)
        )

        requested_target = (
            noisy_action * self._scale
            + self._offset
        )

        # 噪声也不能把最终目标推出关节允许范围。
        self._processed_actions = torch.maximum(
            torch.minimum(
                requested_target,
                self.target_upper,
            ),
            self.target_lower,
        )

        # 世界模型必须记录实际执行目标对应的动作，
        # 不能继续记录裁剪前的 noisy_action。
        self.executed_raw = (
            self._processed_actions - self._offset
        ) / self._scale

        self.last_executed.copy_(
            self.executed_raw
        )


    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.executed_raw[env_ids] = 0


class TrackingCommand(CommandTerm):
    """目标在一次真实控制步中保持固定；步后才推进，避免奖励错一帧。"""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene['robot']
        names = list(self.robot.joint_names)
        self.control_ids = torch.tensor([names.index(n) for n in env.cfg.control_joint_names],device=self.device)
        from x2_style_mimic.assets.x2_ultra_cfg import X2_ULTRA_USD
        self.fk = X2Kinematics(env.cfg.urdf, names, self.control_ids, X2_ULTRA_USD).to(self.device)
        self.library = MotionLibrary(env.cfg.motion_manifest, names,
            tensor(self.robot.data.default_joint_pos)[0], self.fk, env.step_dt, cfg.bin_frames)
        self.heads = AnalyticHeads(self.fk, TRACKED_BODIES, 'torso_link','base_link',
            tensor(self.robot.data.soft_joint_pos_limits)[0].clone(), self.library.default_joint_pos, env.cfg.tracking_reward)
        self.frame = torch.zeros(env.num_envs,device=self.device,dtype=torch.long)
        self.episode_bin = self.frame.clone()
        self.g_max = self.heads.reward_cfg.max_step_reward/(1-cfg.gamma)
        self.bin_return = torch.full((self.library.num_bins,),self.g_max,device=self.device)
        self.bin_memory = torch.full_like(self.bin_return,1/self.library.num_bins)
        self.reward_sum = torch.zeros(env.num_envs,device=self.device)
        self.length = torch.zeros_like(self.reward_sum)
        horizon = int(math.ceil(env.cfg.episode_length_s/env.step_dt))+2
        self.episode_rewards = torch.zeros(env.num_envs,horizon,device=self.device)
        self.episode_bins = torch.zeros(env.num_envs,horizon,device=self.device,dtype=torch.long)
        self.gamma_powers = cfg.gamma**torch.arange(horizon,device=self.device,dtype=torch.float64)
        self.xi = torch.zeros(env.num_envs,35,device=self.device)
        self.xi[:,0] = 1
        self.previous_action = torch.zeros(env.num_envs,29,device=self.device)
        self._reward_stamp = -1
        self._terms = {}
        self._com_initialized = False
        self._foot_shapes = None

    @property
    def command(self):
        return self.library.goal_window(self.frame,self.cfg.future_frames)

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def compute(self, dt):
        # ManagerBased 在自动重置之后调用；推进工作由环境显式完成。
        pass

    def _randomize_physics(self, ids):
        import warp as wp
        robot = self.robot
        if not self._com_initialized:
            self._com_initialized = True
            if self._env.cfg.domain_randomization:
                body = robot.body_names.index('torso_link')
                coms = torch.cat((tensor(robot.data.body_com_pos_b),tensor(robot.data.body_com_quat_b)),-1).clone()
                extent = coms.new_tensor([.025,.05,.05])
                offsets = (torch.rand(self.num_envs,3,device=self.device)*2-1)*extent
                coms[:,body,:3] += offsets
                robot.set_coms_index(coms=coms[:,body:body+1],body_ids=[body])
                self.xi[:,-3:] = offsets
        if self._foot_shapes is None:
            offsets, foot_shapes = 0, []
            for name,path in zip(robot.body_names,robot.root_view.link_paths[0]):
                count = robot._physics_sim_view.create_rigid_body_view(path).max_shapes
                if name in ('left_ankle_roll_link','right_ankle_roll_link'):
                    foot_shapes.extend(range(offsets,offsets+count))
                offsets += count
            if offsets != robot.root_view.max_shapes or not foot_shapes:
                raise RuntimeError('无法确定 X2 足部碰撞形状，禁止写入未知摩擦系数')
            self._foot_shapes = foot_shapes
        friction = torch.empty(len(ids),device=self.device).uniform_(.3,1.2) if self._env.cfg.domain_randomization else torch.ones(len(ids),device=self.device)
        materials = wp.to_torch(robot.root_view.get_material_properties()).clone()
        cpu_ids = ids.to(device=materials.device,dtype=torch.long)
        cols = torch.tensor(self._foot_shapes,device=materials.device)
        materials[cpu_ids[:,None],cols[None,:],0] = friction.to(materials.device)[:,None]
        materials[cpu_ids[:,None],cols[None,:],1] = friction.to(materials.device)[:,None]
        materials[cpu_ids[:,None],cols[None,:],2] = 0
        robot.root_view.set_material_properties(wp.from_torch(materials,dtype=wp.float32),
            wp.from_torch(ids.to(device='cpu',dtype=torch.int32),dtype=wp.int32))
        self.xi[ids,0] = friction
        self.xi[ids,1:32] = ((torch.rand(len(ids),31,device=self.device)*2-1)*.01
                            if self._env.cfg.domain_randomization else 0)

    def log_rewards(self, ids):
        if not len(ids):
            return
        pos = self.length[ids].long()
        if bool((pos >= self.episode_rewards.shape[1]).any()):
            raise RuntimeError('回合奖励日志容量不足，禁止截断回报采样统计')
        rewards = self._env.reward_buf[ids]
        self.episode_rewards[ids,pos] = rewards
        self.episode_bins[ids,pos] = self.library.frame_bin[self.frame[ids]]
        self.reward_sum[ids] += rewards
        self.length[ids] += 1

    def _finalize_returns(self, ids):
        lens = self.length[ids].long()
        if not bool((lens>0).any()):
            return
        count = int(lens.max())
        mask = torch.arange(count,device=self.device)[None] < lens[:,None]
        rewards = self.episode_rewards[ids,:count].double()*mask
        powers = self.gamma_powers[:count]
        rtg = (rewards*powers).flip(1).cumsum(1).flip(1)/powers
        # 与复现项目未接 bootstrap_fn 的默认行为一致：时间截断尾段不参与采样回报。
        terminated = self._env.termination_manager.terminated[ids]
        ended = self.frame[ids]>=self.library.last_frame_of(self.frame[ids])
        truncated = ~terminated & ~ended
        remaining = lens[:,None]-torch.arange(count,device=self.device)[None]
        mask &= ~(truncated[:,None] & (remaining<self.cfg.min_remaining_steps))
        bins = self.episode_bins[ids,:count][mask]
        returns = rtg[mask].float()
        sums = torch.zeros_like(self.bin_return).scatter_add_(0,bins,returns)
        counts = torch.zeros_like(self.bin_return).scatter_add_(0,bins,torch.ones_like(returns))
        valid = counts>0
        self.bin_return[valid] = ((1-self.cfg.ema_alpha)*self.bin_return[valid]
                                 + self.cfg.ema_alpha*sums[valid]/counts[valid])

    def _resample_command(self, env_ids):
        ids = torch.arange(self.num_envs,device=self.device)[env_ids]
        if not len(ids):
            return
        self._finalize_returns(ids)
        hard = torch.softmax(-self.cfg.softmax_temp*self.bin_return/self.g_max,0)
        self.bin_memory.lerp_(hard,self.cfg.memory_alpha)
        probs = (self.cfg.uniform_ratio/self.library.num_bins + self.cfg.hard_ratio*hard
                 + self.cfg.memory_ratio*self.bin_memory/self.bin_memory.sum())
        bins = torch.multinomial(probs,len(ids),replacement=True)
        if self.cfg.sampling_mode == 'start':
            clips = torch.randint(len(self.library.starts),(len(ids),),device=self.device)
            frames = self.library.starts[clips]
            bins = self.library.frame_bin[frames]
        elif self.cfg.sampling_mode == 'uniform':
            frames = torch.randint(len(self.library.states),(len(ids),),device=self.device)
            bins = self.library.frame_bin[frames]
        elif self.cfg.sampling_mode == 'return':
            lo,hi = self.library.bin_starts[bins],self.library.bin_ends[bins]
            frames = lo + (torch.rand(len(ids),device=self.device)*(hi-lo+1)).long()
        else:
            raise ValueError('采样模式必须为 return、uniform 或 start')
        # 至少留两条参考转移，避免刚出生就因片段结束而退出。
        frames = torch.maximum(self.library.starts[self.library.frame_clip[frames]],
                               torch.minimum(frames,self.library.last_frame_of(frames)-2))
        bins = self.library.frame_bin[frames]
        if getattr(self, 'fixed_reset_frames', None) is not None:
            frames = self.fixed_reset_frames[ids]
            bins = self.library.frame_bin[frames]
        self.frame[ids],self.episode_bin[ids] = frames,bins
        self.reward_sum[ids],self.length[ids] = 0,0
        self.previous_action[ids] = 0
        self._randomize_physics(ids)
        state = self.library.states[frames].clone()
        p,q = state[:,:3],state[:,3:7]
        linear,angular = state[:,38:41],quat_apply(q,state[:,41:44])
        if self._env.cfg.reset_noise:
            p += (torch.rand_like(p)*2-1)*p.new_tensor([.05,.05,.01])
            angles = (torch.rand_like(p)*2-1)*p.new_tensor([.1,.1,.2])
            q = quat_mul(quat_from_rotvec(angles),q)
            state[:,7:38][:,self.control_ids] += (torch.rand(len(ids),29,device=self.device)*2-1)*.1
            linear += (torch.rand_like(linear)*2-1)*p.new_tensor([.5,.5,.2])
            angular += (torch.rand_like(angular)*2-1)*p.new_tensor([.52,.52,.78])
        limits = tensor(self.robot.data.joint_pos_limits)[ids]
        joints = state[:,7:38].clamp(limits[...,0],limits[...,1])
        pose = torch.cat((p+self._env.scene.env_origins[ids],to_xyzw(q)),-1)
        self.robot.write_root_pose_to_sim(pose,env_ids=ids)
        self.robot.write_root_velocity_to_sim(torch.cat((linear,angular),-1),env_ids=ids)
        self.robot.write_joint_state_to_sim(joints,state[:,44:75],env_ids=ids)
        targets = joints.clone()
        # 头部不由 Actor 控制，使用固定默认 PD 目标，避免引入未记录的重置目标。
        passive = torch.ones(31,device=self.device,dtype=torch.bool)
        passive[self.control_ids] = False
        targets[:,passive] = self.library.default_joint_pos[passive]
        self.robot.set_joint_position_target(targets,env_ids=ids)
        self._reward_stamp = -1

    def reward_terms(self):
        stamp = self._env.common_step_counter
        if self._reward_stamp != stamp:
            action = self._env.action_manager.get_term('joint_pos').raw_actions
            # 真实奖励读取真实刚体数据，不用预测状态冒充仿真测量。
            robot = self.robot
            body_ids = [robot.body_names.index(n) for n in TRACKED_BODIES]
            pose = tensor(robot.data.body_link_pose_w)[:,body_ids]
            velocity = tensor(robot.data.body_link_vel_w)[:,body_ids]
            body = dict(pos=pose[...,:3]-self._env.scene.env_origins[:,None,:],quat=to_wxyz(pose[...,3:]),
                        lin_vel=velocity[...,:3],ang_vel=velocity[...,3:])
            self._terms = self.heads.reward(self._env.read_tracking_state(),action,self.previous_action,
                                           reference_at(self.library,self.frame),body)
            self._reward_stamp = stamp
        return self._terms


def policy_observation(env):
    cmd = env.command_manager.get_term('motion')
    action = env.action_manager.get_term('joint_pos').raw_actions
    obs = analytic_observation(env.read_tracking_state(),action,cmd.xi,cmd.library.default_joint_pos,cmd.control_ids)
    if env.cfg.obs_noise:
        half = obs.new_tensor([.01]*29+[.5]*29+[.2]*3+[.05]*3+[0.]*29)
        obs = obs + (torch.rand_like(obs)*2-1)*half
    return obs


def goal_observation(env):
    return env.command_manager.get_term('motion').command


def reward_term(env, name):
    # 共享函数已经乘 dt；RewardManager 会再乘 dt，因此先除回去。
    return env.command_manager.get_term('motion').reward_terms()[name] / env.step_dt


def tracking_failure(env, component):
    cmd = env.command_manager.get_term('motion')
    s = env.read_tracking_state()
    ref = reference_at(cmd.library,cmd.frame)
    features = cmd.heads.termination_features(s[:,:3],s[:,3:7],ref)
    return features[:,component].abs() > (env.cfg.root_height_threshold if component==0 else env.cfg.root_tilt_threshold)


def clip_end(env):
    cmd = env.command_manager.get_term('motion')
    return cmd.frame >= cmd.library.last_frame_of(cmd.frame)


def observation_part(env, start, end):
    # 每项只返回自身部分；Manager 的分项历史布局与想象侧一致。
    return policy_observation(env)[:,start:end]
