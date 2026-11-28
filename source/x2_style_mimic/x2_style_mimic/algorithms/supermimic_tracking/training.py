"""采样 → 世界模型监督 → 想象 BPTT → λ-return Critic；不使用 PPO/AMP。"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from .world_model import WorldModel
from .wm_trainer import WorldModelTrainer
from .nets import Policy, Critic
from .policy_trainer import PolicyTrainer
from .replay_buffer import ReplayBuffer

SCHEMA = 'x2_reproduced_tracking_v1'


def asset_contract(env):
    command = env.command_manager.get_term('motion')
    from ...assets.x2_ultra_cfg import X2_ULTRA_USD
    files = [str(X2_ULTRA_USD),str(env.cfg.urdf),*command.library.paths]
    return dict(joints=list(command.robot.joint_names),controls=list(env.cfg.control_joint_names),
        step_dt=env.step_dt,physics_dt=env.cfg.sim.dt,
        files={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files},
        reward=asdict(env.cfg.tracking_reward))


class TrackingTraining:
    def __init__(self, env, cfg, output):
        cfg.validate()
        self.env,self.cfg,self.output = env,cfg,Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.command = env.command_manager.get_term('motion')
        self.lib = self.command.library
        cfg.policy_trainer.obs_noise = cfg.obs_noise
        if abs(env.step_dt-cfg.world_model.state_spec.dt) > 1e-8:
            raise ValueError('世界模型积分周期与真实控制周期不一致')
        self.world = WorldModel(cfg.world_model,self.command.heads,env.device)
        self.policy = Policy(cfg.policy).to(env.device)
        self.critic = Critic(cfg.critic,pcfg=cfg.policy).to(env.device)
        self.world_trainer = WorldModelTrainer(self.world,self.lib,cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(self.world,self.policy,self.critic,self.lib,cfg.policy_trainer)
        self.buffer = ReplayBuffer(env.num_envs,cfg.replay_capacity,75,29,35,env.device)
        self.iteration,self.num_updates = 0,0
        self.contract = asset_contract(env)
        if cfg.resume:
            self.load(cfg.resume)
        self.writer = SummaryWriter(str(self.output/'tb'))
        (self.output/'config.json').write_text(json.dumps(asdict(cfg),indent=2,ensure_ascii=False))
        (self.output/'contract.json').write_text(json.dumps(self.contract,indent=2,ensure_ascii=False))

    def save(self, name='latest.pt'):
        payload = dict(kind=SCHEMA,iteration=self.iteration,num_updates=self.num_updates,
            config=asdict(self.cfg),contract=self.contract,
            world=self.world.state_dict(),policy=self.policy.state_dict(),critic=self.critic.state_dict(),
            world_optimizer=self.world_trainer.opt.state_dict(),actor_optimizer=self.policy_trainer.opt.state_dict(),
            critic_optimizer=self.policy_trainer.critic_opt.state_dict(),
            bin_return=self.command.bin_return,bin_memory=self.command.bin_memory,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        target = self.output/name
        temporary = target.with_suffix('.tmp')
        torch.save(payload,temporary)
        temporary.replace(target)

    def load(self, path):
        ck = torch.load(path,map_location=self.env.device,weights_only=False)
        if ck.get('kind') != SCHEMA or ck.get('contract') != self.contract:
            raise ValueError('检查点任务或机器人／参考动作／奖励合同不一致，不能直接续训')
        # 允许改变总训练步数与日志频率，不允许悄悄换网络、优化器或采样规则。
        old,new = ck['config'].copy(),asdict(self.cfg)
        for key in ('iterations','log_interval','checkpoint_interval','run_name','resume','device'):
            old.pop(key,None);new.pop(key,None)
        if old != new:
            raise ValueError('恢复配置与检查点不一致；需要从头训练或使用原配置')
        for name in ('world','policy','critic'):
            getattr(self,name).load_state_dict(ck[name],strict=True)
        for optimizer,key in ((self.world_trainer.opt,'world_optimizer'),(self.policy_trainer.opt,'actor_optimizer'),
                               (self.policy_trainer.critic_opt,'critic_optimizer')):
            optimizer.load_state_dict(ck[key])
        self.command.bin_return.copy_(ck['bin_return'])
        self.command.bin_memory.copy_(ck['bin_memory'])
        self.iteration,self.num_updates = ck['iteration'],ck['num_updates']
        torch.set_rng_state(ck['rng'].cpu())
        if ck.get('cuda_rng') and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])

    @torch.no_grad()
    def record(self, transition, episode_start):
        t = transition
        ids = torch.arange(self.env.num_envs,device=self.env.device)
        zeros = torch.zeros_like(t['terminated'])
        self.buffer.add(ids,t['state'],t['action'],t['executed'],t['context'],t['reward'],t['terminated'].float(),
                        t['frame'],self.lib.frame_clip[t['frame']],zeros,episode_start)
        done = t['terminated'] | t['truncated']
        end = ids[done]
        if len(end):
            self.buffer.add(end,t['next_state'][end],torch.zeros_like(t['action'][end]),torch.zeros_like(t['action'][end]),
                t['context'][end],torch.zeros_like(t['reward'][end]),t['terminated'][end].float(),t['frame'][end],
                self.lib.frame_clip[t['frame'][end]],torch.ones_like(done[end]),torch.zeros_like(done[end]))
            self.buffer.end_episode(end)
        return done

    def run(self):
        env,cfg = self.env,self.cfg
        obs,_ = env.reset()
        validate_fk(env)
        start_iteration = self.iteration
        start_updates = self.num_updates
        start_time,last_time = time.monotonic(),time.monotonic()
        episode_start = torch.ones(env.num_envs,device=env.device,dtype=torch.bool)
        returns = torch.zeros(env.num_envs,device=env.device)
        lengths = returns.clone()
        collected_returns,collected_lengths = [],[]
        accumulated,number,utd = {},0,0.
        print(f'[输出目录] {self.output}',flush=True)
        print(f'[独立跟踪] {env.num_envs} 环境；世界模型 {cfg.wm_trainer.window} 条转移；'
              f'策略想象 {cfg.policy_trainer.horizon} 步；真实参考目标条件；AMP/PPO 均未接入',flush=True)
        try:
            while self.iteration < cfg.iterations:
                with torch.no_grad():
                    action = self.policy(obs['policy'],obs['goal'])
                    obs,reward,terminated,truncated,extras = env.step(action)
                    episode_start = self.record(extras['tracking_transition'],episode_start)
                    returns += reward
                    lengths += 1
                    if episode_start.any():
                        collected_returns.extend((returns[episode_start]/(lengths[episode_start]*self.command.heads.reward_cfg.max_step_reward)).tolist())
                        collected_lengths.extend(lengths[episode_start].tolist())
                        returns[episode_start],lengths[episode_start] = 0,0
                self.iteration += 1
                if self.iteration-start_iteration >= cfg.warmup_steps:
                    if self.buffer.window_mask(cfg.wm_trainer.window).any():
                        utd += cfg.updates_per_env_step
                        while utd >= 1:
                            utd -= 1
                            progress = min(self.iteration/(cfg.iterations*cfg.lr_decay_fraction),1.)
                            ratio = cfg.lr_final_ratio+(1-cfg.lr_final_ratio)*.5*(1+math.cos(math.pi*progress))
                            self.world_trainer.set_lr(cfg.wm_trainer.lr*ratio)
                            self.policy_trainer.set_lr(cfg.policy_trainer.lr*ratio)
                            metrics = {f'world/{k}':v for k,v in self.world_trainer.update(self.buffer).items()}
                            metrics.update({f'policy/{k}':v for k,v in self.policy_trainer.update(self.buffer).items()})
                            if not all(math.isfinite(v) for v in metrics.values()):
                                raise FloatingPointError('训练指标出现非有限值，停止当前更新循环')
                            self.num_updates += 1
                            number += 1
                            for k,v in metrics.items(): accumulated[k] = accumulated.get(k,0)+v
                if self.iteration % cfg.log_interval == 0:
                    now = time.monotonic()
                    metrics = {k:v/max(number,1) for k,v in accumulated.items()}
                    metrics.update({'time/elapsed_s':now-start_time,'train/updates':self.num_updates,
                                    'time/env_steps_per_s':cfg.log_interval*env.num_envs/(now-last_time)})
                    if collected_returns:
                        metrics['train/normalized_return'] = sum(collected_returns)/len(collected_returns)
                        metrics['train/episode_steps'] = sum(collected_lengths)/len(collected_lengths)
                    for k,v in metrics.items(): self.writer.add_scalar(k,v,self.iteration)
                    with (self.output/'metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(iteration=self.iteration,**metrics))+'\n')
                    print('#'*80,flush=True)
                    print(f' Learning iteration {self.iteration}/{cfg.iterations} | updates {self.num_updates}',flush=True)
                    for key in ('train/normalized_return','world/loss','world/term_bce','policy/imag_return_undisc',
                                'policy/imag_reward_mean','policy/imag_survival_H','policy/action_abs_mean','policy/critic_loss'):
                        if key in metrics: print(f'{key:>43}: {metrics[key]:.6f}',flush=True)
                    accumulated,number,collected_returns,collected_lengths = {},0,[],[]
                    last_time = now
                if self.iteration % cfg.checkpoint_interval == 0:
                    self.save(f'model_{self.iteration}.pt')
                    self.save()
            if self.num_updates == start_updates and self.iteration > start_iteration:
                raise RuntimeError("没有完成任何模型／策略更新，请检查回放连续窗口与预热设置")
            self.save()
        finally:
            self.writer.close()


@torch.no_grad()
def validate_fk(env, tolerance=0.003):
    """实测刚体原点与 USD 校准 FK 必须一致，禁止静默使用错误模型。"""
    from ...utils.rotation import quaternion_angle_error
    from .motion_library import TRACKED_BODIES
    from .rotations import to_xyzw
    command = env.command_manager.get_term('motion')
    predicted = command.fk(env.read_tracking_state())
    ids = [command.robot.body_names.index(n) for n in TRACKED_BODIES]
    value = command.robot.data.body_link_pose_w
    actual = value.torch if hasattr(value,'torch') else value
    actual = actual[:,ids]
    error = (predicted['pos']-(actual[...,:3]-env.scene.env_origins[:,None])).norm(dim=-1).max().item()
    angle = quaternion_angle_error(to_xyzw(predicted['quat']),actual[...,3:]).max().item()
    if error > tolerance or angle > .02:
        raise RuntimeError(f'跟踪 FK 与真实资产不一致：位置 {error:.6f} m，角度 {angle:.6f} rad')
    print(f'[FK 核对] 最大位置差 {error:.6f} m，最大旋转差 {angle:.6f} rad',flush=True)


@torch.no_grad()
def evaluate(env,cfg,checkpoint,output):
    """所有参考逐条从首帧完整评估；只统计首回合，不混入自动重置。"""
    ck = torch.load(checkpoint,map_location=env.device,weights_only=False)
    if ck.get('kind') != SCHEMA or ck['contract'] != asset_contract(env):
        raise ValueError('验收检查点与资产／参考动作／奖励不一致')
    policy = Policy(cfg.policy).to(env.device)
    policy.load_state_dict(ck['policy'],strict=True)
    policy.eval()
    command = env.command_manager.get_term('motion')
    records = []
    for offset in range(0,len(command.library.starts),env.num_envs):
        count = min(env.num_envs,len(command.library.starts)-offset)
        clips = torch.arange(offset,offset+env.num_envs,device=env.device).clamp_max(len(command.library.starts)-1)
        command.fixed_reset_frames = command.library.starts[clips]
        obs,_ = env.reset()
        validate_fk(env)
        active = torch.arange(env.num_envs,device=env.device) < count
        returns = torch.zeros(env.num_envs,device=env.device)
        steps = returns.clone()
        failed = torch.zeros_like(active)
        for _ in range(env.max_episode_length):
            obs,reward,terminated,truncated,_ = env.step(policy(obs['policy'],obs['goal']))
            returns[active] += reward[active]
            steps[active] += 1
            failed |= active & terminated
            active &= ~(terminated|truncated)
            if not active.any(): break
        for i in range(count):
            records.append(dict(motion=command.library.paths[offset+i],steps=int(steps[i]),
                normalized_return=float(returns[i]/(steps[i].clamp_min(1)*command.heads.reward_cfg.max_step_reward)),
                failed=bool(failed[i]),evaluation_limit_reached=bool(active[i])))
    target = Path(output)/'evaluation.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(records,indent=2,ensure_ascii=False))
    print(json.dumps(records,indent=2,ensure_ascii=False),flush=True)
    return records
