"""独立模块初始化后的短程联合训练；按优化更新计数，固定首回合真实评估。"""
import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import tomllib
from zoneinfo import ZoneInfo
import torch
from diagnose_tracking import PROJECT, load_file

HERE = Path(__file__).resolve().parent


@dataclass
class ProbeCfg:
    base_checkpoint: str = ''
    dynamics_checkpoint: str = ''
    fixed_data: str = ''
    updates: int = 500
    evaluation_interval: int = 100
    evaluation_seconds: float = 20.
    collection_warmup: int = 64
    run_name: str = 'joint_probe_wm8_actor32'
    freeze_failure_head: bool = False

    def validate(self):
        if min(self.updates, self.evaluation_interval, self.collection_warmup) < 1:
            raise ValueError('联合更新次数、评估间隔和预采集步数必须为正数')
        if not math.isfinite(self.evaluation_seconds) or self.evaluation_seconds <= 0:
            raise ValueError('评估秒数必须为正有限数')
        if not isinstance(self.freeze_failure_head,bool):
            raise ValueError('freeze_failure_head 必须是布尔值')


def restore_modules(training, base, dynamics):
    """只迁移参数，不伪装成完整续训；失败头来自基础检查点。"""
    if base.get('kind') != 'x2_reproduced_tracking_v1':
        raise ValueError('基础检查点类型不匹配')
    if dynamics.get('kind') != 'fixed_dynamics_refit_v1':
        raise ValueError('动力学诊断产物类型不匹配')
    if base['contract'] != training.contract:
        raise ValueError('机器人、动作集或奖励合同不匹配')
    for name in ('policy', 'critic'):
        getattr(training, name).load_state_dict(base[name], strict=True)
    training.world.load_state_dict(dynamics['world'], strict=True)
    head = {k.removeprefix('term_head.'): v for k, v in base['world'].items()
            if k.startswith('term_head.')}
    training.world.term_head.load_state_dict(head, strict=True)
    # 本入口构造了全新优化器，不加载任何来源优化器状态。
    for optimizer in (training.world_trainer.opt, training.policy_trainer.opt,
                      training.policy_trainer.critic_opt):
        if optimizer.state:
            raise RuntimeError('初始化阶段优化器不应已有训练状态')


def evaluation_frames(library, count, seed):
    """一半首帧覆盖，另一半均匀参考帧；不随训练采样课程变化。"""
    first = (count + 1) // 2
    clips = torch.arange(first) % len(library.starts)
    starts = library.starts.cpu()[clips]
    random_frames = torch.randint(len(library.states), (count-first,),
                                 generator=torch.Generator().manual_seed(seed))
    frames = torch.cat((starts, random_frames)).to(library.starts.device)
    return torch.minimum(frames, library.last_frame_of(frames)-2)


def configure_failure_head(training, freeze):
    """只冻结参数，不切断失败概率对想象状态的梯度。"""
    training.world.term_head.requires_grad_(not freeze)
    return {key:value.detach().cpu().clone() for key,value in training.world.term_head.state_dict().items()}


def assert_head_unchanged(training, initial):
    if not all(torch.equal(value,training.world.term_head.state_dict()[key].detach().cpu())
               for key,value in initial.items()):
        raise RuntimeError('冻结对照中失败头参数发生变化')


@torch.no_grad()
def evaluate_real(training, frames, seconds):
    """评估数据不入池；退出时清掉评估回报，调用者必须重新开始训练回合。"""
    import numpy as np
    env, command = training.env, training.command
    action_term = env.action_manager.get_term('joint_pos')
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    old_frames = getattr(command, 'fixed_reset_frames', None)
    bins = (command.bin_return.clone(), command.bin_memory.clone())
    old_noise = action_term.cfg.noise_std
    was_training = training.policy.training
    try:
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        command.fixed_reset_frames = frames
        command.length.zero_()  # 手动重置不把被截断的训练回报计入课程。
        action_term.cfg.noise_std = 0.
        training.policy.eval()
        obs, _ = env.reset()
        active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        rewards = torch.zeros(env.num_envs, device=env.device)
        steps = torch.zeros_like(rewards)
        failed = torch.zeros_like(active)
        at_bound = torch.zeros_like(rewards)
        lo = torch.as_tensor(training.cfg.policy.action_lower, device=env.device)
        hi = torch.as_tensor(training.cfg.policy.action_upper, device=env.device)
        for _ in range(math.ceil(seconds/env.step_dt)):
            action = training.policy(obs['policy'], obs['goal'])
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards[active] += reward[active];steps[active] += 1
            near = ((action-lo < .01*(hi-lo)) | (hi-action < .01*(hi-lo))).float().mean(-1)
            at_bound[active] += near[active]
            failed |= active & terminated
            active &= ~(terminated | truncated)
            if not active.any():break
        normalized = rewards/(steps.clamp_min(1)*command.heads.reward_cfg.max_step_reward)
        return dict(normalized_return=float(normalized.mean()),failure_rate=float(failed.float().mean()),
                    mean_seconds=float(steps.mean()*env.step_dt),censored_rate=float(active.float().mean()),
                    action_near_bound_fraction=float((at_bound/steps.clamp_min(1)).mean()),
                    episodes=[dict(frame=int(frames[i]),normalized_return=float(normalized[i]),
                                   seconds=float(steps[i]*env.step_dt),failed=bool(failed[i]),
                                   evaluation_limit_reached=bool(active[i])) for i in range(env.num_envs)])
    finally:
        command.fixed_reset_frames = old_frames
        command.length.zero_()
        command.bin_return.copy_(bins[0]);command.bin_memory.copy_(bins[1])
        action_term.cfg.noise_std = old_noise
        training.policy.train(was_training)
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        if rng[3]:torch.cuda.set_rng_state_all(rng[3])


def run_probe(training, probe, data):
    from refit_fixed_dynamics import split_data
    from audit_model_errors import sample_cohort, variant_rollout, component_metrics, validate_data
    from x2_style_mimic.algorithms.supermimic_tracking.rotations import quat_box_minus
    from x2_style_mimic.algorithms.supermimic_tracking.training import validate_fk
    env = training.env
    initial_head = configure_failure_head(training,probe.freeze_failure_head)
    groups, _ = split_data(data, training.cfg.seed)
    validate_data(data)
    cohort = {k: v.to(env.device) if isinstance(v, torch.Tensor) else v
              for k,v in sample_cohort(groups['validation'],128,training.cfg.seed).items()}
    frames = evaluation_frames(training.lib, env.num_envs, 42)
    report = dict(evaluation_frames=frames.tolist(),evaluations=[],freeze_failure_head=probe.freeze_failure_head,notes=[
        '固定真实评估仅统计每环境首回合；首帧与均匀参考帧各半，不等同纯首帧验收。',
        '真实评估不入回放库；评估后显式切断回合并重新采集。',
        '固定旧数据用于模型回归检查，不代表新策略分布的留出集。',
        '学习率保持加载配置值；本次短程诊断不启用余弦衰减。',
        '按完整联合更新次数停止，不按向量环境步停止。'])
    ids = torch.arange(env.num_envs, device=env.device)
    def checkpoint_evaluation():
        if probe.freeze_failure_head:
            assert_head_unchanged(training,initial_head)
        training.buffer.end_episode(ids)
        real = evaluate_real(training,frames,probe.evaluation_seconds)
        with torch.no_grad():
            pred = variant_rollout(training.world,cohort,'baseline')
            fixed = {str(h):component_metrics(pred[:,h-1],cohort['following'][:,h-1],quat_box_minus)
                     for h in (1,8,16,32)}
            mask = training.buffer.window_mask(32)
            online = None
            if mask.any():
                batch = training.buffer.sample_windows(128,32)
                online = training.world.rollout_error(batch.states,batch.a_exec,batch.xi).tolist()
        record = dict(update=training.num_updates,real=real,fixed_world=fixed,replay_world_nse=online,
                      frozen_head_unchanged=True if probe.freeze_failure_head else None)
        report['evaluations'].append(record)
        (training.output/'probe_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        for key,value in real.items():
            if isinstance(value,(float,int)):training.writer.add_scalar('real_eval/'+key,value,training.num_updates)
        training.save(f'joint_{training.num_updates:04d}.pt')
        print(f'[真实评估 {training.num_updates}] 回报={real["normalized_return"]:.4f}，失败率={real["failure_rate"]:.1%}，'
              f'平均时长={real["mean_seconds"]:.2f}s，动作贴边={real["action_near_bound_fraction"]:.1%}',flush=True)
        # evaluate_real 留下的是评估状态；不可把下一次训练 reset 拼到旧回放。
        training.buffer.end_episode(ids)
        obs,_ = env.reset();validate_fk(env)
        return obs,torch.ones(env.num_envs,device=env.device,dtype=torch.bool)
    obs, starts = checkpoint_evaluation()
    credit = 0.;warmup = 0;last_time=time.monotonic()
    while training.num_updates < probe.updates:
        with torch.no_grad():
            action=training.policy(obs['policy'],obs['goal'])
            obs,_,_,_,extras=env.step(action)
            starts=training.record(extras['tracking_transition'],starts)
        training.iteration+=1;warmup+=1
        if warmup < probe.collection_warmup:continue
        if not training.buffer.window_mask(training.cfg.wm_trainer.window).any():continue
        credit+=training.cfg.updates_per_env_step
        while credit>=1 and training.num_updates<probe.updates:
            credit-=1
            metrics={'world/'+k:v for k,v in training.world_trainer.update(training.buffer).items()}
            metrics.update({'policy/'+k:v for k,v in training.policy_trainer.update(training.buffer).items()})
            if not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError('联合训练指标出现非有限值')
            training.num_updates+=1
            for key,value in metrics.items():training.writer.add_scalar(key,value,training.num_updates)
            if training.num_updates==1 or training.num_updates%25==0:
                print(f'[联合更新 {training.num_updates}/{probe.updates}] 模型损失={metrics["world/loss"]:.4f}，'
                      f'想象回报={metrics["policy/imag_return_undisc"]:.4f}，动作均值={metrics["policy/action_abs_mean"]:.3f}，'
                      f'耗时={time.monotonic()-last_time:.1f}s',flush=True)
            if training.num_updates%probe.evaluation_interval==0 or training.num_updates==probe.updates:
                obs,starts=checkpoint_evaluation()
                warmup=0;credit=0.
                break
    training.writer.flush()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'configs/joint_probe.toml')
    parser.add_argument('--run-name')
    parser.add_argument('--updates',type=int,help='仅用于短程冒烟验证')
    parser.add_argument('--urdf',type=Path,default=Path.home()/'robotics_ws/agibot_model/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra.urdf')
    from isaaclab.app import add_launcher_args,launch_simulation
    add_launcher_args(parser);args=parser.parse_args()
    entry=load_file('_joint_config',PROJECT/'scripts/supermimic_tracking/train.py')
    compat=load_file('_joint_compat',HERE.parent/'train_supermimic_tracking.py')
    from x2_style_mimic.algorithms.supermimic_tracking.config import TrackingTrainCfg
    settings=tomllib.loads(args.config.read_text());probe=ProbeCfg()
    entry.update_config(probe,settings.pop('probe'))
    if args.updates is not None:probe.updates=args.updates
    if args.run_name:probe.run_name=args.run_name
    probe.validate()
    sources={key:(PROJECT/getattr(probe,key)).resolve() for key in ('base_checkpoint','dynamics_checkpoint','fixed_data')}
    base=torch.load(sources['base_checkpoint'],map_location='cpu',weights_only=False)
    dynamics=torch.load(sources['dynamics_checkpoint'],map_location='cpu',weights_only=False)
    cfg=TrackingTrainCfg();entry.update_config(cfg,base['config'])
    entry.update_config(cfg.world_model,dynamics['world_config']);entry.update_config(cfg,settings)
    cfg.resume=None;cfg.run_name=probe.run_name;cfg.device=args.device or cfg.device
    cfg.iterations=math.ceil(probe.updates/cfg.updates_per_env_step)+6*probe.collection_warmup
    if cfg.domain_randomization or cfg.reset_noise or cfg.obs_noise:
        raise ValueError('此诊断先关闭域随机化、重置噪声及观测噪声')
    if probe.collection_warmup<=max(cfg.wm_trainer.window,cfg.policy_trainer.history):
        raise ValueError('预采集步数不足以建立监督窗口和策略历史')
    cfg.validate()
    from x2_style_mimic.tasks.manager_based.x2_supermimic_tracking.env_cfg import X2SuperMimicTrackingEnvCfg
    env_cfg=X2SuperMimicTrackingEnvCfg()
    env_cfg.scene.num_envs,env_cfg.seed,env_cfg.sim.device=cfg.num_envs,cfg.seed,cfg.device
    env_cfg.domain_randomization=env_cfg.reset_noise=env_cfg.obs_noise=False
    env_cfg.actions.joint_pos.noise_std=cfg.action_noise_std
    env_cfg.observations.policy.history_length=cfg.policy.history
    env_cfg.commands.motion.future_frames=cfg.policy_trainer.future_frames
    env_cfg.root_height_threshold=cfg.world_model.height_threshold
    env_cfg.root_tilt_threshold=cfg.world_model.tilt_threshold
    env_cfg.urdf=str(args.urdf.resolve());compat._replace_null_manager_configs(env_cfg)
    output=PROJECT/'logs/supermimic_tracking'/f'{datetime.now(ZoneInfo("Asia/Shanghai")):%Y-%m-%d_%H-%M-%S}_{cfg.run_name}'
    sys.argv=[sys.argv[0]]
    with launch_simulation(env_cfg,args):
        import numpy as np
        from x2_style_mimic.tasks.manager_based.x2_supermimic_tracking.env import X2SuperMimicTrackingEnv
        from x2_style_mimic.algorithms.supermimic_tracking.training import TrackingTraining
        random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
        env=X2SuperMimicTrackingEnv(env_cfg);training=None
        try:
            training=TrackingTraining(env,cfg,output)
            restore_modules(training,base,dynamics)
            provenance=dict(probe=asdict(probe),sources={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in sources.items()},
                            optimizers='重新初始化；未加载来源优化器状态',failure_head='来自基础检查点')
            (output/'initialization.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
            data=torch.load(sources['fixed_data'],map_location='cpu',weights_only=False)['data']
            print(f'[输出目录] {output}',flush=True)
            run_probe(training,probe,data)
            print(f'[完成] {output}',flush=True)
        finally:
            if training is not None:training.writer.close()
            env.close()


if __name__=='__main__':main()
