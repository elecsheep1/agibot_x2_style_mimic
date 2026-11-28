"""冻结模型的动作序列梯度方向检查；配对重置、重复回放和正反方向小扰动。"""
import argparse
from pathlib import Path
import json
import math
import sys
from types import SimpleNamespace
import torch
from diagnose_tracking import PROJECT,load_file
from joint_probe import evaluation_frames


def action_direction(gradient):
    # [H,N,A]，各环境独立归一化，并限制个别动作的极端方向分量。
    rms=gradient.square().mean((0,2),keepdim=True).sqrt()
    return (gradient/rms.clamp_min(1e-8)).clamp(-3.,3.)


def bounded_perturbation(actions,direction,epsilon,lower,upper):
    return (actions+epsilon*direction).clamp(lower,upper)


def paired_statistics(predicted_base,real_base,predicted_plus,real_plus,
                      predicted_minus,real_minus,repeated_real,valid):
    noise=(real_base-repeated_real).abs()
    tolerance=(5*noise).clamp_min(1e-5)
    pd=predicted_plus-predicted_minus;rd=real_plus-real_minus
    resolved=valid & (pd.abs()>1e-5) & (rd.abs()>2*tolerance)
    def average(value,mask):return float(value[mask].mean()) if mask.any() else None
    return dict(valid_windows=int(valid.sum()),resolved_windows=int(resolved.sum()),
        sign_agreement_fraction=average((pd*rd>0).float(),resolved),
        predicted_plus_gain=average(predicted_plus-predicted_base,valid),
        real_plus_gain=average(real_plus-real_base,valid),
        predicted_minus_gain=average(predicted_minus-predicted_base,valid),
        real_minus_gain=average(real_minus-real_base,valid),
        real_plus_improvement_fraction=average((real_plus-real_base>tolerance).float(),valid),
        repeat_absolute_difference_mean=average(noise,valid),
        repeat_absolute_difference_max=float(noise[valid].max()) if valid.any() else None)


def imagined_objectives(model,critic,lib,initial,actions,previous,context,frames,future_frames):
    from x2_style_mimic.algorithms.supermimic_tracking.wm_trainer import reference_at
    state=initial;survival=torch.ones(len(initial),device=initial.device)
    task=torch.zeros_like(survival);last=lib.last_frame_of(frames)
    for step in range(len(actions)):
        target=torch.minimum(frames+step,last)
        ref=reference_at(lib,target)
        state=model.step(state,actions[step],context)
        reward=model.heads.reward(state,actions[step],previous,ref)['total']
        task=task+.99**step*survival*reward
        survival=survival*(1-model.termination_logits(state,ref).sigmoid())
        previous=actions[step]
    target=torch.minimum(frames+len(actions),last)
    ref=reference_at(lib,target)
    value=critic(model.spec.canonical(state),lib.goal_window(target,future_frames),
                 model.spec.normalize_xi(context),model.heads.root_error_features(state,ref))
    return dict(task=task,full=task+.99**len(actions)*survival*value)


@torch.no_grad()
def real_trial(env,policy,critic,model,reset_frames,horizon,burn_steps,sequence=None):
    cmd=env.command_manager.get_term('motion')
    cmd.fixed_reset_frames=reset_frames
    cmd.length.zero_();torch.manual_seed(42)
    obs,_=env.reset();valid=torch.ones(env.num_envs,device=env.device,dtype=torch.bool)
    for _ in range(burn_steps):
        obs,_,terminated,truncated,_=env.step(policy(obs['policy'],obs['goal']))
        valid &= ~(terminated|truncated)
    initial=env.read_tracking_state();frames=cmd.frame.clone();context=cmd.xi.clone()
    previous=env.action_manager.get_term('joint_pos').raw_actions.clone()
    active=valid.clone();total=torch.zeros(env.num_envs,device=env.device)
    executed=[];ending=initial.clone();held_action=previous.clone()
    for step in range(horizon):
        action=policy(obs['policy'],obs['goal']) if sequence is None else sequence[step]
        if sequence is None:
            # 终止后的自动重置动作不能混入待比较序列；尾段固定保持最后动作。
            action=torch.where(active[:,None],action,held_action)
            held_action=action.clone()
        obs,reward,terminated,truncated,extra=env.step(action)
        transition=extra['tracking_transition']
        if float((transition['executed']-action).abs().max())>2e-5:
            raise RuntimeError('无噪声动作与实际执行动作不一致，先核对裁剪边界')
        total+=.99**step*active*reward
        ending[active]=transition['next_state'][active]
        active &= ~(terminated|truncated)
        executed.append(action.clone())
    from x2_style_mimic.algorithms.supermimic_tracking.wm_trainer import reference_at
    target=torch.minimum(frames+horizon,cmd.library.last_frame_of(frames))
    ref=reference_at(cmd.library,target)
    value=critic(model.spec.canonical(ending),cmd.library.goal_window(target,env.cfg.commands.motion.future_frames),
                 model.spec.normalize_xi(context),model.heads.root_error_features(ending,ref))
    return dict(initial=initial,frames=frames,context=context,previous=previous,actions=torch.stack(executed),
                valid=valid,task=total,full=total+.99**horizon*active*value,alive_at_end=active)


def run(env,cfg,checkpoint,args):
    from x2_style_mimic.algorithms.supermimic_tracking.world_model import WorldModel
    from x2_style_mimic.algorithms.supermimic_tracking.nets import Policy,Critic
    from x2_style_mimic.algorithms.supermimic_tracking.training import asset_contract,validate_fk
    if checkpoint['contract']!=asset_contract(env):raise ValueError('任务合同与检查点不一致')
    cmd=env.command_manager.get_term('motion');lib=cmd.library
    model=WorldModel(cfg.world_model,cmd.heads,env.device)
    policy=Policy(cfg.policy).to(env.device);critic=Critic(cfg.critic,pcfg=cfg.policy).to(env.device)
    for name,module in [('world',model),('policy',policy),('critic',critic)]:
        module.load_state_dict(checkpoint[name],strict=True);module.eval();module.requires_grad_(False)
    frames=evaluation_frames(lib,env.num_envs,42)
    earliest=lib.starts[lib.frame_clip[frames]]
    frames=torch.maximum(earliest,torch.minimum(frames,lib.last_frame_of(frames)-max(args.horizons)-args.burn_steps-2))
    lower=torch.as_tensor(cfg.policy.action_lower,device=env.device)
    upper=torch.as_tensor(cfg.policy.action_upper,device=env.device)
    report=dict(checkpoint=str(args.checkpoint),horizons={},notes=[
        '独立优化动作序列，不是 Actor 参数更新；不能据此宣称完整 BPTT 已通过。',
        'task 使用折扣奖励与存活权重；full 额外含同一个 Critic 的末端值。',
        '模型存活为学习概率，真实存活为实际终止；差异属于本次检查对象。',
        '基线真实终止后的动作保持最后命令，不采集自动重置后的新回合动作。',
        '只统计重复回放中均未在预运行阶段结束的起点。',
        '显著性容差=max(5倍基线重复差异,1e-5)，是诊断阈值，不是统计置信区间。',
        '正反方向比较使用相同固定起点；窗口来自同一动作库，不能视为独立泛化样本。'])
    artifacts={}
    args.output.mkdir(parents=True,exist_ok=True)
    for horizon in args.horizons:
        base=real_trial(env,policy,critic,model,frames,horizon,args.burn_steps)
        validate_fk(env)
        repeat=real_trial(env,policy,critic,model,frames,horizon,args.burn_steps,base['actions'])
        start_difference=(base['initial']-repeat['initial']).abs().amax(-1)
        valid=base['valid']&repeat['valid']&(start_difference<1e-4)
        preconditions=dict(base_burn_survivors=int(base['valid'].sum()),repeat_burn_survivors=int(repeat['valid'].sum()),
            initial_repeat_max_abs_per_env=start_difference.tolist(),
            initial_repeat_block_max_abs={name:float((base['initial'][:,start:end]-repeat['initial'][:,start:end]).abs().max())
                for name,start,end in (('root_position',0,3),('root_quaternion',3,7),('joints',7,38),
                                       ('root_linear_velocity',38,41),('root_angular_velocity',41,44),('joint_velocity',44,75))})
        if not valid.any():
            report['horizons'][str(horizon)]=dict(status='起点不可重复，未执行梯度方向判定',preconditions=preconditions)
            artifacts[horizon]=dict(base={k:v.detach().cpu() for k,v in base.items()},
                                   repeat={k:v.detach().cpu() for k,v in repeat.items()})
            (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            torch.save(artifacts,args.output/'paired_trials.pt')
            print(f'[回放检查 H={horizon}] '+json.dumps(preconditions,ensure_ascii=False),flush=True)
            continue
        actions=base['actions'].detach().clone().requires_grad_()
        predicted=imagined_objectives(model,critic,lib,base['initial'],actions,base['previous'],
                                     base['context'],base['frames'],cfg.policy_trainer.future_frames)
        gradients={name:torch.autograd.grad((predicted[name]*valid).sum(),actions,retain_graph=name=='task')[0]
                   for name in ('task','full')}
        results=[];trial_data=[]
        for name,gradient in gradients.items():
            if not torch.isfinite(gradient).all():raise RuntimeError('动作方向梯度含非有限值')
            direction=action_direction(gradient).detach()
            for epsilon in args.epsilons:
                trials={};predictions={}
                for sign in (-1,1):
                    perturbed=bounded_perturbation(base['actions'],direction,sign*epsilon,lower,upper)
                    trials[sign]=real_trial(env,policy,critic,model,frames,horizon,args.burn_steps,perturbed)
                    with torch.no_grad():
                        predictions[sign]=imagined_objectives(model,critic,lib,base['initial'],perturbed,
                            base['previous'],base['context'],base['frames'],cfg.policy_trainer.future_frames)
                mask=valid&trials[1]['valid']&trials[-1]['valid']
                for sign in (-1,1):
                    mask &= (trials[sign]['initial']-base['initial']).abs().amax(-1)<1e-4
                comparisons={objective:paired_statistics(predicted[objective].detach(),base[objective],
                    predictions[1][objective],trials[1][objective],predictions[-1][objective],
                    trials[-1][objective],repeat[objective],mask) for objective in ('task','full')}
                survived=mask&base['alive_at_end']&repeat['alive_at_end']
                survivor_comparisons={objective:paired_statistics(predicted[objective].detach(),base[objective],
                    predictions[1][objective],trials[1][objective],predictions[-1][objective],
                    trials[-1][objective],repeat[objective],survived) for objective in ('task','full')}
                results.append(dict(direction=name,epsilon=epsilon,comparisons=comparisons,
                                    baseline_survivor_comparisons=survivor_comparisons))
                trial_data.append(dict(direction=name,epsilon=epsilon,mask=mask.cpu(),
                    plus={k:trials[1][k].cpu() for k in ('task','full')},
                    minus={k:trials[-1][k].cpu() for k in ('task','full')},
                    predicted_plus={k:v.detach().cpu() for k,v in predictions[1].items()},
                    predicted_minus={k:v.detach().cpu() for k,v in predictions[-1].items()}))
                print(f'[方向检查 H={horizon} {name} eps={epsilon}] '+json.dumps(comparisons['task'],ensure_ascii=False),flush=True)
        report['horizons'][str(horizon)]=dict(valid_windows=int(valid.sum()),
            preconditions=preconditions,
            baseline_survivor_count=int((valid&base['alive_at_end']&repeat['alive_at_end']).sum()),
            baseline_initial_state_repeat_max_abs=float(start_difference.max()),results=results)
        artifacts[horizon]=dict(base={k:v.detach().cpu() for k,v in base.items()},
            repeat={k:v.detach().cpu() for k,v in repeat.items()},
            predicted={k:v.detach().cpu() for k,v in predicted.items()},
            gradients={k:v.detach().cpu() for k,v in gradients.items()},trials=trial_data)
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        torch.save(artifacts,args.output/'paired_trials.pt')
    return report


def main():
    from isaaclab.app import add_launcher_args,launch_simulation
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--horizons',type=int,nargs='+',default=[8,32])
    parser.add_argument('--epsilons',type=float,nargs='+',default=[.05,.1])
    parser.add_argument('--burn-steps',type=int,default=4)
    parser.add_argument('--num-envs',type=int,default=32)
    parser.add_argument('--urdf',type=Path,default=Path.home()/'robotics_ws/agibot_model/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra.urdf')
    add_launcher_args(parser);args=parser.parse_args()
    if min(args.horizons)<1 or args.burn_steps<0 or args.num_envs<1 or any(not math.isfinite(x) or x<=0 for x in args.epsilons):
        parser.error('步数、环境数和扰动幅度配置无效')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    entry=load_file('_direction_config',PROJECT/'scripts/supermimic_tracking/train.py')
    compat=load_file('_direction_compat',Path(__file__).resolve().parents[1]/'train_supermimic_tracking.py')
    from x2_style_mimic.algorithms.supermimic_tracking.config import TrackingTrainCfg
    cfg=TrackingTrainCfg();entry.update_config(cfg,checkpoint['config']);cfg.validate()
    if cfg.domain_randomization or cfg.reset_noise or cfg.obs_noise:
        raise ValueError('方向诊断先要求关闭随机物理、重置噪声和观测噪声')
    from x2_style_mimic.tasks.manager_based.x2_supermimic_tracking.env_cfg import X2SuperMimicTrackingEnvCfg
    env_cfg=X2SuperMimicTrackingEnvCfg()
    env_cfg.scene.num_envs=args.num_envs;env_cfg.seed=42;env_cfg.sim.device=args.device or cfg.device
    env_cfg.domain_randomization=env_cfg.reset_noise=env_cfg.obs_noise=False
    env_cfg.actions.joint_pos.noise_std=0.
    env_cfg.observations.policy.history_length=cfg.policy.history
    env_cfg.commands.motion.future_frames=cfg.policy_trainer.future_frames
    env_cfg.root_height_threshold=cfg.world_model.height_threshold;env_cfg.root_tilt_threshold=cfg.world_model.tilt_threshold
    env_cfg.urdf=str(args.urdf.resolve());compat._replace_null_manager_configs(env_cfg)
    sys.argv=[sys.argv[0]]
    with launch_simulation(env_cfg,args):
        from x2_style_mimic.tasks.manager_based.x2_supermimic_tracking.env import X2SuperMimicTrackingEnv
        env=X2SuperMimicTrackingEnv(env_cfg)
        try:run(env,cfg,checkpoint,args)
        finally:env.close()


if __name__=='__main__':main()
