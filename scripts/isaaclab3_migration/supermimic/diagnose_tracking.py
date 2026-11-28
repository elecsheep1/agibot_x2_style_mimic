"""IsaacLab 3.0 独立跟踪诊断：冻结权重，采集真实动作／奖励，核对开环误差。"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]


def load_file(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def statistics(value):
    import torch
    value = value.detach().float().reshape(-1)
    return dict(mean=float(value.mean()),p50=float(torch.quantile(value,.5)),
                p90=float(torch.quantile(value,.9)),p99=float(torch.quantile(value,.99)),maximum=float(value.max()))


def target_statistics(actions,targets,hard_limits,soft_limits,names):
    """actions/targets=[T,N,A]；关节边界=[N,A,2]。"""
    import torch
    hard = ((targets<hard_limits[None,...,0]) | (targets>hard_limits[None,...,1])).float()
    soft = ((targets<soft_limits[None,...,0]) | (targets>soft_limits[None,...,1])).float()
    return dict(raw_action_abs=statistics(actions.abs()),target_abs=statistics(targets.abs()),
        target_hard_limit_fraction=float(hard.mean()),target_soft_limit_fraction=float(soft.mean()),
        per_joint=[dict(name=name,raw_action_abs_mean=float(actions[...,i].abs().mean()),
                        target_min=float(targets[...,i].min()),target_max=float(targets[...,i].max()),
                        target_hard_limit_fraction=float(hard[...,i].mean()),
                        target_soft_limit_fraction=float(soft[...,i].mean())) for i,name in enumerate(names)])


def valid_windows(done,horizon):
    """最后一条转移允许终止，其后的重置状态不能进入同一窗口。"""
    import torch
    count = done.shape[0]-horizon+1
    if count<=0:return torch.empty(0,2,device=done.device,dtype=torch.long)
    mask = torch.ones(count,done.shape[1],device=done.device,dtype=torch.bool)
    for j in range(horizon-1):mask &= ~done[j:j+count]
    return mask.nonzero()


def run_diagnostics(env,cfg,checkpoint,args):
    import random
    import numpy as np
    import torch
    from x2_style_mimic.algorithms.supermimic_tracking.world_model import WorldModel
    from x2_style_mimic.algorithms.supermimic_tracking.nets import Policy
    from x2_style_mimic.algorithms.supermimic_tracking.training import SCHEMA,asset_contract,validate_fk
    from x2_style_mimic.algorithms.supermimic_tracking.wm_trainer import reference_at
    from x2_style_mimic.algorithms.supermimic_tracking.rotations import quat_box_minus
    if checkpoint.get('kind')!=SCHEMA or checkpoint['contract']!=asset_contract(env):
        raise ValueError('检查点与当前资产、参考动作或奖励配置不一致')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    command = env.command_manager.get_term('motion')
    command.bin_return.copy_(checkpoint['bin_return'])
    command.bin_memory.copy_(checkpoint['bin_memory'])
    model = WorldModel(cfg.world_model,command.heads,env.device).eval()
    policy = Policy(cfg.policy).to(env.device).eval()
    model.load_state_dict(checkpoint['world'],strict=True)
    policy.load_state_dict(checkpoint['policy'],strict=True)
    model.requires_grad_(False);policy.requires_grad_(False)
    term = env.action_manager.get_term('joint_pos')
    robot = env.scene['robot']
    ids = command.control_ids
    def tensor(value):return value.torch if hasattr(value,'torch') else value
    hard = tensor(robot.data.joint_pos_limits)[:,ids].clone()
    soft = tensor(robot.data.soft_joint_pos_limits)[:,ids].clone()
    fields = {key:[] for key in ('state','next_state','action','executed','frame','context','terminated','truncated')}
    targets,previous,real_terms = [],[],{}
    obs,_ = env.reset()
    validate_fk(env)
    with torch.no_grad():
        for step in range(args.steps):
            previous.append(term.raw_actions.clone())
            action = policy(obs['policy'],obs['goal'])
            obs,_,_,_,extras = env.step(action)
            transition = extras['tracking_transition']
            for key in fields:fields[key].append(transition[key].detach().clone())
            target = transition['executed']*term._scale+term._offset
            if term.cfg.clip is not None:target = target.clamp(term._clip[...,0],term._clip[...,1])
            targets.append(target.clone())
            # 自动重置只使缓存失效，不清除本步 reward manager 已算出的分项。
            for name,value in command._terms.items():real_terms.setdefault(name,[]).append(value.clone())
            if (step+1)%50==0:print(f'[采集] {step+1}/{args.steps}',flush=True)
        data = {name:torch.stack(value) for name,value in fields.items()}
        data['previous_action'] = torch.stack(previous)
        measured = {name:torch.stack(value) for name,value in real_terms.items()}
        report = dict(checkpoint_iteration=checkpoint['iteration'],seed=args.seed,steps=args.steps,num_envs=env.num_envs,
            settings=dict(domain_randomization=env.cfg.domain_randomization,reset_noise=env.cfg.reset_noise,
                          obs_noise=env.cfg.obs_noise,action_noise_std=term.cfg.noise_std,sampling='saved_return_bins'),
            actions=target_statistics(data['action'],torch.stack(targets),hard,soft,env.cfg.control_joint_names),
            real_rewards={name:statistics(value) for name,value in measured.items()},
            real_terminated_fraction=float(data['terminated'].float().mean()),
            real_truncated_fraction=float(data['truncated'].float().mean()),open_loop={})
        done = data['terminated'] | data['truncated']
        for horizon in (1,8,32):
            candidates = valid_windows(done,horizon)
            if not len(candidates):
                report['open_loop'][str(horizon)] = dict(eligible_windows=0)
                continue
            indices = torch.randperm(len(candidates),device=env.device)[:args.windows]
            starts,env_ids = candidates[indices].unbind(-1)
            state = data['state'][starts,env_ids]
            context = data['context'][starts,env_ids]
            predictions,truth,reward_errors,proxy_errors,probabilities,labels = [],[],[],[],[],[]
            for j in range(horizon):
                times = starts+j
                state = model.step(state,data['executed'][times,env_ids],context)
                following = data['next_state'][times,env_ids]
                ref = reference_at(command.library,data['frame'][times,env_ids])
                action,previous_action = data['action'][times,env_ids],data['previous_action'][times,env_ids]
                predicted_reward = command.heads.reward(state,action,previous_action,ref)
                true_proxy = command.heads.reward(following,action,previous_action,ref)
                reward_errors.append({name:(value-measured[name][times,env_ids]).abs() for name,value in predicted_reward.items()})
                proxy_errors.append({name:(value-measured[name][times,env_ids]).abs() for name,value in true_proxy.items()})
                probabilities.append(model.termination_logits(state,ref).sigmoid())
                labels.append(data['terminated'][times,env_ids])
                predictions.append(state);truth.append(following)
            pred,true = predictions[-1],truth[-1]
            probability,label = torch.stack(probabilities),torch.stack(labels)
            def selected_stats(mask):
                return statistics(probability[mask]) if mask.any() else None
            entry = dict(eligible_windows=len(candidates),sampled_windows=len(indices),
                endpoint_root_position_m=statistics((pred[:,:3]-true[:,:3]).norm(dim=-1)),
                endpoint_root_rotation_deg=statistics(quat_box_minus(pred[:,3:7],true[:,3:7]).norm(dim=-1)*180/torch.pi),
                endpoint_joint_position_rmse_rad=statistics((pred[:,7:38]-true[:,7:38]).square().mean(-1).sqrt()),
                endpoint_root_linear_velocity_m_s=statistics((pred[:,38:41]-true[:,38:41]).norm(dim=-1)),
                endpoint_root_angular_velocity_rad_s=statistics((pred[:,41:44]-true[:,41:44]).norm(dim=-1)),
                endpoint_joint_velocity_rmse_rad_s=statistics((pred[:,44:75]-true[:,44:75]).square().mean(-1).sqrt()),
                endpoint_predicted_soft_limit_fraction=float(((pred[:,7:38]<command.heads.soft_limits[:,0]) |
                    (pred[:,7:38]>command.heads.soft_limits[:,1])).float().mean()),
                reward_mae={name:float(torch.stack([part[name] for part in reward_errors]).mean()) for name in reward_errors[0]},
                true_state_fk_reward_mae={name:float(torch.stack([part[name] for part in proxy_errors]).mean()) for name in proxy_errors[0]},
                termination_positive_fraction=float(label.float().mean()),
                termination_predicted_on_positive=selected_stats(label),termination_predicted_on_negative=selected_stats(~label))
            report['open_loop'][str(horizon)] = entry
            print(f'[开环 {horizon}] 根位置均值 {entry["endpoint_root_position_m"]["mean"]:.4f} m；'
                  f'旋转均值 {entry["endpoint_root_rotation_deg"]["mean"]:.2f}°；样本 {len(indices)}/{len(candidates)}',flush=True)
        report['notes'] = ['1/8/32 步按各自有效窗口采样，不是严格同批起点；需结合覆盖数解读。',
            '固定真实执行动作开环预测；不在预测状态上重新计算 Actor 动作。',
            '真实终止在窗口末端允许保留；没有跨越自动重置。',
            '目标越限是 PD 指令越限，不等于实际关节越限。',
            '终止概率按开环预测状态统计；每个窗口可能不含终止正例。']
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        torch.save(dict(data={key:value.cpu() for key,value in data.items()},
                        real_rewards={key:value.cpu() for key,value in measured.items()}),args.output/'transitions.pt')
        print(f'[报告] {args.output/"report.json"}',flush=True)
    return report


def main():
    from isaaclab.app import add_launcher_args,launch_simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=256)
    parser.add_argument('--windows',type=int,default=128)
    parser.add_argument('--num-envs',type=int,default=16)
    parser.add_argument('--seed',type=int,default=43)
    parser.add_argument('--urdf',type=Path,default=Path.home()/'robotics_ws/agibot_model/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra.urdf')
    parser.add_argument('--motion-manifest',type=Path)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.steps<32 or args.windows<1 or args.num_envs<1:parser.error('至少采集 32 步，窗口数量和环境数量必须为正数')
    import torch
    from x2_style_mimic.algorithms.supermimic_tracking.config import TrackingTrainCfg
    entry = load_file('_tracking_entry',PROJECT/'scripts/supermimic_tracking/train.py')
    migration = load_file('_tracking_migration',HERE.parent/'train_supermimic_tracking.py')
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg = TrackingTrainCfg();entry.update_config(cfg,checkpoint['config']);cfg.validate()
    from x2_style_mimic.tasks.manager_based.x2_supermimic_tracking.env_cfg import X2SuperMimicTrackingEnvCfg
    env_cfg = X2SuperMimicTrackingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device or cfg.device
    env_cfg.domain_randomization,env_cfg.reset_noise,env_cfg.obs_noise = cfg.domain_randomization,cfg.reset_noise,cfg.obs_noise
    env_cfg.actions.joint_pos.noise_std = cfg.action_noise_std
    env_cfg.observations.policy.history_length = cfg.policy.history
    env_cfg.commands.motion.future_frames = cfg.policy_trainer.future_frames
    env_cfg.root_height_threshold,env_cfg.root_tilt_threshold = cfg.world_model.height_threshold,cfg.world_model.tilt_threshold
    env_cfg.urdf = str(args.urdf.expanduser().resolve())
    if args.motion_manifest:env_cfg.motion_manifest = str(args.motion_manifest.expanduser().resolve())
    migration._replace_null_manager_configs(env_cfg)
    sys.argv = [sys.argv[0]]
    with launch_simulation(env_cfg,args):
        from x2_style_mimic.tasks.manager_based.x2_supermimic_tracking.env import X2SuperMimicTrackingEnv
        env = X2SuperMimicTrackingEnv(env_cfg)
        try:run_diagnostics(env,cfg,checkpoint,args)
        finally:env.close()


if __name__=='__main__': main()
