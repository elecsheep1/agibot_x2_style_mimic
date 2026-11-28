"""固定同批真实动作窗口，分解单步／多步、速度／位置修正和分阶段模型误差；纯 CPU。"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from diagnose_tracking import valid_windows,load_file,PROJECT


def stats(x):
    x=x.detach().float().reshape(-1)
    return dict(count=x.numel(),mean=float(x.mean()),rms=float(x.square().mean().sqrt()),
        p50=float(torch.quantile(x,.5)),p90=float(torch.quantile(x,.9)),maximum=float(x.max()))


def component_metrics(pred,true,quat_box_minus):
    rotation=quat_box_minus(pred[...,3:7],true[...,3:7])*180/torch.pi
    return dict(root_position_m=stats((pred[...,:3]-true[...,:3]).norm(dim=-1)),
        root_position_signed_bias_m=(pred[...,:3]-true[...,:3]).mean(0).tolist(),
        root_position_axis_mae_m=(pred[...,:3]-true[...,:3]).abs().mean(0).tolist(),
        root_rotation_deg=stats(rotation.norm(dim=-1)),rotation_vector_axis_mae_deg=rotation.abs().mean(0).tolist(),
        root_linear_velocity_m_s=stats((pred[...,38:41]-true[...,38:41]).norm(dim=-1)),
        root_body_angular_velocity_rad_s=stats((pred[...,41:44]-true[...,41:44]).norm(dim=-1)),
        joint_position_rmse_rad=stats((pred[...,7:38]-true[...,7:38]).square().mean(-1).sqrt()),
        joint_velocity_rmse_rad_s=stats((pred[...,44:75]-true[...,44:75]).square().mean(-1).sqrt()))


def sample_cohort(data,windows,seed,horizon=32):
    options=valid_windows(data['terminated']|data['truncated'],horizon)
    if not len(options):raise ValueError('没有不跨自动重置的 32 步窗口')
    chosen=torch.randperm(len(options),generator=torch.Generator().manual_seed(seed))[:windows]
    starts,envs=options[chosen].unbind(-1)
    times=starts[:,None]+torch.arange(horizon)[None]
    indices=envs[:,None].expand_as(times)
    return dict(initial=data['state'][starts,envs],before=data['state'][times,indices],
        following=data['next_state'][times,indices],actions=data['executed'][times,indices],
        context=data['context'][starts,envs],frames=data['frame'][times,indices],starts=starts,envs=envs,
        eligible_windows=len(options))


@torch.no_grad()
def variant_rollout(model,cohort,mode):
    from x2_style_mimic.algorithms.supermimic_tracking.rotations import heading_quat,quat_apply_inverse
    state=cohort['initial'].clone()
    output=[]
    for step in range(cohort['actions'].shape[1]):
        if mode=='teacher_forced' or (mode=='reset_every_8' and step%8==0):
            state=cohort['before'][:,step].clone()
        true=cohort['following'][:,step]
        delta=model.dynamics(model.spec.canonical(state),cohort['actions'][:,step],model.spec.normalize_xi(cohort['context']))
        if mode in ('oracle_root_velocity','oracle_all_velocity'):
            delta[:,:3]=quat_apply_inverse(heading_quat(state[:,3:7]),true[:,38:41]-state[:,38:41])/model.spec.out_scale[:3]
            delta[:,3:6]=(true[:,41:44]-state[:,41:44])/model.spec.out_scale[3:6]
        if mode in ('oracle_joint_velocity','oracle_all_velocity'):
            delta[:,6:37]=(true[:,44:75]-state[:,44:75])/model.spec.out_scale[6:37]
        if mode=='zero_position_corrections':delta[:,37:]=0
        state=model.spec.integrate(state,delta)
        output.append(state)
    return torch.stack(output,1)


@torch.no_grad()
def one_step_budget(model,cohort):
    from x2_style_mimic.algorithms.supermimic_tracking.rotations import heading_quat,quat_apply,quat_apply_inverse,quat_mul,quat_inv,rotvec_from_quat
    state=cohort['before'].reshape(-1,75);true=cohort['following'].reshape(-1,75)
    actions=cohort['actions'].reshape(-1,29)
    context=cohort['context'][:,None].expand(-1,cohort['actions'].shape[1],-1).reshape(-1,35)
    delta=model.dynamics(model.spec.canonical(state),actions,model.spec.normalize_xi(context))
    scaled=delta*model.spec.out_scale
    pred=model.spec.integrate(state,delta)
    dt=model.spec.dt;heading=heading_quat(state[:,3:7])
    required_dp=quat_apply_inverse(heading,true[:,:3]-state[:,:3]-dt*true[:,38:41])
    velocity_term=dt*(pred[:,38:41]-true[:,38:41])
    correction_term=quat_apply(heading,scaled[:,37:40]-required_dp)
    actual=pred[:,:3]-true[:,:3]
    budget_residual=float((actual-velocity_term-correction_term).abs().max())
    needed_dq=true[:,7:38]-state[:,7:38]-dt*true[:,44:75]
    needed_rotation=rotvec_from_quat(quat_mul(quat_inv(state[:,3:7]),true[:,3:7]))-dt*true[:,41:44]
    def block(predicted,required):
        return dict(predicted_norm=stats(predicted.norm(dim=-1)),required_norm=stats(required.norm(dim=-1)),
                    error_norm=stats((predicted-required).norm(dim=-1)),signed_bias=(predicted-required).mean(0).tolist())
    return dict(root_position_budget_residual_max_m=budget_residual,
        root_position_total_mse=float(actual.square().sum(-1).mean()),
        root_position_velocity_term_mse=float(velocity_term.square().sum(-1).mean()),
        root_position_correction_term_mse=float(correction_term.square().sum(-1).mean()),
        root_position_cross_term=float(2*(velocity_term*correction_term).sum(-1).mean()),
        root_position_correction=block(scaled[:,37:40],required_dp),
        root_rotation_correction=block(scaled[:,40:43],needed_rotation),
        joint_position_correction=block(scaled[:,43:74],needed_dq))


def grouped_metrics(cohort,pred,true,rotations):
    s=cohort['initial']
    root_up=1-2*s[:,4:6].square().sum(-1)
    masks={'initial_low':s[:,2]<.3,'initial_middle':(s[:,2]>=.3)&(s[:,2]<.55),
           'initial_high_upright':(s[:,2]>=.55)&(root_up>=.85),
           'initial_high_tilted':(s[:,2]>=.55)&(root_up<.85)}
    # 速度跳变可能来自落地、支撑转换或执行器动作，不等价于接触力。
    velocity_jump=(cohort['following'][...,38:41]-cohort['before'][...,38:41]).norm(dim=-1).max(-1).values
    threshold=torch.quantile(velocity_jump,.75)
    masks['high_velocity_jump_candidate']=velocity_jump>=threshold
    masks['other_windows']=velocity_jump<threshold
    result={}
    for name,mask in masks.items():
        result[name]=dict(windows=int(mask.sum()))
        if mask.any():result[name].update(component_metrics(pred[mask],true[mask],rotations))
    return result,dict(velocity_jump_threshold_m_s=float(threshold),meaning='同批窗口内速度跳变较大的上四分位候选组，不是实测接触事件')


def validate_data(data):
    done=data['terminated']|data['truncated']
    contiguous=~done[:-1]
    mismatch=(data['state'][1:]-data['next_state'][:-1]).abs().amax(-1)
    largest=float(mismatch[contiguous].max()) if contiguous.any() else 0.
    frame_mismatch=int(((data['frame'][1:]!=data['frame'][:-1]+1)&contiguous).sum())
    if largest>1e-5 or frame_mismatch:raise ValueError('原始转移不连续或参考帧错位，先修数据口径')
    if not all(bool(torch.isfinite(data[key]).all()) for key in ('state','next_state','executed','context')):
        raise ValueError('诊断数据包含非有限值')
    return dict(continuity_max_abs=largest,frame_mismatches=frame_mismatch,
                transitions=data['state'].shape[0]*data['state'].shape[1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--windows',type=int,default=128)
    parser.add_argument('--seed',type=int,default=43)
    args=parser.parse_args()
    if args.windows<1:parser.error("窗口数量必须为正数")
    torch.set_num_threads(2)
    from x2_style_mimic.algorithms.supermimic_tracking.world_model import WorldModel,WorldModelCfg
    from x2_style_mimic.algorithms.supermimic_tracking.heads import AnalyticHeads
    from x2_style_mimic.algorithms.supermimic_tracking.rotations import quat_box_minus
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    data=torch.load(args.data,map_location='cpu',weights_only=False)['data']
    data_check=validate_data(data)
    cfg=WorldModelCfg()
    entry=load_file('_tracking_config_reader',PROJECT/'scripts/supermimic_tracking/train.py')
    entry.update_config(cfg,checkpoint['config']['world_model'])
    model=WorldModel(cfg,SimpleNamespace(termination_features=AnalyticHeads.termination_features),'cpu').eval()
    model.load_state_dict(checkpoint['world'],strict=True);model.requires_grad_(False)
    cohort=sample_cohort(data,args.windows,args.seed)
    true=cohort['following']
    modes=('baseline','teacher_forced','reset_every_8','oracle_root_velocity','oracle_joint_velocity',
           'oracle_all_velocity','zero_position_corrections')
    curves={};endpoints={};baseline=None
    with torch.no_grad():
        for mode in modes:
            prediction=variant_rollout(model,cohort,mode)
            if mode=='baseline':baseline=prediction
            curves[mode]=[dict(step=step+1,root_position_mean_m=float((prediction[:,step,:3]-true[:,step,:3]).norm(dim=-1).mean()),
                root_rotation_mean_deg=float(quat_box_minus(prediction[:,step,3:7],true[:,step,3:7]).norm(dim=-1).mean()*180/torch.pi),
                joint_position_rmse_rad=float((prediction[:,step,7:38]-true[:,step,7:38]).square().mean(-1).sqrt().mean())) for step in range(32)]
            endpoints[mode]={str(h):component_metrics(prediction[:,h-1],true[:,h-1],quat_box_minus) for h in (1,8,16,32)}
            print(f'[{mode}] 32步位置={curves[mode][-1]["root_position_mean_m"]:.4f} m，'
                  f'旋转={curves[mode][-1]["root_rotation_mean_deg"]:.2f}°',flush=True)
        q_error=(baseline[:,-1,7:38]-true[:,-1,7:38]).square().mean(0).sqrt()
        qd_error=(baseline[:,-1,44:75]-true[:,-1,44:75]).square().mean(0).sqrt()
        names=checkpoint['contract']['joints']
        joints=[dict(name=name,position_rmse_rad=float(q_error[i]),velocity_rmse_rad_s=float(qd_error[i])) for i,name in enumerate(names)]
        groups,group_definition=grouped_metrics(cohort,baseline[:,-1],true[:,-1],quat_box_minus)
        import numpy as np
        paths=[path for path in checkpoint['contract']['files'] if path.endswith('.npz')]
        lengths=[]
        for path in paths:
            with np.load(path,allow_pickle=False) as motion:lengths.append(len(motion['time_s']))
        lengths=torch.tensor(lengths)
        ends=lengths.cumsum(0);starts=torch.cat((torch.zeros(1,dtype=torch.long),ends[:-1]))
        initial_frame=cohort['frames'][:,0]
        motion_ids=torch.bucketize(initial_frame,ends,right=True)
        if bool((motion_ids>=len(paths)).any()):raise ValueError('参考帧超出动作库范围')
        phase=(initial_frame-starts[motion_ids]).float()/(lengths[motion_ids]-1).clamp_min(1)
        reference_groups={}
        for i,path in enumerate(paths):
            mask=motion_ids==i
            reference_groups[Path(path).name]={'windows':int(mask.sum())}
            if mask.any():reference_groups[Path(path).name].update(component_metrics(baseline[mask,-1],true[mask,-1],quat_box_minus))
        phase_groups={}
        for i in range(4):
            mask=(phase>=i/4)&(phase<(i+1)/4 if i<3 else phase<=1)
            name=f'reference_phase_{i/4:.2f}_{(i+1)/4:.2f}'
            phase_groups[name]={'windows':int(mask.sum())}
            if mask.any():phase_groups[name].update(component_metrics(baseline[mask,-1],true[mask,-1],quat_box_minus))
        done=data['terminated']|data['truncated']
        episode=torch.cat((torch.zeros(1,done.shape[1],dtype=torch.long),done.long().cumsum(0)[:-1]),0)
        sampled_episodes=torch.stack((cohort['envs'],episode[cohort['starts'],cohort['envs']]),-1).unique(dim=0)
        report=dict(checkpoint=str(args.checkpoint.resolve()),data=str(args.data.resolve()),seed=args.seed,
            file_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.checkpoint,args.data)},
            data_check=data_check,eligible_windows=cohort['eligible_windows'],sampled_windows=len(cohort['starts']),
            start_indices=cohort['starts'].tolist(),environment_indices=cohort['envs'].tolist(),
            endpoints=endpoints,curves=curves,teacher_forced_error_budget=one_step_budget(model,cohort),
            joints_by_position_error=sorted(joints,key=lambda row:row['position_rmse_rad'],reverse=True),
            joints_by_velocity_error=sorted(joints,key=lambda row:row['velocity_rmse_rad_s'],reverse=True),
            groups=groups,group_definition=group_definition,by_reference_motion=reference_groups,
            by_reference_phase=phase_groups,distinct_sampled_episodes=len(sampled_episodes),
            notes=['全部变体使用同一批 32 步有效窗口和实际执行动作，不跨重置。',
                   '起点不重复，但同一回合内窗口可能重叠；统计不是独立试验置信区间。',
                   '32 步共同有效窗偏向较长回合，不覆盖提前终止的短片段，不代表总体失败率。',
                   '参考阶段分组只是动作帧进度的四分位，不自动等价于起身动作语义阶段。',
                   'oracle 使用未来真实速度，仅用于诊断，不能部署。',
                   '移除位置修正项会改变网络使用分布，不能据此直接认定应该删除修正项。',
                   '各变体会改变后续输入，结果是敏感性对照，不是可相加的因果贡献率。',
                   '旋转轴分量是旋转向量，不是欧拉角差；角速度按状态定义使用根局部坐标。',
                   '没有记录接触力，不能把速度跳变组直接归因于碰撞。'])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for mode,rows in curves.items():
            axes[0].plot([r['step'] for r in rows],[r['root_position_mean_m'] for r in rows],label=mode)
            axes[1].plot([r['step'] for r in rows],[r['root_rotation_mean_deg'] for r in rows],label=mode)
        axes[0].set(xlabel='Prediction step',ylabel='Root position error (m)')
        axes[1].set(xlabel='Prediction step',ylabel='Root rotation error (deg)')
        for ax in axes:ax.grid(alpha=.25)
        axes[1].legend(fontsize=7)
        fig.savefig(args.output/'error_curves.png',dpi=150);plt.close(fig)
    except ImportError:print('[图表] 未安装 matplotlib，已保留 JSON 曲线数据',flush=True)
    print(f'[报告] {args.output/"report.json"}',flush=True)


if __name__=='__main__':main()
