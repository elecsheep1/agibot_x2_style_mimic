"""固定真实数据补训动力学；按环境隔离留出集，不更新策略、价值网络和失败头。"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from audit_model_errors import sample_cohort, validate_data, variant_rollout, component_metrics
from diagnose_tracking import PROJECT, load_file, valid_windows


def split_data(data, seed):
    count=data['state'].shape[1]
    if count<6:
        raise ValueError('至少需要 6 个环境进行训练、验证和测试隔离')
    order=torch.randperm(count,generator=torch.Generator().manual_seed(seed))
    held=max(1,count//8)
    ids={'validation':order[:held], 'test':order[held:2*held], 'train':order[2*held:]}
    return {name:{k:v[:,ix] for k,v in data.items() if isinstance(v,torch.Tensor) and v.ndim>=2 and v.shape[1]==count}
            for name,ix in ids.items()}, {name:ix.tolist() for name,ix in ids.items()}


def gather(data, options, batch_size, horizon, generator):
    selected=options[torch.randint(len(options),(batch_size,),generator=generator)]
    start,env=selected.unbind(-1)
    times=start[:,None]+torch.arange(horizon)[None]
    ee=env[:,None].expand_as(times)
    return data['state'][start,env],data['next_state'][times,ee],data['executed'][times,ee],data['context'][start,env]


def initialize_model(cfg, checkpoint, heads, device, *, scratch=False, hidden=None):
    from x2_style_mimic.algorithms.supermimic_tracking.world_model import WorldModel
    if hidden is not None:
        if not scratch:
            raise ValueError('改变网络容量必须显式选择从头拟合，不能部分加载旧网络')
        if not hidden or min(hidden)<1:
            raise ValueError('隐藏层宽度必须为正整数')
        cfg.hidden=tuple(hidden)
    model=WorldModel(cfg,heads,device)
    if scratch:
        model.term_head.load_state_dict({k.removeprefix('term_head.'):v for k,v in checkpoint['world'].items()
                                        if k.startswith('term_head.')},strict=True)
    else:
        model.load_state_dict(checkpoint['world'],strict=True)
    return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--updates',type=int,default=1000)
    parser.add_argument('--checkpoints',type=int,nargs='+',default=[100,250,500,1000])
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--windows',type=int,default=128)
    parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--seed',type=int,default=43)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--scratch',action='store_true',help='仅动力学从头拟合，失败头继续冻结')
    parser.add_argument('--hidden',type=int,nargs='+',help='从头拟合时的隐藏层宽度')
    args=parser.parse_args()
    if min(args.updates,args.batch_size,args.windows)<1 or args.lr<=0:
        parser.error('更新次数、批量、窗口数量和学习率必须为正数')
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    from x2_style_mimic.algorithms.supermimic_tracking.world_model import WorldModel,WorldModelCfg
    from x2_style_mimic.algorithms.supermimic_tracking.heads import AnalyticHeads
    from x2_style_mimic.algorithms.supermimic_tracking.rotations import quat_box_minus
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    data=torch.load(args.data,map_location='cpu',weights_only=False)['data']
    checked=validate_data(data)
    groups,ids=split_data(data,args.seed)
    cfg=WorldModelCfg()
    load_file('_fixed_refit_cfg',PROJECT/'scripts/supermimic_tracking/train.py').update_config(cfg,checkpoint['config']['world_model'])
    # 使用保存配置中的训练转移数，不改变原有窗口口径。
    horizons=[int(checkpoint['config']['wm_trainer']['window'])]
    if not horizons or min(horizons)<1:raise ValueError('检查点训练窗口配置无效')
    model=initialize_model(cfg,checkpoint,SimpleNamespace(termination_features=AnalyticHeads.termination_features),
                           args.device,scratch=args.scratch,hidden=args.hidden)
    model.term_head.requires_grad_(False)
    head_before=copy.deepcopy(model.term_head.state_dict())
    optimizer=torch.optim.Adam(model.dynamics.parameters(),lr=args.lr)
    cohorts={name:{k:(v.to(args.device) if isinstance(v,torch.Tensor) else v) for k,v in sample_cohort(groups[name],args.windows,args.seed).items()}
             for name in ('train','validation','test')}
    options={h:valid_windows(groups['train']['terminated']|groups['train']['truncated'],h) for h in horizons}
    if any(len(v)==0 for v in options.values()):raise ValueError('训练集缺少连续窗口')
    def evaluate(name):
        model.eval()
        c=cohorts[name];pred=variant_rollout(model,c,'baseline')
        return {str(h):component_metrics(pred[:,h-1],c['following'][:,h-1],quat_box_minus) for h in (1,8,16,32)}
    report=dict(checkpoint=str(args.checkpoint),data=str(args.data),data_check=checked,environment_groups=ids,
        transitions=horizons,window_frames=[h+1 for h in horizons],lr=args.lr,seed=args.seed,
        initialization='scratch' if args.scratch else 'checkpoint',hidden=list(cfg.hidden),
        dynamics_parameters=sum(p.numel() for p in model.dynamics.parameters()),
        fixed_cohorts={name:dict(starts=c['starts'].cpu().tolist(),envs=c['envs'].cpu().tolist()) for name,c in cohorts.items()},
        notes=['只更新 dynamics，使用原有归一化状态误差损失；优化器重新初始化。',
               '训练、验证、测试按环境隔离；固定窗口起点唯一，但回合内窗口可以重叠。',
               '仅按验证集 8 步误差选择模型，测试集不参与选择。'],history=[])
    args.output.mkdir(parents=True,exist_ok=True)
    report['before']={name:evaluate(name) for name in cohorts}
    def score(metrics):
        # 与原预热相同，以归一化状态平方误差选模型。
        c=cohorts['validation'];pred=variant_rollout(model,c,'baseline')
        return float(model.spec.normalized_sq_error(pred[:,7],c['following'][:,7]).mean())
    best=score(None);best_update=0;best_state=copy.deepcopy(model.state_dict())
    generator=torch.Generator().manual_seed(args.seed)
    schedule=sorted(set([v for v in args.checkpoints if 0<v<=args.updates]+[args.updates]))
    for update in range(1,args.updates+1):
        model.train();loss=0.
        for h in horizons:
            s,true,actions,context=gather(groups['train'],options[h],args.batch_size,h,generator)
            s,true,actions,context=[v.to(args.device) for v in (s,true,actions,context)]
            subtotal=0.
            for step in range(h):
                s=model.step(s,actions[:,step],context)
                subtotal=subtotal+model.spec.error_vector(s,true[:,step]).square().sum(-1).mean()
            loss=loss+subtotal/h/len(horizons)
        if not torch.isfinite(loss):raise RuntimeError('动力学损失出现非有限值')
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if update in schedule:
            metrics=evaluate('validation');value=score(metrics)
            report['history'].append(dict(update=update,loss=float(loss.detach()),validation=metrics,selection_error=value))
            if value<best:best,best_update,best_state=value,update,copy.deepcopy(model.state_dict())
            print(f'[固定补训 {update}] 损失={float(loss.detach()):.4f}，验证8步={metrics["8"]["root_position_m"]["mean"]:.4f}m，32步={metrics["32"]["root_position_m"]["mean"]:.4f}m',flush=True)
    report['last']={name:evaluate(name) for name in cohorts}
    model.load_state_dict(best_state)
    report['selected_update']=best_update
    report['after']={name:evaluate(name) for name in cohorts}
    if not all(torch.equal(v,model.term_head.state_dict()[k]) for k,v in head_before.items()):
        raise RuntimeError('失败头意外发生变化')
    # 独立诊断产物，不能误当成可直接续训的完整检查点。
    from dataclasses import asdict
    torch.save(dict(kind='fixed_dynamics_refit_v1',world={k:v.cpu() for k,v in best_state.items()},
                    world_config=asdict(cfg),source=str(args.checkpoint)),args.output/'dynamics_refit.pt')
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('[完成] 策略与价值网络未加载、未更新；失败头保持不变',flush=True)


if __name__=='__main__':main()
