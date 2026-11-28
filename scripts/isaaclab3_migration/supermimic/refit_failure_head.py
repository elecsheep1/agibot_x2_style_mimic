"""冻结控制策略与动力学，按完整回合划分数据，仅补训失败分类头；纯 CPU。"""
import argparse
import copy
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from analyze_failure_head import features


def split_episodes(done,seed=43):
    """只保留完整回合；训练／验证／测试分别为约 70%／15%／15%。"""
    if done.ndim!=2:raise ValueError('终止标记必须为 [T,N]')
    steps,envs=done.shape
    local=torch.cat((torch.zeros(1,envs,dtype=torch.long),done.long().cumsum(0)[:-1]),0)
    groups=local+torch.arange(envs)[None]*(steps+1)
    complete=groups[done].unique()
    if len(complete)<10:raise ValueError('完整回合不足 10 个，无法可靠划分三组数据')
    generator=torch.Generator().manual_seed(seed)
    order=complete[torch.randperm(len(complete),generator=generator)]
    held=max(1,int(round(len(order)*.15)))
    members={'train':order[2*held:],'validation':order[:held],'test':order[held:2*held]}
    indices={name:torch.isin(groups,ids).reshape(-1).nonzero().flatten() for name,ids in members.items()}
    return indices,{name:ids.tolist() for name,ids in members.items()},dict(
        complete_episodes=len(complete),discarded_partial_samples=int((~torch.isin(groups,complete)).sum()))


@torch.no_grad()
def evaluate(head,x,y,prior):
    logits=head(x)
    probability=logits.sigmoid()
    positive=y.bool();negative=~positive
    predicted=logits>0
    def conditional(mask):
        if not mask.any():return None
        p=probability[mask]
        return dict(mean_probability=float(p.mean()),p10=float(torch.quantile(p,.1)),
                    p50=float(torch.quantile(p,.5)),p90=float(torch.quantile(p,.9)))
    tp=int((predicted&positive).sum());fn=int((~predicted&positive).sum())
    fp=int((predicted&negative).sum());tn=int((~predicted&negative).sum())
    baseline=F.binary_cross_entropy(torch.full_like(y,prior),y)
    return dict(samples=len(y),positive_count=int(positive.sum()),negative_count=int(negative.sum()),
        tp=tp,fn=fn,fp=fp,tn=tn,recall=tp/(tp+fn) if tp+fn else None,
        false_positive_rate=fp/(fp+tn) if fp+tn else None,
        precision=tp/(tp+fp) if tp+fp else None,
        bce=float(F.binary_cross_entropy_with_logits(logits,y)),constant_train_prior_bce=float(baseline),
        brier=float((probability-y).square().mean()),positive=conditional(positive),negative=conditional(negative))


def refit(head,x,y,indices,updates=2000,batch_size=256,lr=2e-4,seed=43,log_interval=250):
    if updates<1 or batch_size<1 or lr<=0:raise ValueError('更新次数、批量和学习率必须为正数')
    for name,index in indices.items():
        if not len(index) or y[index].min()==y[index].max():
            raise ValueError(f'{name} 缺少失败或正常样本，请采集更多完整回合')
    generator=torch.Generator().manual_seed(seed)
    train,validation,test=(indices[name] for name in ('train','validation','test'))
    prior=float(y[train].mean())
    before={name:evaluate(head,x[idx],y[idx],prior) for name,idx in indices.items()}
    optimizer=torch.optim.Adam(head.parameters(),lr=lr)
    best_bce=before['validation']['bce'];best_update=0
    best_state=copy.deepcopy(head.state_dict())
    history=[]
    for update in range(1,updates+1):
        head.train()
        chosen=train[torch.randint(len(train),(batch_size,),generator=generator)]
        loss=F.binary_cross_entropy_with_logits(head(x[chosen]),y[chosen])
        if not torch.isfinite(loss):raise FloatingPointError('失败头损失出现非有限值')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if update==1 or update%log_interval==0 or update==updates:
            head.eval()
            metrics=evaluate(head,x[validation],y[validation],prior)
            history.append(dict(update=update,validation=metrics))
            if metrics['bce']<best_bce:
                best_bce,best_update=metrics['bce'],update
                best_state=copy.deepcopy(head.state_dict())
            print(f'[补训 {update}/{updates}] 留出 BCE={metrics["bce"]:.6f}，'
                  f'失败召回={metrics["recall"]:.1%}，正常误杀={metrics["false_positive_rate"]:.1%}',flush=True)
    head.load_state_dict(best_state,strict=True);head.eval()
    after={name:evaluate(head,x[idx],y[idx],prior) for name,idx in indices.items()}
    return dict(train_positive_prior=prior,best_update=best_update,before=before,after=after,history=history)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=2000)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--seed',type=int,default=43)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    from x2_style_mimic.algorithms.supermimic_tracking.world_model import TerminationHead
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    data=torch.load(args.data,map_location='cpu',weights_only=False)['data']
    root=[]
    import numpy as np
    for path in checkpoint['contract']['files']:
        if not path.endswith('.npz'):continue
        with np.load(path,allow_pickle=False) as motion:
            p=torch.tensor(motion['root_pos_w'],dtype=torch.float32)
            p[:,2]-=float(json.loads(str(motion['metadata_json'].item()))['ground_z'])
            xyzw=torch.tensor(motion['root_quat_xyzw'],dtype=torch.float32)
            q=torch.cat((xyzw[:,-1:],xyzw[:,:3]),-1)
            q=q/q.norm(dim=-1,keepdim=True)
            root.append(torch.cat((p,q),-1))
    reference=torch.cat(root)[data['frame']]
    x=features(data['next_state'],reference).reshape(-1,2).detach()
    y=data['terminated'].reshape(-1).float()
    cfg=checkpoint['config']['world_model']
    expected=(x[:,0].abs()>cfg['height_threshold']) | (x[:,1].abs()>cfg['tilt_threshold'])
    if not torch.equal(expected,y.bool()):raise ValueError('失败特征与真实终止标签不一致，先核对数据口径')
    indices,groups,split_info=split_episodes(data['terminated']|data['truncated'],args.seed)
    head=TerminationHead(tuple(cfg['term_hidden']),cfg['height_threshold'],cfg['tilt_threshold'])
    prefix='term_head.'
    head.load_state_dict({key[len(prefix):]:value for key,value in checkpoint['world'].items() if key.startswith(prefix)},strict=True)
    head.requires_grad_(True)
    report=refit(head,x,y,indices,args.updates,args.batch_size,args.lr,args.seed)
    report.update(source_checkpoint=str(args.checkpoint.resolve()),source_iteration=checkpoint['iteration'],
        source_data=str(args.data.resolve()),seed=args.seed,updates=args.updates,batch_size=args.batch_size,lr=args.lr,
        split=split_info,episode_groups=groups,
        notes=['原 Actor、Critic、动力学与训练检查点均未修改；输出只含失败头。',
               '选择检查点只使用 validation BCE，test 不参与模型选择。',
               '普通 BCE、自然比例采样，不加 pos_weight、不调整分类阈值。',
               '这是旧策略数据上的分类验证，不代表新有界策略或新倒地分布已经验证。'])
    args.output.mkdir(parents=True,exist_ok=True)
    torch.save(dict(kind='x2_tracking_failure_head_refit_v1',head=head.state_dict(),head_cfg=cfg,
                    source_checkpoint=report['source_checkpoint'],best_update=report['best_update']),args.output/'failure_head.pt')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'best_update':report['best_update'],'test_before':report['before']['test'],
                      'test_after':report['after']['test']},indent=2,ensure_ascii=False),flush=True)
    print(f'[报告] {args.output/"report.json"}',flush=True)


if __name__=='__main__':main()
