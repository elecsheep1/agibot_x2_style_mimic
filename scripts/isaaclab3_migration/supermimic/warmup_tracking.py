"""冻结 Actor／Critic 的真实采集、世界模型预热和失败头留出验证。"""
from dataclasses import dataclass
from pathlib import Path
import copy
import json
import math
import torch
from refit_failure_head import refit,evaluate


@dataclass
class PretrainingCfg:
    head_init: str = ''
    collection_steps: int = 256
    world_update_calls: int = 250
    head_updates: int = 2000
    head_batch_size: int = 256
    head_lr: float = 2e-4
    evaluation_windows: int = 64
    evaluation_interval: int = 50
    holdout_envs_per_group: int = 4
    maximum_position_8_m: float = .05
    maximum_rotation_8_deg: float = 10.
    maximum_position_32_m: float = .15
    maximum_rotation_32_deg: float = 30.
    minimum_validation_positives: int = 3
    minimum_recall: float = .8
    maximum_false_positive_rate: float = .02
    joint_if_ready: bool = True

    def validate(self,num_envs,replay_capacity):
        if min(self.collection_steps,self.world_update_calls,self.head_updates,self.head_batch_size,
               self.evaluation_windows,self.evaluation_interval,self.holdout_envs_per_group)<1:
            raise ValueError('预热步数、更新次数与批量必须为正数')
        if self.collection_steps<33 or replay_capacity<2*self.collection_steps+3:
            raise ValueError('预热至少采集 33 步，回放容量需至少为采集步数的两倍加 3 行')
        if num_envs<=2*self.holdout_envs_per_group:
            raise ValueError('训练环境与两组留出环境必须均非空')
        if not 0<=self.minimum_recall<=1 or not 0<=self.maximum_false_positive_rate<=1:
            raise ValueError('失败头验收比例必须处于 [0,1]')


class ReplayEnvironmentView:
    """只从指定环境取窗口，保留同一回放库的状态／动作／标签布局。"""
    def __init__(self,buffer,env_ids):
        self.buffer=buffer
        self.env_ids=torch.as_tensor(env_ids,device=buffer.device,dtype=torch.long)

    def candidates(self,horizon):
        mask=self.buffer.window_mask(horizon)
        allowed=torch.zeros(self.buffer.E,device=self.buffer.device,dtype=torch.bool)
        allowed[self.env_ids]=True
        return (mask&allowed[:,None]).nonzero()

    def gather(self,positions,horizon):
        from importlib import import_module
        b=self.buffer
        WindowBatch=import_module(type(b).__module__).WindowBatch
        env,start=positions.unbind(-1)
        times=(start[None]+torch.arange(horizon+1,device=b.device)[:,None])%b.T
        ee=env[None].expand_as(times)
        return WindowBatch(states=b.s[ee,times],a_exec=b.a_exec[ee[:-1],times[:-1]],
            a_clean=b.a[ee[:-1],times[:-1]],xi=b.xi[env,start],reward=b.r[ee[:-1],times[:-1]],
            done=b.done[ee[:-1],times[:-1]],frame=b.frame[ee[:-1],times[:-1]])

    def sample_windows(self,n,horizon):
        options=self.candidates(horizon)
        if not len(options):raise ValueError('指定训练环境中没有连续监督窗口')
        positions=options[torch.randint(len(options),(n,),device=self.buffer.device)]
        return self.gather(positions,horizon)

    def fixed_windows(self,n,horizon):
        options=self.candidates(horizon)
        if not len(options):return None,0
        chosen=torch.randperm(len(options),device=self.buffer.device)[:n]
        # 固定窗口不重复起点；同一回合内仍可能重叠。
        return self.gather(options[chosen],horizon),len(options)


@torch.no_grad()
def rollout_metrics(model,batch):
    from x2_style_mimic.algorithms.supermimic_tracking.rotations import quat_box_minus
    if batch is None:return None
    state=batch.states[0]
    for action in batch.a_exec:state=model.step(state,action,batch.xi)
    true=batch.states[-1]
    return dict(windows=state.shape[0],normalized_error=float(model.spec.normalized_sq_error(state,true).mean()),
        root_position_m=float((state[:,:3]-true[:,:3]).norm(dim=-1).mean()),
        root_rotation_deg=float(quat_box_minus(state[:,3:7],true[:,3:7]).norm(dim=-1).mean()*180/torch.pi),
        joint_position_rmse_rad=float((state[:,7:38]-true[:,7:38]).square().mean(-1).sqrt().mean()))


def readiness(world_validation,head_validation,cfg):
    reasons=[]
    for horizon,position,rotation in ((8,cfg.maximum_position_8_m,cfg.maximum_rotation_8_deg),
                                      (32,cfg.maximum_position_32_m,cfg.maximum_rotation_32_deg)):
        metrics=world_validation.get(str(horizon))
        if metrics is None:reasons.append(f'缺少 {horizon} 步留出窗口');continue
        if not math.isfinite(metrics['root_position_m']) or metrics['root_position_m']>position:reasons.append(f'{horizon} 步根位置误差超出预热门槛')
        if not math.isfinite(metrics['root_rotation_deg']) or metrics['root_rotation_deg']>rotation:reasons.append(f'{horizon} 步根旋转误差超出预热门槛')
    if head_validation['positive_count']<cfg.minimum_validation_positives:
        reasons.append('留出失败正例不足')
    if head_validation['recall'] is None or head_validation['recall']<cfg.minimum_recall:
        reasons.append('留出失败召回率不足')
    if head_validation['false_positive_rate'] is None or head_validation['false_positive_rate']>cfg.maximum_false_positive_rate:
        reasons.append('留出正常误杀率不达标')
    if not math.isfinite(head_validation['bce']) or head_validation['bce']>=head_validation['constant_train_prior_bce']:
        reasons.append('留出 BCE 未优于训练先验基线')
    return not reasons,reasons


def prepare(training,cfg,project):
    from x2_style_mimic.algorithms.supermimic_tracking.gradients import frozen_parameters
    from x2_style_mimic.algorithms.supermimic_tracking.training import validate_fk
    env,model=training.env,training.world
    cfg.validate(env.num_envs,training.cfg.replay_capacity)
    if training.cfg.resume:raise ValueError('此验证入口从新有界策略开始，不接收旧训练续训')
    initial=torch.load(Path(project)/cfg.head_init,map_location='cpu',weights_only=False)
    if initial.get('kind')!='x2_tracking_failure_head_refit_v1':raise ValueError('失败头初始化文件类型不正确')
    for key in ('term_hidden','height_threshold','tilt_threshold'):
        expected=getattr(training.cfg.world_model,key)
        actual=initial['head_cfg'][key]
        if isinstance(expected,tuple):actual=tuple(actual)
        if actual!=expected:raise ValueError(f'失败头初始化配置不匹配：{key}')
    model.term_head.load_state_dict(initial['head'],strict=True)
    before_policy={key:value.detach().cpu().clone() for key,value in training.policy.state_dict().items()}
    before_critic={key:value.detach().cpu().clone() for key,value in training.critic.state_dict().items()}
    order=torch.randperm(env.num_envs,device=env.device)
    held=cfg.holdout_envs_per_group
    groups={'validation':order[:held],'test':order[held:2*held],'train':order[2*held:]}
    report={'environment_groups':{name:ids.tolist() for name,ids in groups.items()},
            'notes':['预热期间训练环境与留出环境完全分离。',
                     '固定开环窗口起点不重复，同一回合内窗口可能重叠。',
                     '失败头采用按环境隔离的真实单步数据，包含未结束片段。',
                     '预热门槛是本次试验的工程检查条件，不是论文统一收敛标准。',
                     '进入联合训练后将恢复使用全部环境，预热留出集不再是联合训练留出集。']}
    features,labels=[],[]
    bound_violations=0;target_count=0
    with frozen_parameters(training.policy,training.critic):
        obs,_=env.reset();validate_fk(env)
        starts=torch.ones(env.num_envs,device=env.device,dtype=torch.bool)
        action_term=env.action_manager.get_term('joint_pos')
        with torch.no_grad():
            for step in range(cfg.collection_steps):
                action=training.policy(obs['policy'],obs['goal'])
                obs,_,_,_,extras=env.step(action)
                transition=extras['tracking_transition']
                starts=training.record(transition,starts)
                ref=training.lib.ref(transition['frame'])
                f=model.heads.termination_features(transition['next_state'][:,:3],transition['next_state'][:,3:7],ref)
                expected=(f[:,0].abs()>model.cfg.height_threshold)|(f[:,1].abs()>model.cfg.tilt_threshold)
                if not torch.equal(expected,transition['terminated']):raise RuntimeError('真实失败特征与终止标签不一致')
                features.append(f.cpu());labels.append(transition['terminated'].cpu().float())
                target=transition['executed']*action_term._scale+action_term._offset
                bound_violations+=int(((target<action_term.target_lower-1e-6)|(target>action_term.target_upper+1e-6)).sum())
                target_count+=target.numel()
                if (step+1)%64==0:print(f'[真实预热采集] {step+1}/{cfg.collection_steps}',flush=True)
        report['pd_target_limit_fraction']=bound_violations/target_count
        views={name:ReplayEnvironmentView(training.buffer,ids) for name,ids in groups.items()}
        batches={name:{} for name in ('validation','test')}
        coverage={name:{} for name in batches}
        for name in batches:
            for horizon in (1,8,32):
                batch,count=views[name].fixed_windows(cfg.evaluation_windows,horizon)
                batches[name][str(horizon)]=batch;coverage[name][str(horizon)]=count
        def check(name):return {h:rollout_metrics(model,batch) for h,batch in batches[name].items()}
        report['eligible_windows']=coverage
        report['world_before']={name:check(name) for name in batches}
        if batches['validation']['8'] is None:raise RuntimeError('预热没有产生 8 步验证窗口')
        best_error=report['world_before']['validation']['8']['normalized_error'];best_update=0
        best_world=copy.deepcopy(model.state_dict());best_optimizer=copy.deepcopy(training.world_trainer.opt.state_dict())
        history=[]
        with frozen_parameters(model.term_head):
            for update in range(1,cfg.world_update_calls+1):
                metrics=training.world_trainer.update(views['train'])
                if update==1 or update%cfg.evaluation_interval==0 or update==cfg.world_update_calls:
                    validation=check('validation')
                    history.append(dict(update=update,validation=validation,training=metrics))
                    error=validation['8']['normalized_error']
                    if error<best_error:
                        best_error,best_update=error,update
                        best_world=copy.deepcopy(model.state_dict())
                        best_optimizer=copy.deepcopy(training.world_trainer.opt.state_dict())
                    print(f'[模型预热 {update}/{cfg.world_update_calls}] 损失={metrics["loss"]:.4f}，'
                          f'留出 8 步位置={validation["8"]["root_position_m"]:.4f} m',flush=True)
        model.load_state_dict(best_world,strict=True);training.world_trainer.opt.load_state_dict(best_optimizer)
        report['world_best_update']=best_update
        report['world_history']=history
        report['world_after']={name:check(name) for name in batches}
        x=torch.stack(features).reshape(-1,2);y=torch.stack(labels).reshape(-1)
        environment_grid=torch.arange(env.num_envs)[None].expand(cfg.collection_steps,-1).reshape(-1)
        indices={name:torch.isin(environment_grid,ids.cpu()).nonzero().flatten() for name,ids in groups.items()}
        head=copy.deepcopy(model.term_head).cpu()
        prior=float(y[indices['train']].mean().clamp(1e-6,1-1e-6))
        has_classes=all(bool(y[idx].min()!=y[idx].max()) for idx in indices.values())
        if has_classes:
            head_report=refit(head,x,y,indices,updates=cfg.head_updates,batch_size=cfg.head_batch_size,
                             lr=cfg.head_lr,seed=training.cfg.seed,log_interval=500)
            model.term_head.load_state_dict(head.state_dict(),strict=True)
        else:
            metrics={name:evaluate(head,x[idx],y[idx],prior) for name,idx in indices.items()}
            head_report={'status':'缺少分类样本，未更新失败头','before':metrics,'after':metrics}
        report['head']=head_report
    report['actor_unchanged']=all(torch.equal(value,training.policy.state_dict()[key].cpu()) for key,value in before_policy.items())
    report['critic_unchanged']=all(torch.equal(value,training.critic.state_dict()[key].cpu()) for key,value in before_critic.items())
    if not report['actor_unchanged'] or not report['critic_unchanged']:raise RuntimeError('预热期间策略或价值网络发生变化')
    ready,reasons=readiness(report['world_after']['validation'],report['head']['after']['validation'],cfg)
    if bound_violations:ready=False;reasons.append('实际 PD 目标仍越限')
    report['ready_for_joint_training'],report['reasons']=ready,reasons
    (training.output/'warmup_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    training.save('warmup_only.pt')
    # 后续 run() 会手动 reset；显式切断回放的回合编号，避免把两个状态拼为连续转移。
    training.buffer.end_episode(torch.arange(env.num_envs,device=env.device))
    print(f'[预热结果] 允许联合更新={ready}，原因={reasons}',flush=True)
    return report
