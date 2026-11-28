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
    parser.add_argument(
        "--state-loss-weights",
        type=float,
        nargs=6,
        default=None,
        metavar=(
            "ROOT_POS",
            "ROOT_ROT",
            "ROOT_LINEAR",
            "ROOT_ANGULAR",
            "JOINT_POS",
            "JOINT_VEL",
        ),
        help="覆盖六个状态误差分块的训练权重",
    )
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
    if args.state_loss_weights is not None:
        cfg.state_loss_weights = tuple(
            args.state_loss_weights
        )
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
        state_loss_weights=list(
            cfg.state_loss_weights
        ),
        dynamics_parameters=sum(p.numel() for p in model.dynamics.parameters()),
        fixed_cohorts={name:dict(starts=c['starts'].cpu().tolist(),envs=c['envs'].cpu().tolist()) for name,c in cohorts.items()},
        notes=['只更新 dynamics，使用原有归一化状态误差损失；优化器重新初始化。',
               '训练、验证、测试按环境隔离；固定窗口起点唯一，但回合内窗口可以重叠。',
               '仅按验证集 8 步误差选择模型，测试集不参与选择。'],history=[])
    args.output.mkdir(parents=True,exist_ok=True)
    report['before']={name:evaluate(name) for name in cohorts}
    @torch.no_grad()
    def selection_scores():
        cohort = cohorts["validation"]

        prediction = variant_rollout(
            model,
            cohort,
            "baseline",
        )

        errors = {
            horizon: float(
                model.spec.normalized_sq_error(
                    prediction[:, horizon - 1],
                    cohort["following"][:, horizon - 1],
                ).mean()
            )
            for horizon in (8, 16, 32)
        }

        return {
            "8step": errors[8],
            "long_rollout": (
                errors[8]
                + errors[16]
                + errors[32]
            ) / 3.0,
        }


    best_models = {
        name: {
            "score": value,
            "update": 0,
            "state": copy.deepcopy(
                model.state_dict()
            ),
        }
        for name, value in selection_scores().items()
    }
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
                squared_error = model.spec.error_vector(
                    s,
                    true[:, step],
                ).square()

                subtotal = (
                    subtotal
                    + model.state_prediction_loss(
                        squared_error
                    ).mean()
                )
            loss=loss+subtotal/h/len(horizons)
        if not torch.isfinite(loss):raise RuntimeError('动力学损失出现非有限值')
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if update in schedule:
            metrics = evaluate("validation")
            scores = selection_scores()

            report["history"].append(
                {
                    "update": update,
                    "loss": float(loss.detach()),
                    "validation": metrics,
                    "selection_scores": scores,
                }
            )

            for name, value in scores.items():
                if value < best_models[name]["score"]:
                    best_models[name] = {
                        "score": value,
                        "update": update,
                        "state": copy.deepcopy(
                            model.state_dict()
                        ),
                    }

            position_8 = metrics["8"][
                "root_position_m"
            ]["mean"]

            position_32 = metrics["32"][
                "root_position_m"
            ]["mean"]

            print(
                f"[固定补训 {update}] "
                f"损失={float(loss.detach()):.4f}，"
                f"验证8步={position_8:.4f}m，"
                f"32步={position_32:.4f}m",
                flush=True,
            )
    from dataclasses import asdict

    report["last"] = {
        name: evaluate(name)
        for name in cohorts
    }

    report["selected"] = {}

    for selection_name, selected in best_models.items():
        model.load_state_dict(
            selected["state"],
            strict=True,
        )

        if not all(
            torch.equal(
                value,
                model.term_head.state_dict()[key],
            )
            for key, value in head_before.items()
        ):
            raise RuntimeError(
                "失败头意外发生变化"
            )

        report["selected"][selection_name] = {
            "update": selected["update"],
            "score": selected["score"],
            "metrics": {
                name: evaluate(name)
                for name in cohorts
            },
        }

        torch.save(
            {
                "kind": "fixed_dynamics_refit_v1",
                "world": {
                    key: value.detach().cpu()
                    for key, value
                    in selected["state"].items()
                },
                "world_config": asdict(cfg),
                "source": str(args.checkpoint),
                "selection": selection_name,
                "selected_update": selected["update"],
            },
            args.output
            / f"dynamics_best_{selection_name}.pt",
        )

    report["notes"] = [
        "仅更新动力学，策略与价值网络未加载。",
        "失败头保持冻结。",
        "固定窗口起点唯一，同一回合内窗口可能重叠。",
        "分别按验证集8步误差与8/16/32步平均误差选择模型。",
        "选择指标不使用训练分块权重。",
        "测试集不参与模型选择。",
    ]

    (args.output / "report.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "[完成] 已分别保存短期与长时最优模型；"
        "策略、价值网络和失败头未更新",
        flush=True,
    )


if __name__=='__main__':main()
