"""Smoke-test production mixed resets and one policy episode; no training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.fallen_bank import load_bank, TRAIN
from common.pre_entrapment import load as load_recovery_bank
from common.birth_checks import check_births as verify_births


def main():
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=16)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--self-contact-audit',action='store_true')
    parser.add_argument('--assist-mode', choices=['configured', 'zero'], default='configured', help='Use the scratch-training course by default; zero is a diagnostic override only')
    parser.add_argument('--recovery-bank', type=Path, default=None)
    parser.add_argument('--steps',type=int,default=1100)
    parser.add_argument('--warmup-steps',type=int,default=20)
    parser.add_argument('--reset-repeats',type=int,default=8)
    parser.add_argument('--real-time' ,action='store_true')
    required=[x for x in parser._actions if x.required]
    for x in required:x.required=False
    try:add_launcher_args(parser)
    finally:
        for x in required:x.required=True
    args,rest=setup_preset_cli(parser);sys.argv=[sys.argv[0]]+rest
    if not args.checkpoint.is_file():parser.error('Checkpoint not found')
    if args.report.exists():parser.error('Report exists; choose a new path')
    if args.num_envs<1 or args.steps<1 or args.reset_repeats<1 or not 0<=args.warmup_steps<args.steps:parser.error('Invalid steps, warmup, resets or env count')
    import x2_style_mimic.tasks
    cfg,agent=resolve_task_config('X2-Prone-MultiMotion-v0','rsl_rl_cfg_entry_point')
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from importlib.metadata import version
    agent=handle_deprecated_rsl_rl_cfg(agent,version('rsl-rl-lib'))
    cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
    if args.device is not None:cfg.sim.device=args.device;agent.device=args.device
    if args.assist_mode == 'zero':
        cfg.assist_curriculum_enabled=False
        cfg.assist_initial_stage=len(cfg.assist_force_levels)-1
        cfg.assist_zero_probability=1.
    cfg.events.reset_from_reference.params['perturb_state_bank'] = str(args.bank.resolve())
    if args.recovery_bank is not None:
        cfg.events.reset_from_reference.params['recovery_state_bank'] = str(args.recovery_bank.resolve())
    probabilities = tuple(cfg.events.reset_from_reference.params['birth_probabilities'])
    assist_peak_force_n=0.
    records=[];complete=False;verified_steps=0;contact_aggregate={};contact_samples=0;contact_audit=None
    with launch_simulation(cfg,args):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_getup_env import X2MultiMotionGetUpEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        torch.manual_seed(args.seed)
        setup_started=time.perf_counter()
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        env=X2MultiMotionGetUpEnv(cfg=cfg)
        try:
            wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        except Exception:
            robot=env.scene['robot']
            data=load_bank(args.bank,joint_names=robot.joint_names)
            q=torch.as_tensor(data['joint_pos'],device=env.device)
            limits=robot.data.joint_pos_limits.torch[0]
            err=(q-q.clamp(limits[:,0],limits[:,1])).abs()
            bad=err.amax(dim=1)>1e-5
            detail=[]
            for j in (err.amax(dim=0)>1e-5).nonzero().flatten().tolist():
                detail.append(dict(joint=robot.joint_names[j],max_violation_rad=float(err[:,j].max()),
                    lower=float(limits[j,0]),upper=float(limits[j,1]),
                    bank_min=float(q[:,j].min()),bank_max=float(q[:,j].max())))
            diagnostic=dict(complete=False,error='reset initialization failed',
                violating_rows=int(bad.sum()),train_violating_rows=int((bad.cpu().numpy() & (data['split']==TRAIN)).sum()),
                joints=detail)
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(diagnostic,indent=2))
            print('[mixed smoke diagnostic] '+json.dumps(diagnostic),flush=True)
            env.close()
            raise
        try:
            runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=agent.device)
            # Same actor/critic normalization; do not restore optimizers, AMP or curriculum.
            runner.load(str(args.checkpoint.resolve()),load_cfg={
                'actor':True,'critic':True,'optimizer':False,'iteration':False,
                'rnd':False,'amp':False,'assist_curriculum':False})
            policy=runner.get_inference_policy(device=env.device)
            obs=wrapped.get_observations()
            bank=load_bank(args.bank,joint_names=env.scene['robot'].joint_names)
            recovery_bank = None
            if probabilities[4] > 0.:
                recovery_bank = load_recovery_bank(
                    cfg.events.reset_from_reference.params['recovery_state_bank'],
                    args.bank, env.scene['robot'].joint_names)
            counts = {i: 0 for i in range(5)}
            forced_counts = {i: 0 for i in range(5)}
            birth_checks = dict(checks=0, automatic_checks=0, max_errors={})
            def check_births(forced=False, env_ids=None):
                result = verify_births(env, bank, recovery_bank, env_ids)
                if env_ids is None:
                    target = forced_counts if forced else counts
                    for kind, value in result['counts'].items(): target[kind] += value
                else:
                    birth_checks['automatic_checks'] += len(env_ids)
                birth_checks['checks'] += 1
                for name, value in result['max_errors'].items():
                    birth_checks['max_errors'][name] = max(birth_checks['max_errors'].get(name, 0.), value)
            for _ in range(args.reset_repeats):
                wrapped.reset(); check_births()
            # Forced coverage is kept separate from natural sampling counts: 1% branches
            # must not make a small diagnostic randomly fail to exercise a branch.
            reset_params = env.event_manager.get_term_cfg('reset_from_reference').params
            try:
                for kind, probability in enumerate(probabilities):
                    if probability <= 0.: continue
                    reset_params['birth_probabilities'] = tuple(float(i == kind) for i in range(5))
                    wrapped.reset(); check_births(forced=True)
                    assert bool((env._getup_birth_types == kind).all()), 'Forced branch coverage failed'
            finally:
                reset_params['birth_probabilities'] = probabilities
            wrapped.reset(); check_births()
            print('[mixed reset verified]', counts, 'forced_coverage', forced_counts, flush=True)
            obs=wrapped.get_observations()
            if args.self_contact_audit:
                from common.self_contact_audit import SelfContactAudit
                contact_audit=SelfContactAudit(env.step_dt)
            setup_seconds=time.perf_counter()-setup_started
            benchmark_started=None
            for step in range(args.steps):
                if step==args.warmup_steps:
                    if torch.cuda.is_available():torch.cuda.synchronize()
                    benchmark_started=time.perf_counter()
                old_rows=env._getup_bank_row_ids.clone()
                old_recovery_rows=env._getup_recovery_row_ids.clone()
                with torch.inference_mode():
                    obs,task_rewards,dones,extras=wrapped.step(policy(obs))
                    progress=extras['getup_progress']
                    if contact_audit is not None:
                        contact_audit.update(env,extras,torch.arange(env.num_envs,device=env.device),torch.ones(env.num_envs,dtype=torch.bool,device=env.device),task_rewards)
                    assist_peak_force_n=max(assist_peak_force_n,float(progress['assist_applied_force_n'].abs().max()))
                    if args.assist_mode == 'zero':
                        assert bool((progress['assist_applied_force_n'].abs()<1e-6).all())
                    assert torch.equal(progress['bank_row_ids'],old_rows)
                    assert torch.equal(progress['recovery_row_ids'],old_recovery_rows)
                    if bool(dones.any()): check_births(env_ids=dones.nonzero().flatten())
                    assert bool(torch.isfinite(env.scene['robot'].data.root_link_pose_w.torch).all())
                    if 'self_contact' in extras:
                        for name,values in extras['self_contact'].items():
                            values=values.detach()
                            if name not in contact_aggregate:
                                contact_aggregate[name]=dict(sum=torch.zeros((),device=values.device),
                                    min=torch.full((),float('inf'),device=values.device),
                                    max=torch.full((),float('-inf'),device=values.device))
                            a=contact_aggregate[name]
                            a['sum'].add_(values.sum())
                            a['min'].copy_(torch.minimum(a['min'],values.min()))
                            a['max'].copy_(torch.maximum(a['max'],values.max()))
                        contact_samples+=env.num_envs
                        monitor=env.scene.sensors['self_contact']
                        if bool(dones.any()):
                            assert bool((monitor.duration[dones.bool()]==0).all()), 'Automatic reset retained duration'
                    verified_steps+=1
                    for e in dones.nonzero().flatten().tolist():
                        records.append(dict(birth_type=int(progress['birth_types'][e]),
                            bank_row=int(progress['bank_row_ids'][e]),
                            recovery_row=int(progress['recovery_row_ids'][e]),
                            succeeded=bool(progress['succeeded'][e])))
                    policy.reset(dones)
                if step%250==0:print(f'[mixed smoke] steps={step} completed={len(records)}',flush=True)
            if torch.cuda.is_available():torch.cuda.synchronize()
            benchmark_seconds=time.perf_counter()-benchmark_started
            if args.steps>=round(cfg.episode_length_s/env.step_dt):
                assert records, 'No completed episode'
            complete=True
        except KeyboardInterrupt:
            print('Interrupted; saving partial acceptance report',flush=True)
        finally:
            report=dict(complete=complete,checkpoint=str(args.checkpoint.resolve()),
                bank=str(args.bank.resolve()),verified_steps=verified_steps,
                performance=dict(setup_seconds=locals().get('setup_seconds'),benchmark_seconds=locals().get('benchmark_seconds'),
                    measured_control_steps=max(0,verified_steps-args.warmup_steps),
                    environment_steps_per_second=(max(0,verified_steps-args.warmup_steps)*env.num_envs/benchmark_seconds) if locals().get('benchmark_seconds') else None,
                    torch_peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None,
                    torch_peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_available() else None,
                    driver_used_mb=(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/2**20 if torch.cuda.is_available() else None,
                    note='Inference smoke throughput, not PPO training; driver memory includes all processes'),
                birth_probabilities=probabilities, assist_mode=args.assist_mode,
                assist_settings=dict(initial_stage=cfg.assist_initial_stage, zero_probability=cfg.assist_zero_probability, curriculum_enabled=cfg.assist_curriculum_enabled),
                assist_peak_force_n=assist_peak_force_n,
                forced_reset_counts=locals().get('forced_counts', {}),
                reset_counts=locals().get('counts',{}),birth_checks=locals().get('birth_checks',{}),
                self_contact_audit=contact_audit.report() if contact_audit is not None else None,
                self_contact={k:dict(min=float(v['min']),max=float(v['max']),mean=float(v['sum'])/max(1,contact_samples)) for k,v in contact_aggregate.items()},episodes=records)
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(report,indent=2))
            print(json.dumps({k:v for k,v in report.items() if k!='episodes'}),flush=True)
            wrapped.close()


if __name__=='__main__':main()
