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
from common.fallen_bank import load_bank, reset_from_bank_for_eval, HOLDOUT, TRAIN


def main():
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--split',choices=['holdout','train'],default='holdout')
    parser.add_argument('--source',choices=['all','near','fall'],default='all')
    parser.add_argument('--num-envs',type=int,default=16)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--real-time',action='store_true')
    required=[x for x in parser._actions if x.required]
    for x in required:x.required=False
    try:add_launcher_args(parser)
    finally:
        for x in required:x.required=True
    args,rest=setup_preset_cli(parser);sys.argv=[sys.argv[0]]+rest
    if not args.checkpoint.is_file():parser.error('Checkpoint not found')
    if args.report.exists():parser.error('Report exists; choose a new path')
    if args.num_envs<1:parser.error('num-envs must be positive')
    import x2_style_mimic.tasks
    cfg,agent=resolve_task_config('X2-Prone-MultiMotion-v0','rsl_rl_cfg_entry_point')
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from importlib.metadata import version
    agent=handle_deprecated_rsl_rl_cfg(agent,version('rsl-rl-lib'))
    cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
    if args.device is not None:cfg.sim.device=args.device;agent.device=args.device
    cfg.assist_curriculum_enabled=False
    cfg.assist_initial_stage=len(cfg.assist_force_levels)-1
    cfg.assist_zero_probability=1.
    cfg.events.reset_from_reference.params.update(
        initial_frame_probability=0.5, perturb_initial_probability=0.4,
        perturb_state_bank=str(args.bank.resolve()))
    records=[];complete=False;verified_steps=0
    with launch_simulation(cfg,args):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_getup_env import X2MultiMotionGetUpEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        torch.manual_seed(args.seed)
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
            counts={i:0 for i in range(4)}
            birth_checks=dict(checks=0,bank_samples=0,max_local_xy_m=0.0,
                              max_height_error_m=0.0,max_orientation_error=0.0,
                              max_joint_pos_error_rad=0.0,max_root_velocity_error=0.0,
                              max_joint_velocity_error=0.0)
            def check_births():
                types=env._getup_birth_types
                rows=env._getup_bank_row_ids
                for i in counts: counts[i]+=int((types==i).sum())
                mask=rows>=0
                idx=rows[mask].cpu().numpy()
                assert (bank['split'][idx]==TRAIN).all(), 'Holdout sampled'
                assert bool(((types==1)|(types==3)).eq(mask).all())
                assert (bank['source_type'][idx]==(types[mask]==3).long().cpu().numpy()).all()
                assert bool((env._getup_reference_motion_ids[types==3]==-1).all())
                birth_checks['checks']+=1
                if bool(mask.any()):
                    robot=env.scene['robot']
                    pose=robot.data.root_link_pose_w.torch[mask].clone()
                    pose[:,:3]-=env.scene.env_origins[mask]
                    def expected(key):
                        return torch.as_tensor(bank[key][idx],device=env.device,dtype=torch.float32)
                    errors=dict(max_local_xy_m=pose[:,:2].abs().max(),
                        max_height_error_m=(pose[:,2]-expected('root_pose')[:,2]).abs().max(),
                        max_orientation_error=torch.minimum(
                            (pose[:,3:]-expected('root_pose')[:,3:]).abs().amax(-1),
                            (pose[:,3:]+expected('root_pose')[:,3:]).abs().amax(-1)).max(),
                        max_joint_pos_error_rad=(robot.data.joint_pos.torch[mask]-expected('joint_pos')).abs().max(),
                        max_root_velocity_error=(robot.data.root_link_vel_w.torch[mask]-expected('root_velocity')).abs().max(),
                        max_joint_velocity_error=(robot.data.joint_vel.torch[mask]-expected('joint_vel')).abs().max())
                    for name,error in errors.items():
                        birth_checks[name]=max(birth_checks[name],float(error))
                        assert float(error)<1e-4, f'Birth state mismatch: {name}={float(error)}'
                    birth_checks['bank_samples']+=int(mask.sum())
            # Several real resets exercise all branches without stepping a reference pose.
            for _ in range(8):
                wrapped.reset()
                check_births()
            assert all(counts.values()), counts
            obs=wrapped.get_observations()
            for step in range(1100):
                old_rows=env._getup_bank_row_ids.clone()
                with torch.inference_mode():
                    obs,_,dones,extras=wrapped.step(policy(obs))
                    progress=extras['getup_progress']
                    assert bool((progress['assist_applied_force_n'].abs()<1e-6).all())
                    assert torch.equal(progress['bank_row_ids'],old_rows)
                    assert bool(torch.isfinite(env.scene['robot'].data.root_link_pose_w.torch).all())
                    verified_steps+=1
                    for e in dones.nonzero().flatten().tolist():
                        records.append(dict(birth_type=int(progress['birth_types'][e]),
                            bank_row=int(progress['bank_row_ids'][e]),
                            succeeded=bool(progress['succeeded'][e])))
                    policy.reset(dones)
                if step%250==0:print(f'[mixed smoke] steps={step} completed={len(records)}',flush=True)
            assert records, 'No completed episode'
            complete=True
        except KeyboardInterrupt:
            print('Interrupted; saving partial acceptance report',flush=True)
        finally:
            report=dict(complete=complete,checkpoint=str(args.checkpoint.resolve()),
                bank=str(args.bank.resolve()),verified_steps=verified_steps,
                reset_counts=locals().get('counts',{}),birth_checks=locals().get('birth_checks',{}),episodes=records)
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(report,indent=2))
            print(json.dumps({k:v for k,v in report.items() if k!='episodes'}),flush=True)
            wrapped.close()


if __name__=='__main__':main()
