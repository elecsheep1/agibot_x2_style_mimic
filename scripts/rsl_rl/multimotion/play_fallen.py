"""Evaluate an unchanged policy on a fixed fallen-bank split, once per state."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
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
    parser.add_argument('--rows', type=int, nargs='+', default=None)
    parser.add_argument('--center-xy', action='store_true', help='Diagnostic only: recenter root XY; not comparable to original acceptance')
    parser.add_argument('--trace', action='store_true', help='Save pre-step state every 0.1s for targeted diagnosis')
    parser.add_argument('--trace-rows',type=int,nargs='+',default=None)
    parser.add_argument('--detailed-trace',action='store_true',help='Script-only contact/target/estimated-PD diagnostics')
    parser.add_argument('--self-contact-audit',action='store_true')
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
    bank=load_bank(args.bank)
    mask=bank['split']==(HOLDOUT if args.split=='holdout' else TRAIN)
    if args.source!='all':mask&=bank['source_type']==(0 if args.source=='near' else 1)
    selected=np.flatnonzero(mask)
    if args.rows is not None:
        if len(set(args.rows)) != len(args.rows) or not np.isin(args.rows, selected).all():
            parser.error('rows must be unique and belong to requested split/source')
        selected=np.asarray(args.rows)
    if not len(selected):parser.error('No states in requested split/source')
    import x2_style_mimic.tasks
    cfg,agent=resolve_task_config('X2-Prone-MultiMotion-v0','rsl_rl_cfg_entry_point')
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from importlib.metadata import version
    agent=handle_deprecated_rsl_rl_cfg(agent,version('rsl-rl-lib'))
    cfg.scene.num_envs=min(args.num_envs,len(selected));cfg.seed=args.seed
    if args.device is not None:cfg.sim.device=args.device;agent.device=args.device
    cfg.assist_curriculum_enabled=False
    cfg.assist_initial_stage=len(cfg.assist_force_levels)-1
    cfg.assist_zero_probability=1.
    params=dict(cfg.events.reset_from_reference.params)
    params.update(bank_path=str(args.bank.resolve()),bank_split=args.split,bank_source=args.source,bank_rows=args.rows,bank_center_xy=args.center_xy)
    cfg.events.reset_from_reference.func=reset_from_bank_for_eval
    cfg.events.reset_from_reference.params=params
    records=[];complete=False;verified_steps=0;traces=[];contact_audit=None
    with launch_simulation(cfg,args):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_getup_env import X2MultiMotionGetUpEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        torch.manual_seed(args.seed)
        env=X2MultiMotionGetUpEnv(cfg=cfg)
        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        try:
            runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=agent.device)
            # Same actor/critic normalization; do not restore optimizers, AMP or curriculum.
            runner.load(str(args.checkpoint.resolve()),load_cfg={
                'actor':True,'critic':True,'optimizer':False,'iteration':False,
                'rnd':False,'amp':False,'assist_curriculum':False})
            policy=runner.get_inference_policy(device=env.device)
            obs=wrapped.get_observations()
            contacts=None
            if args.detailed_trace:
                import warp as wp
                from checks.build_fallen_state_bank import Geometry, Contacts
                geometry=Geometry(env.sim.stage,list(env.scene['robot'].body_names),env.device)
                contacts=Contacts(geometry,env.num_envs,float(cfg.sim.dt),env.device,262144)
            if args.self_contact_audit:
                from common.self_contact_audit import SelfContactAudit
                contact_audit=SelfContactAudit(env.step_dt)
            seen=set();required_hold=float(cfg.terminations.progress.params['hold_time'])
            while len(seen)<len(selected):
                started=time.monotonic()
                index=env._fallen_eval_index.clone();serial=env._fallen_eval_serial.clone()
                with torch.inference_mode():
                    if (args.trace or args.detailed_trace) and verified_steps % 5 == 0:
                        robot=env.scene['robot']
                        pose=robot.data.root_link_pose_w.torch
                        vel=robot.data.root_link_vel_w.torch
                        bodies=robot.data.body_link_pose_w.torch
                        if contacts is not None:
                            contact_metrics=contacts.metrics()
                            ground=wp.to_torch(contacts.ground_view.get_contact_force_matrix(contacts.dt)).reshape(env.num_envs,contacts.b,1,3)
                            self_force=wp.to_torch(contacts.self_view.get_contact_force_matrix(contacts.dt)).reshape(env.num_envs,contacts.b,contacts.b,3).norm(dim=-1)
                        for e in range(env.num_envs):
                            if int(serial[e])>=len(selected) or int(index[e]) in seen:continue
                            if args.trace_rows is not None and int(index[e]) not in args.trace_rows:continue
                            traces.append(dict(bank_row=int(index[e]),t=float(env.episode_length_buf[e]*env.step_dt),
                                root_pose=(torch.cat((pose[e,:3]-env.scene.env_origins[e],pose[e,3:]))).cpu().tolist(),
                                root_velocity=vel[e].cpu().tolist(),
                                gravity=robot.data.projected_gravity_b.torch[e].cpu().tolist(),
                                joint_pos=robot.data.joint_pos.torch[e].cpu().tolist(),
                                body_pose=(torch.cat((bodies[e,:,:3]-env.scene.env_origins[e],bodies[e,:,3:]),dim=-1)).cpu().tolist()))
                            if contacts is not None:
                                term=env.action_manager.get_term('joint_pos')
                                requested=term.raw_actions*term._scale+term._offset
                                q=robot.data.joint_pos.torch[e]; qd=robot.data.joint_vel.torch[e]
                                target=robot.data.joint_pos_target.torch[e]
                                pd=robot.data.joint_stiffness.torch[e]*(target-q)+robot.data.joint_damping.torch[e]*(robot.data.joint_vel_target.torch[e]-qd)
                                flat=self_force[e].flatten();top=torch.topk(flat,min(8,len(flat)))
                                traces[-1].update(
                                    joint_velocity=qd.cpu().tolist(),joint_target=target.cpu().tolist(),
                                    requested_target=requested[e].cpu().tolist(),action_joint_ids=list(range(robot.num_joints)) if isinstance(term._joint_ids,slice) else list(term._joint_ids),
                                    estimated_pd_torque_nm=pd.cpu().tolist(),
                                    effort_limit_nm=robot.data.joint_effort_limits.torch[e].cpu().tolist(),
                                    ground_force_by_body_w=ground[e,:,0].cpu().tolist(),
                                    self_force_pairs=[dict(body_a=robot.body_names[int(k)//contacts.b],body_b=robot.body_names[int(k)%contacts.b],force_n=float(v)) for v,k in zip(top.values,top.indices) if float(v)>1.],
                                    ground_min_separation_m=float(contact_metrics[2][e]),self_min_separation_m=float(contact_metrics[3][e]))
                    actions=policy(obs)
                    obs,task_rewards,dones,extras=wrapped.step(actions)
                    progress=extras['getup_progress']
                    if contact_audit is not None:
                        valid=(serial<len(selected)) & torch.tensor([int(x) not in seen for x in index],device=env.device)
                        contact_audit.update(env,extras,index,valid,task_rewards)
                    if bool((progress['assist_applied_force_n'].abs()>1e-6).any()):
                        raise RuntimeError('Nonzero assist during fallen-bank acceptance')
                    verified_steps+=1
                    for e in dones.nonzero().flatten().tolist():
                        if int(serial[e])>=len(selected):continue
                        row_id=int(index[e])
                        if row_id in seen:raise RuntimeError('Repeated bank row counted twice')
                        seen.add(row_id)
                        records.append(dict(terminated=bool(env.termination_manager.terminated[e]),timed_out=bool(env.termination_manager.time_outs[e]),bank_row=row_id,source_type=int(bank['source_type'][row_id]),
                            motion_id=int(bank['motion_id'][row_id]),family_id=int(bank['family_id'][row_id]),
                            succeeded=bool(progress['succeeded'][e]),
                            stable_at_end=bool(progress['hold_seconds'][e]+1e-6>=required_hold),
                            max_hold_seconds=float(progress['max_hold_seconds'][e]),
                            final_hold_seconds=float(progress['hold_seconds'][e]),
                            first_success_seconds=float(progress['first_success_time'][e])))
                    policy.reset(dones)
                if dones.any():
                    print(f'[fallen eval] {len(seen)}/{len(selected)} states',flush=True)
                if args.real_time:
                    delay=env.step_dt-(time.monotonic()-started)
                    if delay>0:time.sleep(delay)
            complete=True
        except KeyboardInterrupt:
            print('Interrupted; saving partial acceptance report',flush=True)
        finally:
            def stats(rows):
                return dict(count=len(rows),success_rate=sum(x['succeeded'] for x in rows)/len(rows) if rows else None,
                    stable_at_end_rate=sum(x['stable_at_end'] for x in rows)/len(rows) if rows else None)
            report=dict(complete=complete,target_states=len(selected),checkpoint=str(args.checkpoint.resolve()),
                bank=str(args.bank.resolve()),split=args.split,source=args.source,seed=args.seed,
                zero_assist_verified=verified_steps>0,verified_steps=verified_steps,required_hold_seconds=float(cfg.terminations.progress.params['hold_time']),
                overall=stats(records),by_source={name:stats([r for r in records if r['source_type']==i])
                    for i,name in [(0,'near_reference'),(1,'simulated_fall')]},num_envs=env.num_envs,detailed_trace=args.detailed_trace,torque_note='PD estimate only, not actual drive measurement; self contacts subject to asset collision filters',self_contact_audit=contact_audit.report() if contact_audit is not None else None,center_xy=args.center_xy,episodes=records,traces=traces,joint_names=env.scene['robot'].joint_names,body_names=env.scene['robot'].body_names)
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(report,indent=2))
            print(json.dumps(report['overall']),flush=True)
            wrapped.close()


if __name__=='__main__':main()
