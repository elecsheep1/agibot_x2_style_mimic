"""Exercise the production self-contact sensor on separated ground/collision cases."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    required=[x for x in p._actions if x.required]
    for x in required:x.required=False
    try:add_launcher_args(p)
    finally:
        for x in required:x.required=True
    args,rest=setup_preset_cli(p);sys.argv=[sys.argv[0]]+rest
    if args.report.exists():p.error('Report already exists')
    import x2_style_mimic.tasks
    cfg,_=resolve_task_config('X2-Prone-MultiMotion-v0','')
    cfg.scene.num_envs=4;cfg.scene.env_spacing=3.
    if args.device is not None:cfg.sim.device=args.device
    result=dict(complete=False,scenarios=[])
    with launch_simulation(cfg,args):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.getup import _write_reference_state
        from checks.build_fallen_state_bank import Geometry,Contacts
        sim=sim_utils.SimulationContext(cfg.sim)
        scene=InteractiveScene(cfg.scene);sim.reset()
        robot=scene['robot'];monitor=scene.sensors['self_contact'];dt=float(cfg.sim.dt)
        direct=Contacts(Geometry(sim.stage,list(robot.body_names),sim.device),4,dt,sim.device,262144)
        ids=torch.arange(4,device=sim.device);q0=robot.data.default_joint_pos.torch.clone()
        zero=torch.zeros_like(q0)
        def run(name,q,height,steps):
            scene.reset()
            pose=robot.data.default_root_state.torch[:,:7].clone()
            pose[:,:2]=0.;pose[:,2]=height;pose[:,:3]+=scene.env_origins
            _write_reference_state(None,ids,robot,pose,torch.zeros(4,6,device=sim.device),q,zero)
            sim.forward();scene.update(dt)
            assert all(bool((v==0).all()) for v in monitor.snapshot().values())
            peak=torch.zeros(4,device=sim.device);ground_peak=peak.clone();pairs={}
            for step in range(steps):
                if step%10==0:monitor.begin_control_window()
                scene.write_data_to_sim();sim.step(render=False);scene.update(dt)
                ground_peak=torch.maximum(ground_peak,direct.metrics()[0])
                if step%10==9:
                    monitor.end_control_window();m=monitor.snapshot()
                    assert bool((m['updates_per_control_step']==10).all())
                    peak=torch.maximum(peak,m['peak_pair_force_n'])
                    for e in range(4):
                        active=(monitor.peak_force[e]>10).nonzero().flatten()
                        for k in active.tolist():
                            key=f'{monitor.body_names[int(monitor._i[k])]} / {monitor.body_names[int(monitor._j[k])]}'
                            pairs.setdefault(str(e),{})[key]=max(pairs.get(str(e),{}).get(key,0),float(monitor.peak_force[e,k]))
            out=dict(name=name,self_peak_n=peak.cpu().tolist(),ground_peak_n=ground_peak.cpu().tolist(),pairs=pairs)
            result['scenarios'].append(out);print('[self-contact probe] '+json.dumps(out),flush=True)
            # Check masked reset: resetting env0 must not clear env1's state.
            before=monitor.duration[1].clone();scene.reset(torch.tensor([0],device=sim.device))
            assert bool((monitor.duration[0]==0).all())
            assert torch.equal(monitor.duration[1],before)
            return peak,ground_peak
        try:
            f,_=run('neutral_air',q0,3.,20);assert bool((f<1.).all()),'Neutral air false positive'
            f,g=run('neutral_ground',q0,.68,400)
            assert bool((g>20.).all()),'No real ground contact in neutral-ground test'
            assert bool((f<1.).all()),'Ground-only false positive'
            archive=Path('docs/self_contact_gpu_20261105/collision_pose.json')
            old=json.loads(archive.read_text());case=next(c for c in old['scenarios'] if c['name'].startswith('air_arm'))
            order=[old['joint_names'].index(n) for n in robot.joint_names]
            collision=torch.tensor(case['joint_pos'],device=sim.device,dtype=torch.float32)[order]
            for target in [[0,2],[1,3]]:
                q=q0.clone();q[target]=collision
                f,g=run('air_collision_env'+'_'.join(map(str,target)),q,3.,20)
                rest=[i for i in range(4) if i not in target]
                assert bool((f[target]>10.).all()),'Known collision not detected'
                assert bool((f[rest]<1.).all()),'Collision leaked across environments'
                assert bool((g<1.).all()),'Air collision touched ground'
            result['complete']=True
        finally:
            result['body_names']=monitor.body_names
            result['pair_count_per_env']=int(monitor._i.numel())
            result['note']='Checks neutral ground vs known arm-body collision and environment isolation; not every pair validated'
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    with torch.inference_mode():main()
