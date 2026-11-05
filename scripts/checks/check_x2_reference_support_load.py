"""Compare ground-load responses of backed-up and repaired reference frames.

Free root, current task PD, reference initial velocities, no policy or assist.
Short open-loop responses are diagnostic and are not contact-force labels.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.x2_motion_utils import load_motion
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
import x2_style_mimic.tasks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair-dir',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    required=[x for x in parser._actions if x.required]
    for x in required:x.required=False
    add_launcher_args(parser)
    for x in required:x.required=True
    args,rest=setup_preset_cli(parser);sys.argv=[sys.argv[0]]+rest
    if args.report.exists():raise FileExistsError(args.report)
    times={'x2_getup_139_16_umr_v1_50hz.npz':[4.,4.54,5.02,6.],
           'x2_getup_139_17_umr_v1_50hz.npz':[4.8,5.62,6.14,7.76]}
    cases=[]
    for variant in ['before','candidates']:
        for name,seconds in times.items():
            motion=load_motion(args.repair_dir/variant/name)
            for second in seconds:cases.append((variant,name,second,motion,int(round(second/.02))))
    cfg,_=resolve_task_config('X2-Prone-MultiMotion-v0','');cfg.scene.num_envs=len(cases)
    if args.device is not None:cfg.sim.device=args.device
    with launch_simulation(cfg,args):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        sim=sim_utils.SimulationContext(cfg.sim)
        try:
            scene=InteractiveScene(cfg.scene);sim.reset();robot=scene['robot'];device=sim.device
            orders=[[z['joint_names'].tolist().index(n) for n in robot.joint_names] for _,_,_,z,_ in cases]
            root=np.stack([np.r_[z['root_pos_w'][f],z['root_quat_xyzw'][f]] for _,_,_,z,f in cases]);pose=torch.tensor(root,dtype=torch.float32,device=device);pose[:,:3]+=scene.env_origins;pose[:,2]+=.002
            q=torch.tensor(np.stack([z['joint_pos'][f,o] for (_,_,_,z,f),o in zip(cases,orders)]),device=device)
            qv=torch.tensor(np.stack([z['joint_vel'][f,o] for (_,_,_,z,f),o in zip(cases,orders)]),device=device)
            rv=torch.tensor(np.stack([np.r_[z['root_lin_vel_w'][f],z['root_ang_vel_w'][f]] for _,_,_,z,f in cases]),device=device)
            robot.write_root_link_pose_to_sim_index(root_pose=pose)
            offset=robot.data.body_com_pose_b.torch[:,0,:3];quat=pose[:,3:7];cross=2*torch.cross(quat[:,:3],offset,dim=-1);off_w=offset+quat[:,3:4]*cross+torch.cross(quat[:,:3],cross,dim=-1);comvel=rv.clone();comvel[:,:3]+=torch.cross(rv[:,3:6],off_w,dim=-1)
            robot.write_root_com_velocity_to_sim_index(root_velocity=comvel);robot.write_joint_state_to_sim_index(position=q,velocity=qv);robot.set_joint_position_target_index(target=q);sim.forward();scene.update(cfg.sim.dt)
            sensors=['left_foot_ground_contact','right_foot_ground_contact','left_hand_ground_contact','right_hand_ground_contact','head_yaw_ground_contact','head_pitch_ground_contact'];records=[]
            for step in range(150):
                control_step=step//10
                target=torch.tensor(np.stack([z['joint_pos'][min(f+control_step,len(z['time_s'])-1),o] for (_,_,_,z,f),o in zip(cases,orders)]),device=device)
                robot.set_joint_position_target_index(target=target);scene.write_data_to_sim();sim.step();scene.update(cfg.sim.dt)
                if (step+1)%5==0:
                    forces=torch.stack([scene.sensors[n].data.force_matrix_w.torch[:,0,0,2] for n in sensors],-1)
                    height=robot.data.root_pos_w.torch[:,2]-scene.env_origins[:,2];reference_h=torch.tensor([z['root_pos_w'][min(f+control_step,len(z['time_s'])-1),2]+.002 for _,_,_,z,f in cases],device=device)
                    tracking=(robot.data.joint_pos.torch-target).square().mean(-1).sqrt();records.append(torch.cat((height[:,None],(height-reference_h)[:,None],tracking[:,None],forces),-1).cpu().numpy())
            array=np.asarray(records);rows=[]
            for i,(variant,name,second,z,f) in enumerate(cases):
                r={'variant':variant,'motion':name,'reference_time_s':second,'initial_frame':f,'samples':array[:,i].tolist()}
                for label,sl in [('early_004to012',slice(3,12)),('later_012to030',slice(11,None))]:
                    a=array[sl,i];fz=np.maximum(a[:,3:7],0);den=fz.sum(-1);r[label]={'feet_fz_mean_N':fz[:,:2].mean(0).tolist(),'hands_fz_mean_N':fz[:,2:4].mean(0).tolist(),'hands_fraction_of_observed_hand_foot_fz':float((fz[:,2:4].sum(-1)/np.maximum(den,1e-6)).mean()),'head_fz_mean_N':np.maximum(a[:,7:9],0).mean(0).tolist(),'height_drift_mean_m':float(a[:,1].mean()),'joint_tracking_rms_mean_rad':float(a[:,2].mean())}
                rows.append(r)
            args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps({'definition':'Free-root 0.3s reference-follow response, current task PD, reference initial velocities, no policy/assist. Forces reflect drifted simulated trajectory, not exact reference force labels. Contact-force shares include only observed hands and feet, not other body support.','sensor_names':sensors,'sample_columns':['root_height','height_minus_reference','joint_tracking_rms',*sensors],'sample_dt':.01,'cases':rows},indent=2));print('SUPPORT_REPORT',args.report,flush=True)
        finally:sim.clear_instance()


if __name__=='__main__':main()
