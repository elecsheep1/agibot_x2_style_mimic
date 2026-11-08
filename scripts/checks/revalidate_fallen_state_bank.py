"""Replay every unchanged fallen state under the current asset, preserving data splits."""
from pathlib import Path
import argparse,copy,json,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.fallen_bank import load_bank,save_bank,asset_hashes,sha256,summary,ARRAY_FIELDS

def main():
 from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=256)
 p.add_argument('--replay-seconds',type=float,default=1.);p.add_argument('--seed',type=int,default=42)
 required=[a for a in p._actions if a.required]
 for a in required:a.required=False
 try:add_launcher_args(p)
 finally:
  for a in required:a.required=True
 args,rest=setup_preset_cli(p);sys.argv=[sys.argv[0]]+rest
 if args.num_envs<1 or not np.isfinite(args.replay_seconds) or args.replay_seconds<1.:p.error('Need positive env count and at least one-second replay')
 report_path=args.output.with_suffix('.revalidation.json')
 if args.output.exists() or report_path.exists():p.error('Output/report exists')
 original=load_bank(args.bank)
 import x2_style_mimic.tasks
 from x2_style_mimic.assets.x2_ultra_cfg import X2_ULTRA_USD
 current_hashes=asset_hashes(X2_ULTRA_USD.parent);old_hashes=original['metadata']['asset_hashes']
 changes=[k for k in set(current_hashes)|set(old_hashes) if current_hashes.get(k)!=old_hashes.get(k)]
 if changes!=['x2_ultra.usda']:raise ValueError(f'Unexpected asset differences: {changes}')
 backup=Path('docs/self_contact_gpu_20261105/implementation/x2_ultra.before_hip_roll_filter.usda')
 if sha256(backup)!=old_hashes['x2_ultra.usda']:raise ValueError('Original asset provenance mismatch')
 cfg,_=resolve_task_config('X2-Prone-MultiMotion-v0','');cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
 if args.device:cfg.sim.device=args.device
 # Standalone scene: omit sensors that are unrelated to validation.
 from isaaclab.sensors import SensorBaseCfg
 for key,value in list(vars(cfg.scene).items()):
  if isinstance(value,SensorBaseCfg):setattr(cfg.scene,key,None)
 report={'complete':False,'source_bank':str(args.bank.resolve()),'source_sha256':sha256(args.bank),'source_asset_hashes':old_hashes,'asset_hashes':current_hashes,'asset_changes':changes,'tested_rows':[],'rejected_rows':[],'criteria':{'seconds':args.replay_seconds,'max_linear_mps':3.,'max_angular_radps':12.,'max_ground_force_n':6000.,'max_penetration_m':.015,'joint_limit_tolerance_rad':.003},'control':'Original state velocities; root XY centered; normal PD holds original joint pose; no assist/policy; every physical substep checked'}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 with launch_simulation(cfg,args):
  import isaaclab.sim as sim_utils
  from isaaclab.scene import InteractiveScene
  from checks.build_fallen_state_bank import Geometry,Contacts
  from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.getup import _write_reference_state
  sim=sim_utils.SimulationContext(cfg.sim);scene=InteractiveScene(cfg.scene);sim.reset();robot=scene['robot'];n=args.num_envs;device=sim.device;dt=float(cfg.sim.dt)
  data=load_bank(args.bank,joint_names=robot.joint_names);geo=Geometry(sim.stage,list(robot.body_names),device);contacts=Contacts(geo,n,dt,device,max(262144,n*32*16))
  ids=torch.arange(n,device=device);limits=robot.data.joint_pos_limits.torch;kept=[]
  try:
   for start in range(0,len(data['root_pose']),n):
    count=min(n,len(data['root_pose'])-start);index=np.minimum(np.arange(start,start+n),len(data['root_pose'])-1)
    tensors={k:torch.as_tensor(data[k][index],device=device,dtype=torch.float32) for k in ['root_pose','root_velocity','joint_pos','joint_vel']};pose=tensors['root_pose'].clone();pose[:,:2]=0.;pose[:,:3]+=scene.env_origins
    scene.reset();robot.permanent_wrench_composer.reset();_write_reference_state(None,ids,robot,pose,tensors['root_velocity'],tensors['joint_pos'],tensors['joint_vel']);robot.set_joint_position_target(tensors['joint_pos']);robot.set_joint_velocity_target(torch.zeros_like(tensors['joint_vel']));sim.forward();scene.update(dt)
    ok=((tensors['joint_pos']>=limits[:,:,0]-.003)&(tensors['joint_pos']<=limits[:,:,1]+.003)).all(-1)
    peak_lin=torch.zeros(n,device=device);peak_ang=peak_lin.clone();peak_ground=peak_lin.clone();minimum_ground=peak_lin.clone();minimum_self=peak_lin.clone();peak_limit=peak_lin.clone()
    for _ in range(round(args.replay_seconds/dt)):
     scene.write_data_to_sim();sim.step(render=False);scene.update(dt)
     pose_now=robot.data.root_link_pose_w.torch;v=robot.data.root_link_vel_w.torch;q=robot.data.joint_pos.torch;gf,sf,gd,sd=contacts.metrics();lin=v[:,:3].norm(dim=-1);ang=v[:,3:].norm(dim=-1)
     peak_lin=torch.maximum(peak_lin,lin);peak_ang=torch.maximum(peak_ang,ang);peak_ground=torch.maximum(peak_ground,gf);minimum_ground=torch.minimum(minimum_ground,gd);minimum_self=torch.minimum(minimum_self,sd)
     violation=torch.maximum((limits[:,:,0]-q).clamp_min(0),(q-limits[:,:,1]).clamp_min(0)).amax(-1);peak_limit=torch.maximum(peak_limit,violation)
     ok&=(torch.isfinite(pose_now).all(-1)&torch.isfinite(v).all(-1)&torch.isfinite(q).all(-1)&(lin<3.)&(ang<12.)&(gf<6000.)&(gd>=-.015)&(sd>=-.015)&(violation<=.003))
    stats=torch.stack([ok.float(),peak_lin,peak_ang,peak_ground,minimum_ground,minimum_self,peak_limit],-1).cpu().numpy()
    for i in range(count):
     source=start+i;values=stats[i];valid=bool(values[0]);report['tested_rows'].append({'source_row':source,'accepted':valid,'peak_linear':float(values[1]),'peak_angular':float(values[2]),'peak_ground_force':float(values[3]),'min_ground_separation':float(values[4]),'min_self_separation':float(values[5]),'peak_joint_violation':float(values[6])})
     if valid:kept.append(source)
     else:report['rejected_rows'].append(source)
    print(f'[bank revalidation] {start+count}/{len(data["root_pose"])} accepted={len(kept)} rejected={len(report["rejected_rows"])}',flush=True)
   if asset_hashes(X2_ULTRA_USD.parent)!=current_hashes:raise RuntimeError('Asset changed during validation')
   rows=[]
   for source in kept:
    row={k:original[k][source].copy() for k in ARRAY_FIELDS};row['quality'][-2:]=[report['tested_rows'][source]['peak_linear'],report['tested_rows'][source]['peak_angular']];rows.append(row)
   kept_array=np.array(kept);source=original['source_type'][kept_array];split=original['split'][kept_array];motion=original['motion_id'][kept_array]
   ready=all(np.any((source==s)&(split==t)) for s in [0,1] for t in [0,1]) and all(np.any((source==0)&(split==0)&(motion==m)) for m in set(original['motion_id'][original['source_type']==0]))
   if not ready:raise RuntimeError('Revalidated bank lost required training coverage')
   report.update(complete=True,training_ready=ready,accepted_count=len(kept),source_row_mapping=kept)
   meta=copy.deepcopy(original['metadata']);meta.update(asset_hashes=current_hashes,revalidation={'source_bank':str(args.bank.resolve()),'source_sha256':report['source_sha256'],'source_asset_hashes':old_hashes,'source_rows':kept,'report_file':str(report_path.resolve()),'criteria':report['criteria'],'control':report['control']})
   meta['collection_report']=dict(meta.get('collection_report',{}),complete=True,training_ready=ready,revalidated=True)
   saved=save_bank(args.output,rows,original['joint_names'],meta);report['summary']=summary(saved);print(json.dumps(report['summary']),flush=True)
  finally:report_path.write_text(json.dumps(report,indent=2))
if __name__=='__main__':
 with torch.inference_mode():main()
