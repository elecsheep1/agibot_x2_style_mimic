# Archived GPU self-contact probe; run from the project root. See README.md.
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import numpy as np, torch
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
p=argparse.ArgumentParser();add_launcher_args(p);p.set_defaults(visualizer=[]);a,rest=setup_preset_cli(p);sys.argv=[sys.argv[0]]+rest
import x2_style_mimic.tasks
from isaaclab.sensors import ContactSensorCfg
from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
cfg,_=resolve_task_config('Template-X2-Style-Mimic-v0','');cfg.sim.dt=.002;cfg.sim.render_interval=10
if a.device is not None:cfg.sim.device=a.device
cfg.sim.physics=PhysxCfg(solver_type=0);sc=cfg.scene.copy();sc.num_envs=4;sc.env_spacing=3.;sc.contact_forces=None
root='{ENV_REGEX_NS}/Robot/Geometry/base_link'
import xml.etree.ElementTree as ET
paths={}
def walk(body,path):
 name=body.attrib['name']; current=path if name=='pelvis' else path+'/'+name
 if name=='pelvis' or body.find('joint') is not None: paths['base_link' if name=='pelvis' else name]=current
 for child in body.findall('body'):walk(child,current)
walk(ET.parse('data/motions/lafan1_getup_umr/models/x2.xml').getroot().find('worldbody/body'),root)
cfg.scene=sc
with launch_simulation(cfg,a):
 import isaaclab.sim as sim_utils
 from isaaclab.scene import InteractiveScene
 sim=sim_utils.SimulationContext(cfg.sim);scene=InteractiveScene(sc)
 from pxr import Usd,UsdPhysics,PhysxSchema
 stage=sim.stage
 reporters=[];nonrigid=[]
 for prim in stage.Traverse():
  if '/Robot/' in str(prim.GetPath()) and prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
   reporters.append(str(prim.GetPath()))
   if not prim.HasAPI(UsdPhysics.RigidBodyAPI):nonrigid.append(str(prim.GetPath()))
 colliders={}
 for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
  path=str(prim.GetPath())
  if '/env_0/Robot/' in path and prim.HasAPI(UsdPhysics.CollisionAPI):
   parent=prim
   while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):parent=parent.GetParent()
   if parent:colliders.setdefault(str(parent.GetPath()),[]).append(path)
 Path('logs/reference_physics/self_contact_archive_replay_stage.json').write_text(json.dumps({'all_reporters':reporters,'nonrigid_reporters':nonrigid,'colliders':colliders},indent=2))
 sim.set_setting('/physics/disableContactProcessing', False);sim.reset();sim.set_setting('/physics/disableContactProcessing', False);r=scene['robot'];dt=cfg.sim.dt
 from isaaclab_physx.physics import PhysxManager as SimulationManager
 import warp as wp
 pv=SimulationManager.get_physics_sim_view()
 body_names=list(paths)
 sources=[paths[n].replace('{ENV_REGEX_NS}',f'/World/envs/env_{e}') for e in range(4) for n in body_names]
 filters=[[paths[m].replace('{ENV_REGEX_NS}',f'/World/envs/env_{e}') for m in body_names if m!=n] for e in range(4) for n in body_names]
 cv=pv.create_rigid_contact_view(sources,filter_patterns=filters,max_contact_data_count=4096)
 print('EXPLICIT_VIEW',cv.sensor_count,cv.filter_count,flush=True)
 report={'scenarios':[]}
 q0=r.data.default_joint_pos.torch.clone();limits=r.data.joint_pos_limits.torch;gen=torch.Generator(device=sim.device).manual_seed(320)
 def run(name,q,h,steps):
  pose=r.data.default_root_state.torch[:,:7].clone();pose[:,:3]+=scene.env_origins;pose[:,2]=scene.env_origins[:,2]+h
  r.write_root_link_pose_to_sim_index(root_pose=pose);r.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros((4,6),device=sim.device));r.write_joint_state_to_sim_index(position=q,velocity=torch.zeros_like(q));r.set_joint_position_target_index(target=q);scene.reset();sim.forward();scene.update(dt)
  maxima={};pairs={}
  for step in range(steps):
   scene.write_data_to_sim();sim.step(render=False);scene.update(dt)
   f=wp.to_torch(cv.get_contact_force_matrix(dt=dt)).reshape(4,-1,3).norm(dim=-1).amax(dim=-1)
   for e in range(4):maxima[e]=max(maxima.get(e,0.),float(f[e]))
  report['scenarios'].append({'name':name,'max_self_force_n':maxima,'joint_pos':q[0].cpu().tolist()})
 run('neutral_air',q0,3.,20)
 run('neutral_ground',q0,.68,250)
 old=json.loads((Path(__file__).resolve().parent / 'collision_pose.json').read_text())
 case=next(c for c in old['scenarios'] if c['name'].startswith('air_arm'))
 order=[old['joint_names'].index(n) for n in r.joint_names]
 collision=torch.tensor(np.array(case['joint_pos'])[order],device=sim.device,dtype=torch.float32)
 q=q0.clone();q[[0,2]]=collision;run('air_collision_env0_2',q,3.,12)
 q=q0.clone();q[[1,3]]=collision;run('air_collision_env1_3',q,3.,12)
 report['joint_names']=r.joint_names;out=Path('logs/reference_physics/self_contact_archive_replay_gpu.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
