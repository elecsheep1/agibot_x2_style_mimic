"""Read-only GPU PhysX single-joint sweep of X2 near-neighbour contacts."""
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
 from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
 p=argparse.ArgumentParser();p.add_argument("--report",type=Path,default=Path("logs/reference_physics/x2_neighbor_collision_audit/physx_probe.json"));add_launcher_args(p);args,rest=setup_preset_cli(p);sys.argv=[sys.argv[0]]+rest
 import x2_style_mimic.tasks
 cfg,_=resolve_task_config('X2-Prone-MultiMotion-v0','');cfg.scene.num_envs=64;cfg.scene.env_spacing=3.
 if args.device:cfg.sim.device=args.device
 out=args.report;out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError(out)
 result={'complete':False,'cases':[]}
 with launch_simulation(cfg,args):
  import isaaclab.sim as sim_utils
  from isaaclab.scene import InteractiveScene
  from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.getup import _write_reference_state
  sim=sim_utils.SimulationContext(cfg.sim);scene=InteractiveScene(cfg.scene);sim.reset();robot=scene['robot'];sensor=scene.sensors['self_contact'];dt=float(cfg.sim.dt)
  ids=torch.arange(64,device=sim.device);limits=robot.data.joint_pos_limits.torch[0];nj=len(robot.joint_names)
  cases=[('zero',torch.zeros(nj,device=sim.device))]
  for j,n in enumerate(robot.joint_names):
   for v in torch.linspace(float(limits[j,0]),float(limits[j,1]),13,device=sim.device):
    q=torch.zeros(nj,device=sim.device);q[j]=v;cases.append((f'{n}={float(v):.6f}',q))
  try:
   for start in range(0,len(cases),64):
    batch=cases[start:start+64];q=torch.zeros(64,nj,device=sim.device)
    for e,(_,v) in enumerate(batch):q[e]=v
    scene.reset();pose=robot.data.default_root_state.torch[:,:7].clone();pose[:,:3]=scene.env_origins;pose[:,2]+=3.;pose[:,3:]=0.;pose[:,3]=1.
    _write_reference_state(None,ids,robot,pose,torch.zeros(64,6,device=sim.device),q,torch.zeros_like(q));robot.set_joint_position_target(q);sim.forward();scene.update(dt);sensor.begin_control_window()
    # One substep detects initial geometry response before substantial motion.
    scene.write_data_to_sim();sim.step(render=False);scene.update(dt);sensor.end_control_window()
    forces=sensor.peak_force.cpu();actual=robot.data.joint_pos.torch
    for e,(label,_) in enumerate(batch):
     hits=[]
     for k in (forces[e]>1.).nonzero().flatten().tolist():hits.append({'body_a':sensor.body_names[int(sensor._i[k])],'body_b':sensor.body_names[int(sensor._j[k])],'force_n':float(forces[e,k])})
     result['cases'].append({'label':label,'contacts':hits,'max_joint_displacement_rad':float((actual[e]-q[e]).abs().max())})
    print('cases',len(result['cases']),flush=True)
   result['complete']=True
  finally:
   result['note']='64-env one-physics-substep air probe of zero and single-joint sweeps; contact force is initial penetration response, not steady-state force. No production filters changed.';out.write_text(json.dumps(result,indent=2))
if __name__=='__main__':
 with torch.inference_mode():main()
