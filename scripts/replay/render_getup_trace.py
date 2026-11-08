"""Render recorded policy poses for diagnosis; no physics replay or mesh edits."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--rows',type=int,nargs='+',required=True);p.add_argument('--times',type=float,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert len(a.rows)==len(a.times)
r=Path(__file__).resolve().parents[2];z=json.loads(a.report.read_text());m=mujoco.MjModel.from_xml_path(str(r/'data/motions/lafan1_getup_umr/models/x2.xml'));d=mujoco.MjData(m);renderer=mujoco.Renderer(m,height=480,width=640);sheet=Image.new('RGB',(1920,480*len(a.rows)))
for row,(state,time) in enumerate(zip(a.rows,a.times)):
 candidates=[t for t in z['traces'] if t['bank_row']==state];tr=min(candidates,key=lambda t:abs(t['t']-time));d.qpos[:]=m.qpos0;d.qpos[:3]=tr['root_pose'][:3];d.qpos[3:7]=np.asarray(tr['root_pose'])[[6,3,4,5]]
 for name,q in zip(z['joint_names'],tr['joint_pos']):
  j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);assert j>=0;d.qpos[m.jnt_qposadr[j]]=q
 mujoco.mj_forward(m,d)
 for view,az in enumerate([30,150,270]):
  c=mujoco.MjvCamera();c.lookat[:]=d.qpos[:3]+[0,0,.12];c.distance=1.65;c.elevation=-30;c.azimuth=az;renderer.update_scene(d,camera=c);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
  k=renderer.scene.ngeom;mujoco.mjv_initGeom(renderer.scene.geoms[k],mujoco.mjtGeom.mjGEOM_PLANE,np.array([2.,2.,.001]),np.array([d.qpos[0],d.qpos[1],0.]),np.eye(3).ravel(),np.array([.3,.33,.36,1],np.float32));renderer.scene.ngeom+=1
  im=Image.fromarray(renderer.render().copy());dr=ImageDraw.Draw(im);dr.rectangle((0,0,640,43),fill='black');dr.text((8,6),f'Bank row {state} | t={tr["t"]:.2f}s | view {view+1}',fill='white');dr.text((8,24),'Recorded policy state; offline visual mesh rendering',fill='white');sheet.paste(im,(640*view,480*row))
a.output.parent.mkdir(parents=True,exist_ok=True);sheet.save(a.output);renderer.close();print(a.output)
