"""Render recorded Isaac get-up poses with the local X2 visual model; no physics stepping."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[2]
model=mujoco.MjModel.from_xml_path(str(ROOT/'data/motions/lafan1_getup_umr/models/x2.xml'))
data=mujoco.MjData(model)
renderer=mujoco.Renderer(model,height=480,width=640)
configs=[('603','model15199_v2_row603_detailed64.json',['right_ankle_roll_link','left_knee_link']),('1348','model15199_v1_row1348_detailed64.json',['left_knee_link','left_shoulder_roll_link','left_shoulder_yaw_link'])]
sheet=Image.new('RGB',(1920,960))
for row,(state,file,highlight) in enumerate(configs):
 d=json.loads((ROOT/'logs/reference_physics'/file).read_text());tr=min(d['traces'],key=lambda x:abs(x['t']-5))
 data.qpos[:]=model.qpos0;data.qpos[:3]=tr['root_pose'][:3];data.qpos[3:7]=np.asarray(tr['root_pose'])[[6,3,4,5]]
 for name,q in zip(d['joint_names'],tr['joint_pos']):
  j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
  if j<0:raise ValueError(name)
  data.qpos[model.jnt_qposadr[j]]=q
 mujoco.mj_forward(model,data)
 for view,az in enumerate([30,150,270]):
  cam=mujoco.MjvCamera();cam.lookat[:]=data.qpos[:3]+[0,0,.12];cam.distance=1.65;cam.elevation=-30;cam.azimuth=az
  renderer.update_scene(data,camera=cam);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
  for g in renderer.scene.geoms[:renderer.scene.ngeom]:
   if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0:
    b=int(model.geom_bodyid[g.objid]);name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b)
    if name in highlight:g.rgba[:]=[1,.18,.08,1]
  if not np.any(model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE):
   k=renderer.scene.ngeom;mujoco.mjv_initGeom(renderer.scene.geoms[k],mujoco.mjtGeom.mjGEOM_PLANE,np.array([2.,2.,.001]),np.array([0.,0.,0.]),np.eye(3).ravel(),np.array([.3,.33,.36,1],np.float32));renderer.scene.ngeom+=1
  im=Image.fromarray(renderer.render().copy());dr=ImageDraw.Draw(im);dr.rectangle((0,0,640,47),fill='black');dr.text((10,7),f'State {state} | recorded t={tr["t"]:.1f}s | view {view+1}',fill='white');dr.text((10,26),'Red: persistent-contact bodies | kinematic replay',fill='white');sheet.paste(im,(view*640,row*480))
out=ROOT/'logs/reference_physics/stalled_getup_603_1348.png';sheet.save(out);renderer.close();print(out)
