"""Read-only USD convex-hull neighbour screening using MuJoCo distance queries.
No asset edits; penetration is a geometry candidate, not proof of unreasonable contact.
"""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from pxr import Usd,UsdGeom,UsdPhysics,Gf
ROOT=Path(__file__).resolve().parents[2]
USD=ROOT/'source/x2_style_mimic/x2_style_mimic/assets/robots/x2_ultra/x2_ultra/x2_ultra.usda'
OUT=ROOT/'logs/reference_physics/x2_neighbor_collision_audit';OUT.mkdir(parents=True,exist_ok=True)
s=Usd.Stage.Open(str(USD));cache=UsdGeom.XformCache();bodies={p.GetName():p for p in s.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
xml=ET.parse(ROOT/'data/motions/lafan1_getup_umr/models/x2.xml');x=xml.getroot();x.find('.//worldbody/body').set('name','base_link')
for par in x.iter():
 for e in list(par):
  if e.tag in ['geom','mesh','contact','sensor','actuator','keyframe']:par.remove(e)
asset=x.find('asset');nodes={b.get('name'):b for b in x.iter('body')};shape_info=[]
for p in Usd.PrimRange.Stage(s,Usd.TraverseInstanceProxies()):
 if not p.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get():continue
 b=p
 while b and not b.HasAPI(UsdPhysics.RigidBodyAPI):b=b.GetParent()
 name=b.GetName();points=np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(),dtype=float)
 T=np.array(cache.GetLocalToWorldTransform(p)*cache.GetLocalToWorldTransform(b).GetInverse())
 points=points@T[:3,:3]+T[3,:3];h=ConvexHull(points);v=points[h.vertices]
 mesh='usd_'+str(len(shape_info));ET.SubElement(asset,'mesh',name=mesh,vertex=' '.join(map(str,v.ravel())))
 ET.SubElement(nodes[name],'geom',name=mesh,type='mesh',mesh=mesh,rgba='.6 .7 .8 1',contype='1',conaffinity='1',group='0')
 shape_info.append(dict(body=name,path=str(p.GetPath()),hull_vertices=len(v),approximation=str(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get())))
modelpath=OUT/'usd_hulls.xml';xml.write(modelpath);m=mujoco.MjModel.from_xml_path(str(modelpath));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
# Validate kinematic zero poses against USD, root-relative.
base=cache.GetLocalToWorldTransform(bodies['base_link']).GetInverse();errors={}
for n,p in bodies.items():
 bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n);T=np.array(cache.GetLocalToWorldTransform(p)*base).T
 errors[n]=dict(position_m=float(np.max(abs(T[:3,3]-(d.xpos[bid]-d.xpos[1])))),rotation_matrix=float(np.max(abs(T[:3,:3]-d.xmat[bid].reshape(3,3)))))
assert max(e['position_m'] for e in errors.values())<.0001
filtered=set()
for p in s.Traverse():
 if p.HasAPI(UsdPhysics.FilteredPairsAPI):
  for t in UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets():filtered.add(tuple(sorted([p.GetName(),str(t).split('/')[-1]])))
def chain(b):
 a=[]
 while b:a.append(b);b=int(m.body_parentid[b])
 return a
def distance(a,b):
 ca=chain(a);cb=chain(b);return min(ca.index(k)+cb.index(k) for k in set(ca)&set(cb))
pairs=[]
for a in range(m.ngeom):
 for b in range(a+1,m.ngeom):
  ba=int(m.geom_bodyid[a]);bb=int(m.geom_bodyid[b]);dist=distance(ba,bb)
  if dist>3:continue
  na=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,ba);nb=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bb)
  pairs.append(dict(a=a,b=b,body_a=na,body_b=nb,tree_distance=dist,explicitly_filtered=tuple(sorted([na,nb])) in filtered,parent_child=dist==1,min_distance_m=1.,overlap_samples=0,overlap_gt1mm_samples=0,first_case=None,worst_case=None,zero_distance_m=None))
poses=[('zero',m.qpos0.copy())];rng=np.random.default_rng(42)
for j in range(1,m.njnt):
 adr=int(m.jnt_qposadr[j]);lo,hi=m.jnt_range[j]
 for v in np.linspace(lo,hi,13):
  q=m.qpos0.copy();q[adr]=v;poses.append((f'{mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)}={v:.5f}',q))
for k in range(500):
 q=m.qpos0.copy()
 for j in range(1,m.njnt):q[m.jnt_qposadr[j]]=rng.uniform(*m.jnt_range[j])
 poses.append((f'random_limits_{k}',q))
# Real policy trajectories, every 0.5 seconds; no new physics.
for fn in ['model15199_v2_row603_detailed64.json','model15199_v1_row1348_detailed64.json']:
 z=json.loads((ROOT/'logs/reference_physics'/fn).read_text())
 for k,tr in enumerate(z['traces']):
  if k%5:continue
  q=m.qpos0.copy();q[:7]=tr['root_pose']
  for n,v in zip(z['joint_names'],tr['joint_pos']):q[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)]]=v
  poses.append((f'{fn}:t={tr["t"]}',q))
for side in ['left','right']:
 j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'{side}_hip_roll_joint')
 for v in np.linspace(-.235,.1,68)*(1 if side=='left' else -1):
  q=m.qpos0.copy();q[m.jnt_qposadr[j]]=v;poses.append((f'fine_{side}_hip_roll={v:.5f}',q))
manifest=ROOT/'data/motions/prone_getup2/motions.json'
for entry in json.loads(manifest.read_text())['motions']:
 z=np.load(manifest.parent/entry['file']);names=z['joint_names'].tolist()
 for k in range(0,len(z['joint_pos']),10):
  q=m.qpos0.copy()
  for n,v in zip(names,z['joint_pos'][k]):
   j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)
   if j>=0:q[m.jnt_qposadr[j]]=v
  poses.append((f'reference_{entry["file"]}:frame={k}',q))
worst={};seg=np.zeros(6);pose_hits=[]
for k,(label,q) in enumerate(poses):
 d.qpos[:]=q;mujoco.mj_forward(m,d)
 for p in pairs:
  value=float(mujoco.mj_geomDistance(m,d,p['a'],p['b'],.1,seg))
  if k==0:p['zero_distance_m']=value
  p['overlap_samples']+=int(value<-.00001);p['overlap_gt1mm_samples']+=int(value<-.001)
  if value<p['min_distance_m']:
   p['min_distance_m']=value;p['worst_case']=label;worst[f'{p["a"]}_{p["b"]}']=q.tolist()
  if value<-.001 and p['first_case'] is None:p['first_case']=label
  if value<-.00001 and not p['parent_child'] and not p['explicitly_filtered']:
   pose_hits.append(dict(case=label,body_a=p['body_a'],body_b=p['body_b'],distance_m=value))
 if k%250==0:print('poses',k,'/',len(poses),flush=True)
report=dict(usd=str(USD),shapes=shape_info,kinematic_zero_errors=errors,explicit_filters=[list(t) for t in sorted(filtered)],pose_count=len(poses),pose_hits=pose_hits,pairs=pairs,worst_qpos=worst,note='USD authored convex hulls queried in MuJoCo; PhysX cooking/contact offsets differ. Full-limit random poses are screening, not normal poses. Parent-child pairs generally disabled by articulation. No collision filters changed.')
(OUT/'report.json').write_text(json.dumps(report,indent=2))
for p in pairs:
 if not p['parent_child'] and not p['explicitly_filtered'] and p['overlap_gt1mm_samples']:print(p['body_a'],p['body_b'],p['tree_distance'],round(1000*p['min_distance_m'],2),p['overlap_gt1mm_samples'],p['worst_case'])
