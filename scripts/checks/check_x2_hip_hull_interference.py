"""Compare authored USD triangle solids against convex hull overlap at hip adduction."""
from pathlib import Path
import json
import numpy as np
import trimesh,mujoco
from pxr import Usd,UsdGeom,UsdPhysics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'logs/reference_physics/x2_neighbor_collision_audit';stage=Usd.Stage.Open(str(ROOT/'source/x2_style_mimic/x2_style_mimic/assets/robots/x2_ultra/x2_ultra/x2_ultra.usda'));cache=UsdGeom.XformCache();meshes={};meta={}
for p in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
 if not p.HasAPI(UsdPhysics.CollisionAPI):continue
 b=p
 while b and not b.HasAPI(UsdPhysics.RigidBodyAPI):b=b.GetParent()
 n=b.GetName()
 if n not in ['base_link','left_hip_roll_link','right_hip_roll_link']:continue
 g=UsdGeom.Mesh(p);v=np.array(g.GetPointsAttr().Get(),float);counts=np.array(g.GetFaceVertexCountsAttr().Get());indices=np.array(g.GetFaceVertexIndicesAttr().Get());assert np.all(counts==3)
 T=np.array(cache.GetLocalToWorldTransform(p)*cache.GetLocalToWorldTransform(b).GetInverse());v=v@T[:3,:3]+T[3,:3]
 mesh=trimesh.Trimesh(vertices=v,faces=indices.reshape(-1,3),process=True);meshes[n]=mesh
 meta[n]=dict(vertices=len(mesh.vertices),faces=len(mesh.faces),watertight=bool(mesh.is_watertight),winding_consistent=bool(mesh.is_winding_consistent),volume_m3=float(mesh.volume),hull_volume_m3=float(mesh.convex_hull.volume),body_local_bounds=mesh.bounds.tolist())
 print(n,meta[n],flush=True)
m=mujoco.MjModel.from_xml_path(str(OUT/'usd_hulls.xml'));d=mujoco.MjData(m);rng=np.random.default_rng(42);result={'meshes':meta,'cases':[]};saved=[]
for side,sign in [('left',-1),('right',1)]:
 name=f'{side}_hip_roll_link';bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name);baseid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'base_link');j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'{side}_hip_roll_joint')
 for angle in [0,.1,.235]:
  d.qpos[:]=m.qpos0;d.qpos[m.jnt_qposadr[j]]=sign*angle;mujoco.mj_forward(m,d)
  world=[]
  for n,idx in [('base_link',baseid),(name,bid)]:
   mm=meshes[n].copy();T=np.eye(4);T[:3,:3]=d.xmat[idx].reshape(3,3);T[:3,3]=d.xpos[idx]-d.xpos[baseid];mm.apply_transform(T);world.append(mm)
  a,b=world;ha,hb=a.convex_hull,b.convex_hull;bounds=np.array([np.maximum(ha.bounds[0],hb.bounds[0]),np.minimum(ha.bounds[1],hb.bounds[1])]);entry={'side':side,'hip_roll_rad':sign*angle}
  if np.any(bounds[1]<=bounds[0]):entry['aabb_disjoint']=True;result['cases'].append(entry);continue
  # Uniform volume samples confined to overlap AABB; convex inequalities first.
  points=rng.uniform(bounds[0],bounds[1],(30000,3));inside=np.ones(len(points),bool)
  for hull in [ha,hb]:
   equations=hull.facets_normal if False else None
   from scipy.spatial import ConvexHull
   eq=ConvexHull(hull.vertices).equations
   inside&=np.all(points@eq[:,:3].T+eq[:,3]<=1e-9,axis=1)
  overlap=points[inside];entry['sample_count']=len(points);entry['hull_overlap_samples']=len(overlap);entry['estimated_hull_overlap_mm3']=float(np.prod(bounds[1]-bounds[0])*inside.mean()*1e9)
  if len(overlap):
   # Bound memory for ray queries. Closed-mesh parity is not reliable on nonwatertight solids.
   test=overlap[:3000];ia=a.contains(test);ib=b.contains(test)
   entry.update(tested_overlap_points=len(test),inside_base_raw=int(ia.sum()),inside_hip_raw=int(ib.sum()),inside_both_raw=int((ia&ib).sum()),inside_neither_raw=int((~ia&~ib).sum()),raw_test_valid=bool(a.is_watertight and b.is_watertight and a.is_winding_consistent and b.is_winding_consistent))
   np.savez(OUT/f'{side}_hip_{angle}_overlap_points.npz',points=test,inside_base=ia,inside_hip=ib)
  if angle==.235:
   a.export(OUT/f'{side}_base_raw.ply');b.export(OUT/f'{side}_hip_raw.ply');ha.export(OUT/f'{side}_base_hull.ply');hb.export(OUT/f'{side}_hip_hull.ply')
  result['cases'].append(entry);print(entry,flush=True)
result['note']='Monte Carlo finite sampling of authored closed triangle meshes vs convex hull. No raw overlap samples is not proof of zero surface intersection. USD is a simplified asset, not CAD or measured hardware.'
(OUT/'hip_hull_interference.json').write_text(json.dumps(result,indent=2))
