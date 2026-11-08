import json
from pathlib import Path
import trimesh,numpy as np
r=Path(__file__).resolve().parents[2]/'logs/reference_physics/x2_neighbor_collision_audit';out={}
for side in ['left','right']:
 a=trimesh.load(r/f'{side}_base_raw.ply',process=True);b=trimesh.load(r/f'{side}_hip_raw.ply',process=True);results=[]
 for source,target in [(a,b),(b,a)]:
  edges=source.vertices[source.edges_unique];lo=edges.min(1);hi=edges.max(1);keep=np.all(hi>=target.bounds[0],1)&np.all(lo<=target.bounds[1],1);edges=edges[keep];v=edges[:,1]-edges[:,0];length=np.linalg.norm(v,axis=1);valid=length>1e-9;edges=edges[valid];v=v[valid];length=length[valid];n=0;mins=[]
  for k in range(0,len(edges),500):
   e=edges[k:k+500];directions=v[k:k+500]/length[k:k+500,None];points,rays,tris=target.ray.intersects_location(e[:,0],directions,multiple_hits=True)
   dist=np.linalg.norm(points-e[rays,0],axis=1);hit=(dist>1e-7)&(dist<length[k+rays]-1e-7);n+=int(hit.sum())
  results.append({'tested_edges':len(edges),'segment_surface_hits':n});print(side,results[-1],flush=True)
 out[side]=results
(r/'hip_triangle_edge_test.json').write_text(json.dumps(out,indent=2))
