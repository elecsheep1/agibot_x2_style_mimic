"""Reproduce the local temporal/support repairs into a separate motion dataset."""

# Support direct execution: python scripts/<category>/<tool>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import hashlib
import json
import shutil
from pathlib import Path
import mujoco
import numpy as np
from checks.check_x2_retarget import AUTHORED_FILTERS, DEFAULTS, audit
from repair.repair_x2_reference_temporal import repair
from repair.repair_x2_reference_support import solve
from repair.smooth_x2_root_preserve_feet import repair as smooth_root
from common.x2_motion_utils import load_motion


ROOT = Path(__file__).resolve().parents[2]
RECIPES = [
    dict(steps=[('arms',(2.4,3.4),.09),('legs',(2.65,3.75),.07)], starts=[3.5,3.4], transition=.5),
    dict(steps=[('arms',(0,.65),.07),('arms',(3.85,4.8),.07),('arms',(5.7,6.7),.07)], starts=[5.8,5.7], transition=.4),
    dict(steps=[('arms',(0,.65),.07),('arms',(6.6,7.6),.07),('arms',(8.1,9.1),.07)], starts=[6.7,7.65], transition=.4),
    dict(steps=[('arms',(1.2,2.5),.08),('legs',(3.5,4.5),.07)], starts=[3.7,3.9], transition=.4),
]


def main():
    source = ROOT/'data/motions/prone_getup'
    output = ROOT/'data/motions/prone_getup2'
    work = ROOT/'logs/retarget_quality/prone_getup2_build'
    output.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'motions.json').read_text())
    reports=[]
    for number,(entry,recipe) in enumerate(zip(manifest['motions'], RECIPES),1):
        original=source/entry['file'];current=original
        destination=output/entry['file']
        if destination.exists():
            sidecar=destination.with_suffix('.json')
            saved=json.loads(sidecar.read_text()) if sidecar.exists() else {}
            provenance=saved.get('dataset_repair',{})
            if provenance.get('source_sha256')!=hashlib.sha256(original.read_bytes()).hexdigest() or provenance.get('recipe')!=json.loads(json.dumps(recipe)) or provenance.get('pipeline_revision')!=3:
                raise FileExistsError(f'Existing candidate has different/unverified provenance: {destination}')
            reports.append(audit(destination,dict(DEFAULTS)))
            print(f'Already repaired, verified source: {destination.name}',flush=True)
            continue
        print(f'REPAIR MOTION {number}: {original.name}',flush=True)
        z=load_motion(original);model=mujoco.MjModel.from_xml_path(z['metadata']['mujoco_xml'])
        def bodyname(b):
            name=model.body(b).name
            return 'base_link' if name=='pelvis' else name
        filters={'excluded_adjacent_pairs':[[bodyname(b),bodyname(int(model.body_parentid[b]))] for b in range(1,model.nbody)],
                 'excluded_authored_filters':AUTHORED_FILTERS}
        filterpath=work/f'{number}_collision_filters.json';filterpath.write_text(json.dumps(filters))
        for stage,(mode,interval,sigma) in enumerate(recipe['steps']):
            target=work/f'{number}_stage{stage}_{mode}.npz'
            if not target.exists():repair(current,target,None,filterpath,mode,interval,sigma,.2)
            current=target
        anchorpath=None
        if 'support_repair' in z['metadata']:
            anchors=json.loads(json.dumps(z['metadata']['support_repair']['anchors']))
            for anchor,start in zip(anchors,recipe['starts']):
                anchor['start_s']=start;anchor['fully_locked_s']=start+recipe['transition']
            anchorpath=work/f'{number}_preserved_anchors.json'
            anchorpath.write_text(json.dumps({'anchors':anchors},indent=2))
        solve(current,destination,filterpath,recipe['starts'],recipe['transition'],35,
              smooth_interval=(min(recipe['starts']),float(z['time_s'][-1])),smooth_strength=800.,anchor_report=anchorpath)
        if number==1:
            initial=work/'1_initial_support.npz';shutil.copy2(destination,initial)
            stages=[
                ('1_root_fixedfeet', 'root', (3.9,5.8), .10, None),
                ('1_root_cleararms', 'arms', (3.6,4.8), .10, initial),
                ('1_early_arms', 'arms', (0,2.1), .08, None),
                ('1_root_pose', 'root_rotation', (3.9,5.8), .07, None),
                ('1_final_pose', 'arms', (3.6,4.9), .10, work/'1_early_arms.npz'),
                ('1_final_complete', 'legs', (0,3.8), .07, None)]
            current=initial
            for tag,mode,interval,sigma,reference in stages:
                target=work/(tag+'.npz')
                if not target.exists():
                    if mode.startswith('root'):smooth_root(current,target,interval,sigma,mode=='root_rotation')
                    else:repair(current,target,reference,filterpath,mode,interval,sigma,.3 if interval[0]>3 else .2)
                current=target
            checked=audit(current,dict(DEFAULTS))
            if checked['status']=='fail':raise RuntimeError('First motion refinement failed QA')
            shutil.copy2(current,destination)
        if number==4:
            initial=work/'4_initial_support.npz';shutil.copy2(destination,initial)
            target=work/'4_rightleg_temporal.npz'
            if not target.exists():repair(initial,target,None,filterpath,'legs',(3.3,4.3),.07,.2,side='right')
            checked=audit(target,dict(DEFAULTS))
            if checked['status']=='fail':raise RuntimeError('Fourth motion refinement failed QA')
            shutil.copy2(target,destination)
        report=audit(destination,dict(DEFAULTS));reports.append(report)
        if report['status']=='fail':raise RuntimeError(f'Geometry/data QA failed: {destination}')
        metadata=load_motion(destination)['metadata']
        metadata['dataset_repair']={'source':str(original),'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'recipe':recipe,'pipeline_revision':3}
        payload=load_motion(destination);payload.pop('metadata')
        payload['metadata_json']=np.array(json.dumps(metadata))
        np.savez_compressed(destination,**payload)
        report['sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
        destination.with_suffix('.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
        (work/f'{number}_quality.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    (output/'motions.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    (work/'summary.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
