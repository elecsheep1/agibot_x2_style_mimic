"""Merge independently collected banks, preserve split membership and remap families."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.fallen_bank import load_bank, save_bank, sha256, dedup_key, ARRAY_FIELDS, summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.report.json').exists():
        parser.error('Output/report exists')
    banks=[load_bank(p) for p in args.inputs]
    first=banks[0];rows=[];seen={};offset=0;provenance=[];modes={}
    for path,bank in zip(args.inputs,banks):
        meta=bank['metadata']
        if not meta.get('collection_report',{}).get('complete') or not meta['collection_report'].get('training_ready'):
            raise ValueError(f'Input is incomplete or not training ready: {path}')
        for key in ['asset_hashes','reference_files','reference_sha256']:
            if meta.get(key)!=first['metadata'].get(key):raise ValueError(f'{key} differs: {path}')
        if bank['joint_names'].tolist()!=first['joint_names'].tolist():raise ValueError('Joint ordering differs')
        begin=len(rows)
        for i in range(len(bank['root_pose'])):
            key=dedup_key(bank['joint_pos'][i],bank['root_gravity_b'][i],bank['root_pose'][i,2])
            if key in seen:raise ValueError(f'Quantized duplicate across merged rows: {path}, row {i}, previous {seen[key]}')
            seen[key]=len(rows)
            row={k:bank[k][i].copy() for k in ARRAY_FIELDS}
            row['family_id']=np.int64(int(row['family_id'])+offset)
            rows.append(row)
            original_mode=meta.get('family_collection_modes',{}).get(str(int(bank['family_id'][i])))
            if original_mode is not None:modes[str(int(row['family_id']))]=original_mode
        provenance.append(dict(path=str(path.resolve()),sha256=sha256(path),count=len(bank['root_pose']),
            merged_begin=begin,merged_end=len(rows),family_offset=offset))
        offset+=int(bank['family_id'].max())+1
    coverage={f'{s}/{t}':sum(int(r['source_type'])==s and int(r['split'])==t for r in rows)
              for s in [0,1] for t in [0,1]}
    near_coverage={f'{m}/{t}':sum(int(r['source_type'])==0 and int(r['motion_id'])==m and int(r['split'])==t for r in rows)
                   for m in sorted(set(int(r['motion_id']) for r in rows if int(r['source_type'])==0)) for t in [0,1]}
    report=dict(complete=True,training_ready=all(coverage.values()) and all(near_coverage.values()),
        count=len(rows),coverage=coverage,near_coverage=near_coverage,inputs=provenance,
        split_policy='Original split membership preserved; family IDs offset by input; no holdout moved to train')
    meta={k:first['metadata'][k] for k in ['asset_hashes','reference_files','reference_sha256']}
    meta.update(collection_report=report,family_collection_modes=modes,
        generator=str(Path(__file__).resolve()),generator_sha256=sha256(__file__),
        merge_provenance=provenance,reset_policy='Training resets must center XY; velocity, height and orientation preserved',
        source_fall_semantics='See input collection metadata: standing push or independent side-oriented passive placement')
    bank=save_bank(args.output,rows,first['joint_names'].tolist(),meta)
    report.update(summary(bank));args.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
