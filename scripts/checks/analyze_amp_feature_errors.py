"""Offline AMP window mismatch and checkpoint sensitivity; does not change training."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

BLOCKS = {
    'root_height': (0, 1), 'root_gravity': (1, 4), 'root_linear': (4, 7),
    'root_angular_scaled': (7, 10), 'joint_position': (10, 39),
    'joint_velocity_scaled': (39, 68), 'body_position': (68, 101),
    'body_rotation6d': (101, 167), 'body_linear': (167, 200),
    'body_angular_scaled': (200, 233),
}


def groups(joints, bodies):
    result = {k: list(range(a, b)) for k, (a, b) in BLOCKS.items()}
    joint = {name: [10+i, 39+i] for i, name in enumerate(joints)}
    body = {}
    body_components = {}
    for i, name in enumerate(bodies):
        components = {
            k: list(range(start+i*width, start+(i+1)*width))
            for k, start, width in [('position',68,3),('rotation6d',101,6),
                                     ('linear',167,3),('angular_scaled',200,3)]
        }
        body[name] = sum(components.values(), [])
        body_components.update({name+'/'+k: v for k, v in components.items()})
    candidates = {
        'intermediate_body_rotation6d_36': sum([
            body_components[name+'/rotation6d'] for name in bodies
            if 'shoulder' in name or 'hip' in name or 'knee' in name
        ], []),
        'torso_wrists_ankles_rotation6d_30': sum([
            body_components[name+'/rotation6d'] for name in bodies
            if 'torso' in name or 'wrist' in name or 'ankle' in name
        ], []),
        'all_body_relative_velocities_66': result['body_linear']+result['body_angular_scaled'],
    }
    return dict(blocks=result, joints=joint, bodies=body, body_components=body_components,
                candidate_groups=candidates)


def nearest_match(windows, references, std, device):
    ref = (references.to(device) / std.to(device)).flatten(1)
    ref_sq = ref.square().sum(-1)
    matches = []
    for chunk in windows.split(128):
        z = (chunk.to(device) / std.to(device)).flatten(1)
        dist = (z.square().sum(-1,keepdim=True)+ref_sq[None]-2*z@ref.T).clamp_min(0)
        matches.append(dist.argmin(-1).cpu())
    return torch.cat(matches)


def error_rows(sse, raw_sse, group, count, total):
    rows=[]
    for name, ids in group.items():
        value=float(sse[ids].sum())
        rows.append(dict(name=name, dimensions=len(ids), normalized_rms=(value/(count*len(ids)))**.5,
                         normalized_sse_share_percent=100*value/total if total else 0.,
                         raw_feature_rms=float(raw_sse[ids].sum()/(count*len(ids)))**.5))
    return sorted(rows,key=lambda r:r['normalized_sse_share_percent'],reverse=True)


def mismatch(windows, refs, matches, std, mask, group_sets):
    ids=mask.nonzero().flatten()
    if not len(ids):return dict(count=0)
    sse=torch.zeros(233,dtype=torch.float64);raw_sse=torch.zeros_like(sse)
    for chunk in ids.split(256):
        delta=(windows[chunk]-refs[matches[chunk]]).double()
        raw_sse+=delta.square().sum((0,1))
        sse+=(delta/std.double()).square().sum((0,1))
    count=len(ids)*windows.shape[1];total=float(sse.sum())
    return dict(count=len(ids),normalized_rms=(total/(count*233))**.5,
                **{k:error_rows(sse,raw_sse,g,count,total) for k,g in group_sets.items()})


def score_stats(scores, mask):
    s=scores[mask]
    if not len(s):return dict(count=0)
    r=(1-.25*(s-1).square()).clamp(0,1)
    return dict(count=len(s),score_mean=float(s.mean()),p10_p50_p90=s.quantile(torch.tensor([.1,.5,.9])).tolist(),
                reward_mean=float(r.mean()),zero_reward_fraction=float((r==0).float().mean()))


def sensitivity(data, refs, matches, group_sets, mask, device, max_samples=1024):
    ids=mask.nonzero().flatten()
    if len(ids)>max_samples:ids=ids[torch.linspace(0,len(ids)-1,max_samples).long()]
    if not len(ids):return dict(count=0)
    ckpt=torch.load(data['checkpoint'],map_location='cpu',weights_only=False)['amp_window']
    state=ckpt['discriminator']
    if not torch.equal(data['state_std'],state['state_std']) or not torch.equal(data['state_mean'],state['state_mean']):
        raise ValueError('Capture normalization differs from checkpoint')
    net=nn.Sequential(nn.Linear(5*233,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1)).to(device)
    net.load_state_dict({k.removeprefix('network.'):v for k,v in state.items() if k.startswith('network.')})
    net.eval()
    mean=data['state_mean'].to(device);std=data['state_std'].to(device)
    w=data['windows'][ids].to(device);r=refs[matches[ids]].to(device)
    def score(x):return net(((x-mean)/std).flatten(1)).squeeze(-1)
    with torch.no_grad():
        base=score(w)
        if not torch.allclose(base.cpu(),data['score'][ids],atol=1e-4,rtol=1e-4):
            raise ValueError('Offline discriminator score differs from capture')
        out={}
        for category in ['blocks','bodies','joints','body_components','candidate_groups']:
            rows=[]
            for name,dims in group_sets[category].items():
                hybrid=w.clone();hybrid[:,:,dims]=r[:,:,dims]
                changed=score(hybrid);delta=changed-base
                rows.append(dict(name=name,score_change_mean=float(delta.mean()),
                                 score_change_median=float(delta.median()),
                                 score_increase_fraction=float((delta>0).float().mean())))
            out[category]=sorted(rows,key=lambda x:x['score_change_mean'],reverse=True)
    return dict(count=len(ids),**out)


def analyze(path, output, device='cpu'):
    d=torch.load(path,map_location='cpu',weights_only=False)
    w=d['windows'];refs=d['reference_windows'];std=d['state_std']
    if w.ndim!=3 or w.shape[1:]!=(5,233) or len(d['joint_names'])!=29 or len(d['body_names'])!=11:
        raise ValueError('Expected five-state AMP233 windows with named joints/bodies')
    gs=groups(d['joint_names'],d['body_names'])
    match=d.get('nearest_reference_window_index')
    if match is None:match=nearest_match(w,refs,std,device)
    phases={'all':torch.ones(len(w),dtype=torch.bool),
            'pre_fade_hold_le_010':d['hold_s']<=.10,
            'fade_hold_010_to_050':(d['hold_s']>.10)&(d['hold_s']<.50),
            'post_fade_hold_ge_050':d['hold_s']>=.50}
    report=dict(checkpoint=d['checkpoint'],capture=str(Path(path).resolve()),
        definitions=dict(phase='Window endpoint hold_seconds; pre-fade includes failures, prone stillness and recovery. Complete valid windows only.',
                         matching='Nearest whole 5-state window across all clips/phases. Same-birth-motion matching also reported.',
                         normalization='Checkpoint reference std, floor 0.05; scaled feature units are preserved.',
                         errors='Aggregate normalized RMS and squared-error share, not discriminator importance.',
                         sensitivity='Replace selected features across all five states with nearest-reference values. Hybrid inputs can be physically inconsistent/out of distribution; descriptive sensitivity, not causal proof or a reduced-feature retraining result.'),
        dimensions=233,reference_files=d['reference_files'],reference_window_counts=d['reference_window_counts'],
        phases={})
    for name,mask in phases.items():
        report['phases'][name]=dict(scores=score_stats(d['score'],mask),errors=mismatch(w,refs,match,std,mask,gs))
    report['by_motion']={}
    offset=0
    for motion,n in enumerate(d['reference_window_counts']):
        mask=(d['motion_id']==motion)&phases['pre_fade_hold_le_010']
        local_ids=mask.nonzero().flatten();local=match.clone()
        if len(local_ids):local[local_ids]=nearest_match(w[local_ids],refs[offset:offset+n],std,device)+offset
        report['by_motion'][str(motion)]=dict(file=d['reference_files'][motion],scores=score_stats(d['score'],mask),
            global_match=mismatch(w,refs,match,std,mask,gs),same_motion_match=mismatch(w,refs,local,std,mask,gs))
        offset+=n
    # Score sensitivity is intentionally limited to the phase targeted by this experiment.
    report['pre_fade_sensitivity']=sensitivity(d,refs,match,gs,phases['pre_fade_hold_le_010'],device)
    match_motion=torch.bucketize(match,torch.tensor(d['reference_window_counts']).cumsum(0),right=True)
    mask=phases['pre_fade_hold_le_010']
    report['pre_fade_nearest_motion_counts']={str(i):int((match_motion[mask]==i).sum()) for i in range(len(d['reference_files']))}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'feature_errors.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# AMP233 特征误差分析','',f"检查点：`{d['checkpoint']}`",'',
           '仅描述当前策略与参考的差异；高误差不等于特征不重要。混合替换输入可能不满足物理一致性。','',
           '|阶段|窗口数|P10 / P50 / P90|原始风格奖励|零奖励比例|归一化 RMS|',
           '|---|---:|---|---:|---:|---:|']
    for name,item in report['phases'].items():
        s=item['scores'];err=item['errors']
        if not s['count']:continue
        lines.append(f"|{name}|{s['count']}|{' / '.join(f'{v:.4f}' for v in s['p10_p50_p90'])}|{s['reward_mean']:.4f}|{100*s['zero_reward_fraction']:.2f}%|{err['normalized_rms']:.4f}|")
    early=report['phases']['pre_fade_hold_le_010']['errors']
    for category in ['blocks','bodies','joints','body_components']:
        lines+=['',f'## 未淡出阶段：{category}','','|特征|维数|归一化 RMS|平方误差占比|','|---|---:|---:|---:|']
        for row in early.get(category,[])[:20]:lines.append(f"|{row['name']}|{row['dimensions']}|{row['normalized_rms']:.4f}|{row['normalized_sse_share_percent']:.2f}%|")
    for category in ['blocks','bodies','joints','body_components','candidate_groups']:
        lines+=['',f'## 未淡出阶段替换敏感性：{category}','','|替换项|平均分数变化|中位变化|','|---|---:|---:|']
        for row in report['pre_fade_sensitivity'].get(category,[])[:15]:lines.append(f"|{row['name']}|{row['score_change_mean']:+.4f}|{row['score_change_median']:+.4f}|")
    lines+=['','逐动作全库/同动作最近邻对照及全部排名见 feature_errors.json。',
            '参考与策略均为同一特征提取路径；归一化沿用检查点统计。窗口不是独立试验，百分位不是置信区间。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(output/'report.md',flush=True)
    print(json.dumps({k:v['scores'] for k,v in report['phases'].items()}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args();analyze(a.capture,a.output_dir,a.device)
