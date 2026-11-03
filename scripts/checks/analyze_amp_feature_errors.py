"""Offline AMP window mismatch and checkpoint sensitivity; does not change training."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.amp_feature_layout import feature_blocks

BLOCKS = feature_blocks(175)


def groups(joints, bodies, state_dim=175):
    blocks = feature_blocks(state_dim, len(bodies), len(joints))
    result = {k: list(range(a, b)) for k, (a, b) in blocks.items()}
    joint = ({name: [blocks['joint_position'][0]+i, blocks['joint_velocity_scaled'][0]+i]
              for i, name in enumerate(joints)} if 'joint_position' in blocks else {})
    body = {}
    body_components = {}
    for i, name in enumerate(bodies):
        components = {
            k: list(range(start+i*width, start+(i+1)*width))
            for k, start, width in [('position',blocks['body_position'][0],3),
                                   ('rotation6d',blocks['body_rotation6d'][0],6),
                                   ('linear',blocks['body_linear'][0],3),
                                   ('angular_scaled',blocks['body_angular_scaled'][0],3)]
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
    state_dim=windows.shape[-1]
    sse=torch.zeros(state_dim,dtype=torch.float64);raw_sse=torch.zeros_like(sse)
    for chunk in ids.split(256):
        delta=(windows[chunk]-refs[matches[chunk]]).double()
        raw_sse+=delta.square().sum((0,1))
        sse+=(delta/std.double()).square().sum((0,1))
    count=len(ids)*windows.shape[1];total=float(sse.sum())
    return dict(count=len(ids),normalized_rms=(total/(count*state_dim))**.5,
                **{k:error_rows(sse,raw_sse,g,count,total) for k,g in group_sets.items()})


def score_stats(scores, mask):
    s=scores[mask]
    if not len(s):return dict(count=0)
    r=(1-.25*(s-1).square()).clamp(0,1)
    return dict(count=len(s),score_mean=float(s.mean()),p10_p50_p90=s.quantile(torch.tensor([.1,.5,.9])).tolist(),
                reward_mean=float(r.mean()),zero_reward_fraction=float((r==0).float().mean()))



def checkpoint_normalization(checkpoint):
    """Old checkpoints used fixed statistics without clipping."""
    state = checkpoint['discriminator']
    running = 'state_count' in state or 'state_var' in state
    clip = checkpoint['settings'].get('normalization_clip')
    if running and clip is None:
        raise ValueError('Running-normalization checkpoint is missing normalization_clip')
    if clip is not None and (not torch.isfinite(torch.tensor(float(clip))) or float(clip) <= 0):
        raise ValueError('normalization_clip must be finite and positive')
    return dict(mode='joint_running' if running else 'fixed_reference',
                clip=None if clip is None else float(clip))


def discriminator_input(windows, mean, std, normalization):
    normalized = (windows - mean) / std
    clip = normalization['clip']
    if clip is not None:
        normalized = normalized.clamp(-clip, clip)
    return normalized.flatten(1)


def sensitivity(data, refs, matches, group_sets, mask, device, max_samples=1024):
    ids=mask.nonzero().flatten()
    if len(ids)>max_samples:ids=ids[torch.linspace(0,len(ids)-1,max_samples).long()]
    if not len(ids):return dict(count=0)
    ckpt=torch.load(data['checkpoint'],map_location='cpu',weights_only=False)['amp_window']
    state=ckpt['discriminator']
    normalization=checkpoint_normalization(ckpt)
    if not torch.equal(data['state_std'],state['state_std']) or not torch.equal(data['state_mean'],state['state_mean']):
        raise ValueError('Capture normalization differs from checkpoint')
    # Reconstruct both the original and enlarged ELU networks from checkpoint shapes.
    network_state = {k.removeprefix("network."): v for k, v in state.items()
                     if k.startswith("network.")}
    linear_indices = sorted(int(k.split(".")[0]) for k in network_state
                            if k.endswith(".weight"))
    layers = []
    for index in linear_indices:
        weight = network_state[f"{index}.weight"]
        layers.append(nn.Linear(weight.shape[1], weight.shape[0]))
        if index != linear_indices[-1]:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers).to(device)
    net.load_state_dict(network_state)
    net.eval()
    mean=data['state_mean'].to(device);std=data['state_std'].to(device)
    w=data['windows'][ids].to(device);r=refs[matches[ids]].to(device)
    def score(x):return net(discriminator_input(x,mean,std,normalization)).squeeze(-1)
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
    if w.ndim!=3 or w.shape[1]<2 or len(d['joint_names'])!=29 or len(d['body_names'])!=11:
        raise ValueError('Expected AMP windows with named joints/bodies')
    num_steps,state_dim=w.shape[1:]
    if refs.ndim!=3 or refs.shape[1:]!=w.shape[1:] or std.shape!=(state_dim,):
        raise ValueError('Capture/reference/statistic dimensions disagree')
    gs=groups(d['joint_names'],d['body_names'],state_dim)
    checkpoint=torch.load(d['checkpoint'],map_location='cpu',weights_only=False)['amp_window']
    normalization=checkpoint_normalization(checkpoint)
    # Descriptive distances stay unclipped, even when D inputs are clipped.
    # Recompute so older captures with clipped nearest-neighbor matching are safe.
    match=nearest_match(w,refs,std,device)
    phases={'all':torch.ones(len(w),dtype=torch.bool),
            'pre_fade_hold_le_010':d['hold_s']<=.10,
            'fade_hold_010_to_050':(d['hold_s']>.10)&(d['hold_s']<.50),
            'post_fade_hold_ge_050':d['hold_s']>=.50}
    report=dict(checkpoint=d['checkpoint'],capture=str(Path(path).resolve()),
        definitions=dict(phase='Window endpoint hold_seconds; pre-fade includes failures, prone stillness and recovery. Complete valid windows only.',
                         matching=f'Nearest whole {num_steps}-state window across all clips/phases. Same-birth-motion matching also reported.',
                         normalization=normalization,
                         distance='Unclipped differences divided by checkpoint std (floor 0.05), for both matching and error blocks; scaled feature units are preserved. D scoring separately applies checkpoint clipping.',
                         errors='Aggregate normalized RMS and squared-error share, not discriminator importance.',
                         sensitivity=f'Replace selected features across all {num_steps} states with nearest-reference values. Hybrid inputs can be physically inconsistent/out of distribution; descriptive sensitivity, not causal proof or a reduced-feature retraining result.'),
        dimensions=state_dim,num_steps=num_steps,feature_version=d.get('feature_version'),
        reference_files=d['reference_files'],reference_window_counts=d['reference_window_counts'],
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
    lines=[f'# AMP{state_dim} 特征误差分析','',f"检查点：`{d['checkpoint']}`",'',
           '仅描述当前策略与参考的差异；高误差不等于特征不重要。混合替换输入可能不满足物理一致性。',
           f"标准化：{normalization['mode']}；D 输入裁剪：{normalization['clip']}。匹配和误差统计均使用未裁剪距离。",'',
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
