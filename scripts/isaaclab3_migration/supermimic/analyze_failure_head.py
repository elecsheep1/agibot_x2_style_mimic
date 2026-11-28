"""用真实下一状态单独检查失败头，区分分类错误与动力学预测错误；纯 CPU。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def features(state,reference):
    q = state[...,3:7]
    q = q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    ref_q = reference[...,3:7]
    gravity_z = -(1-2*q[...,1:3].square().sum(-1))
    ref_gravity_z = -(1-2*ref_q[...,1:3].square().sum(-1))
    return torch.stack((reference[...,2]-state[...,2],ref_gravity_z-gravity_z),-1)


def predict(weights,feats):
    normalized = feats*weights['term_head.scale']
    x = torch.cat((normalized,normalized.abs()),-1)
    keys = sorted([key for key in weights if key.startswith('term_head.mlp.net.') and key.endswith('.weight')],
                  key=lambda key:int(key.split('.')[-2]))
    for index,key in enumerate(keys):
        x = F.linear(x,weights[key],weights[key.replace('.weight','.bias')])
        if index<len(keys)-1:x = F.silu(x)
    return x.squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    data = torch.load(args.data,map_location='cpu',weights_only=False)['data']
    references = []
    for file in checkpoint['contract']['files']:
        if not file.endswith('.npz'):continue
        with np.load(file,allow_pickle=False) as motion:
            p = torch.tensor(motion['root_pos_w'],dtype=torch.float32)
            p[:,2] -= float(json.loads(str(motion['metadata_json'].item()))['ground_z'])
            xyzw = torch.tensor(motion['root_quat_xyzw'],dtype=torch.float32)
            q = torch.cat((xyzw[:,-1:],xyzw[:,:3]),-1)
            q = q/q.norm(dim=-1,keepdim=True)
            references.append(torch.cat((p,q),-1))
    reference = torch.cat(references)[data['frame']]
    feats = features(data['next_state'],reference)
    cfg = checkpoint['config']['world_model']
    analytic = (feats[...,0].abs()>cfg['height_threshold']) | (feats[...,1].abs()>cfg['tilt_threshold'])
    label = data['terminated'].bool()
    logits = predict(checkpoint['world'],feats)
    prob = logits.sigmoid()
    def subset(mask):
        if not mask.any():return dict(count=0)
        p = prob[mask]
        return dict(count=len(p),mean_probability=float(p.mean()),minimum=float(p.min()),maximum=float(p.max()),
                    predicted_failure_fraction=float((p>=.5).float().mean()))
    prior = label.float().mean()
    baseline = -(prior*prior.clamp_min(1e-8).log()+(1-prior)*(1-prior).clamp_min(1e-8).log())
    report = dict(checkpoint_iteration=checkpoint['iteration'],samples=label.numel(),
        feature_label_mismatch=int((analytic!=label).sum()),positive=subset(label),negative=subset(~label),
        false_negatives=int((label & (prob<.5)).sum()),false_positives=int((~label & (prob>=.5)).sum()),
        true_state_bce=float(F.binary_cross_entropy_with_logits(logits,label.float())),
        label_positive_fraction=float(prior),constant_prior_bce=float(baseline),
        notes=['这里输入真实下一状态，不包含动力学开环误差。',
               'feature_label_mismatch 应为零，否则先检查参考帧及失败条件口径。'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
