"""读取独立跟踪训练的 JSONL，汇总动作、奖励与预测误差趋势；不启动仿真。"""
import argparse
import json
from pathlib import Path

POSITIVE = ('motion_global_root_pos','motion_global_root_ori','motion_body_pos',
            'motion_body_ori','motion_body_lin_vel','motion_body_ang_vel')


def summarize(rows):
    if not rows:
        raise ValueError('日志没有完整指标记录')
    selected = [row for row in rows if 'policy/imag_reward_mean' in row]
    if not selected:
        raise ValueError('日志尚未包含策略更新指标')
    chunks = []
    for name,batch in [('early',selected[:min(20,len(selected))]),('late',selected[-min(20,len(selected)):])]:
        keys = set.intersection(*(set(row) for row in batch))
        mean = {key:sum(row[key] for row in batch)/len(batch) for key in keys
                if isinstance(batch[0][key],(int,float))}
        positive = sum(mean.get('policy/imag_r/'+term,0) for term in POSITIVE)
        negative = {key.split('/')[-1]:value for key,value in mean.items()
                    if key.startswith('policy/imag_r/') and key.split('/')[-1] not in POSITIVE}
        chunks.append(dict(section=name,records=len(batch),first_iteration=batch[0]['iteration'],
            last_iteration=batch[-1]['iteration'],tracking_positive_sum=positive,penalties=negative,metrics=mean))
    return dict(records=len(selected),last=selected[-1],sections=chunks,
        notes=['训练 normalized_return 不是逐动作验收成功率。',
               '终止分类准确率需要结合正例比例；累计存活权重不是实测成功率。',
               '动作与关节越限的真实分布需要单独运行 diagnose_tracking.py。'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    path = args.run/'metrics.jsonl'
    rows = []
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: continue  # 正在运行时最后一行可能尚未写完。
    report = summarize(rows)
    target = args.output or args.run/'diagnostics/log_summary.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    for section in report['sections']:
        print(f"[{section['section']}] iteration {section['first_iteration']}～{section['last_iteration']}")
        for key in ('train/normalized_return','world/loss','world/nse_1step','world/nse_kstep',
                    'policy/action_abs_mean','policy/imag_survival_H'):
            if key in section['metrics']:print(f'  {key}: {section["metrics"][key]:.6g}')
        print(f"  跟踪正奖励合计: {section['tracking_positive_sum']:.6g}")
        print(f"  惩罚: {section['penalties']}")
    print(f'[报告] {target}')


if __name__=='__main__': main()
