"""只读检查检查点中的动作反馈及限幅梯度，不启动仿真，不修改 source。"""
import argparse
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import torch

SOURCE = Path(__file__).resolve().parents[2] / 'source/x2_style_mimic/x2_style_mimic'
for name, path in [('action_audit', SOURCE), ('action_audit.algorithms', SOURCE / 'algorithms'),
                   ('action_audit.algorithms.supermimic', SOURCE / 'algorithms/supermimic')]:
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
prefix = 'action_audit.algorithms.supermimic.'
state_module = importlib.import_module(prefix + 'state')
actors = importlib.import_module(prefix + 'actor_critic')
observations = importlib.import_module(prefix + 'observations')
worlds = importlib.import_module(prefix + 'world_model')


def rms(x):
    return float(x.detach().square().mean().sqrt())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {'checkpoints': []}
    for path in sorted(args.run.glob('checkpoint_*.pt')):
        data = torch.load(path, map_location='cpu', weights_only=False)
        spec = state_module.StateSpec(**data['metadata']['state_spec'])
        modules = data['modules']
        p = modules['proprioception']
        proprio = observations.Proprioception(spec, p['joint_ids'], p['default_joint_position'], p['default_joint_velocity'])
        proprio.load_state_dict(p)
        if not {'action_lower', 'action_upper'}.issubset(modules['actor']):
            raise ValueError('该检查点使用旧无界 actor；请使用对应版本的历史诊断，不按新动作边界解释旧策略')
        actor = actors.GoalConditionedActor(proprio.output_dim, 0, spec.num_actions,
            action_lower=modules['actor']['action_lower'], action_upper=modules['actor']['action_upper'])
        actor.load_state_dict(modules['actor'])
        w = modules['world_model']
        limiter = worlds.TargetLimiter(**{key: w['limiter.' + key] for key in ('offset', 'scale', 'lower', 'upper', 'max_delta')})
        context = data['physics_context']
        world = worlds.WorldModel(spec, context.shape[-1], limiter, tuple(data['metadata']['world_hidden_dims']))
        world.load_state_dict(w)
        cohort = data['cohort']['states']
        ids = p['joint_ids']
        initial = state_module.pack_state(cohort['root_pose'], cohort['root_velocity'], cohort['joint_pos'], cohort['joint_vel'],
                                         torch.zeros(len(context), spec.num_actions), cohort['joint_pos'][:, ids])
        history = initial[:, None].expand(-1, 5, -1).clone()
        goals = torch.empty(len(context), 0)
        fixed = []
        with torch.no_grad():
            for _ in range(32):
                action = actor(proprio(history), goals)
                fixed.append(rms(action))
                following = initial.clone()
                following[:, 13 + 2 * spec.num_joints:13 + 2 * spec.num_joints + spec.num_actions] = action
                history = torch.cat((history[:, 1:], following[:, None]), dim=1)
        # 硬限位完全饱和区：改变原始动作不改变执行目标，但可能改变网络动力学预测。
        a1 = torch.full((len(context), spec.num_actions), 100.)
        a2 = a1 * 2
        with torch.no_grad():
            predicted1, predicted2 = world(initial, a1, context), world(initial, a2, context)
        target = initial[:, -spec.num_actions:]
        a = a1.clone().requires_grad_(True)
        limited = limiter(target, a)
        target_gradient = torch.autograd.grad(limited.sum(), a)[0]
        model_gradient = torch.autograd.grad(world(initial, a, context)[:, :13 + 2 * spec.num_joints].sum(), a)[0]
        row = dict(iteration=data['iteration'], fixed_physics_action_rms=fixed,
                   saturated_target_delta_max=float((limiter(target, a1) - limiter(target, a2)).abs().max()),
                   saturated_world_physical_prediction_delta_rms=rms(predicted1[:, :13+2*spec.num_joints]-predicted2[:, :13+2*spec.num_joints]),
                   saturated_target_gradient_rms=rms(target_gradient), saturated_model_gradient_rms=rms(model_gradient))
        report['checkpoints'].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    rows = [json.loads(s) for s in (args.run / 'metrics.jsonl').read_text().splitlines()]
    train = [r for r in rows if r['kind'] == 'train']
    report['first_action_rms_threshold_crossings'] = {str(t): next(({'iteration': r['iteration'], 'rms': r['metrics']['collection/action_rms']} for r in train if r['metrics']['collection/action_rms'] > t), None) for t in (1, 5, 10, 100)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(report['first_action_rms_threshold_crossings'])


if __name__ == '__main__':
    main()
