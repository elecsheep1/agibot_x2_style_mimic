"""检查待应用共享奖励与原 PPO 公式的数值和梯度一致性。"""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic/mdp/getup.py'
PROPOSAL = ROOT / 'scripts/patches/standing_reward_shared/standing_reward.py'
spec = importlib.util.spec_from_file_location('_standing_proposal', PROPOSAL)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
namespace = dict(torch=torch, math=math)
tree = ast.parse(SOURCE.read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('smoothstep', 'standing_quality', 'standing_limb_quality')]
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])), str(SOURCE), 'exec'), namespace)


class SharedRewardProposalTests(unittest.TestCase):
    def test_standing_matches_original_values_and_gradients(self):
        torch.manual_seed(4)
        args = [torch.rand(9, dtype=torch.double, requires_grad=True) for _ in range(3)]
        args += [torch.randn(9, 3, dtype=torch.double, requires_grad=True) * .1 for _ in range(2)]
        args += [torch.rand(9, dtype=torch.double, requires_grad=True), torch.rand(9, dtype=torch.double, requires_grad=True) * 10,
                 torch.rand(9, dtype=torch.double, requires_grad=True)]
        old = namespace['standing_quality'](*args, limb_fraction=.2)
        new = shared.standing_quality(*args, limb_fraction=.2)
        torch.testing.assert_close(old, new, atol=0, rtol=0)
        for a, b in zip(torch.autograd.grad(old.sum(), args), torch.autograd.grad(new.sum(), args)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_limb_matches_original_and_preserves_gradient(self):
        torch.manual_seed(8)
        q = torch.randn(5, 14, dtype=torch.double, requires_grad=True)
        root = torch.randn(5, 7, dtype=torch.double, requires_grad=True)
        feet = torch.randn(5, 2, 7, dtype=torch.double, requires_grad=True)
        robot = SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=q), root_link_pose_w=SimpleNamespace(torch=root), body_link_pose_w=SimpleNamespace(torch=feet)))
        env = SimpleNamespace(scene={'robot': robot})
        targets = (.2, .15, 0., -.3, 0., 0., 0., .2, -.15, 0., -.3, 0., 0., 0.)
        old = namespace['standing_limb_quality'](env, SimpleNamespace(joint_ids=list(range(14))), SimpleNamespace(body_ids=[0, 1]), targets)
        new = shared.standing_limb_quality_tensor(q, root[:, 3:7], feet, targets)
        torch.testing.assert_close(old, new, atol=0, rtol=0)
        for a, b in zip(torch.autograd.grad(old.sum(), (q, root, feet)), torch.autograd.grad(new.sum(), (q, root, feet))):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
