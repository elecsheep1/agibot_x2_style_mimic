"""辅助规则、原 AMP 兼容性和想象动力学梯度。"""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch

BASE = Path(__file__).resolve().parents[2] / 'source/x2_style_mimic/x2_style_mimic'
TASK = BASE / 'tasks/manager_based/x2_style_mimic'
p = TASK / 'mdp/assistance.py'
spec = importlib.util.spec_from_file_location('_height_assist_test', p)
a = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = a
spec.loader.exec_module(a)
p = BASE / 'algorithms/supermimic/imagined_env.py'
node = next(n for n in ast.parse(p.read_text()).body if isinstance(n, ast.ClassDef))
ns = dict(torch=torch, height_assistance=a.height_assistance)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), 'exec'), ns)
ImaginedEnv = ns['ImaginedGetupEnv']
p = TASK / 'mdp/getup_assist.py'
cls = next(n for n in ast.parse(p.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'GetUpAssist')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'apply')
exec(compile(ast.Module(body=[method], type_ignores=[]), str(p), 'exec'), ns)


class AssistanceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = a.HeightAssistanceCfg(.2, .5)

    def test_matches_actual_original_apply_and_withdraw_time(self):
        height = torch.tensor([.1, .2, .35, .5, .7, .1, .3])
        cap = torch.tensor([100., 75., 50., 100., 100., 100., 0.])
        withdrawn = torch.tensor([False, False, False, False, False, True, True])
        pose = torch.zeros(7, 3)
        pose[:, 2] = height
        obj = SimpleNamespace(env=SimpleNamespace(episode_length_buf=torch.full((7,), 12), step_dt=.02,
                              scene=SimpleNamespace(env_origins=torch.zeros(7, 3))),
                              robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=pose)),
                              permanent_wrench_composer=SimpleNamespace(reset=lambda: None, add_forces_and_torques_index=lambda **kw: None)),
                              full_below_height=.2, zero_above_height=.5, withdrawn=withdrawn.clone(),
                              height_assistance_cfg=self.cfg,
                              withdraw_time=torch.full((7,), float('inf')), episode_force=cap,
                              applied_force=torch.zeros(7), forces=torch.zeros(7, 1, 3), torques=torch.zeros(7, 1, 3), body_ids=torch.tensor([0]))
        ns['apply'](obj)
        force, flags = a.height_assistance(height, cap, withdrawn, self.cfg)
        torch.testing.assert_close(force, obj.applied_force, atol=0, rtol=0)
        torch.testing.assert_close(obj.applied_force, torch.tensor([100., 75., 25., 0., 0., 0., 0.]))
        torch.testing.assert_close(flags, obj.withdrawn)
        torch.testing.assert_close(obj.withdraw_time[3:5], torch.full((2,), .24))
        # 高度重新降低后不恢复辅助，也不覆盖首次撤力时间。
        pose[:, 2] = .1
        obj.env.episode_length_buf.fill_(20)
        ns['apply'](obj)
        torch.testing.assert_close(obj.applied_force[3:], torch.zeros(4))
        torch.testing.assert_close(obj.withdraw_time[3:5], torch.full((2,), .24))

    def test_zero_assistance_and_lock_after_falling_back(self):
        force, flags = a.height_assistance(torch.tensor([.6, .1]), torch.tensor([100., 0.]), torch.tensor([False, True]), self.cfg)
        next_force, _ = a.height_assistance(torch.tensor([.1, .1]), torch.tensor([100., 0.]), flags, self.cfg)
        torch.testing.assert_close(force, torch.zeros(2))
        torch.testing.assert_close(next_force, torch.zeros(2))

    def test_multistep_action_gradient_and_reset_copies(self):
        class ToyModel:
            def __call__(self, state, action, context):
                return torch.cat((state[:, :2], state[:, 2:3] + action, state[:, 3:]), -1)
        env = ImaginedEnv(ToyModel())
        initial = torch.zeros(1, 13)
        initial[:, 2] = .3
        env.reset(initial, torch.zeros(1, 1))
        first = torch.tensor([[.01]], requires_grad=True)
        env.step(first)
        last = env.step(torch.tensor([[.01]], requires_grad=True))
        self.assertGreater(float(torch.autograd.grad(last[:, 2].sum(), first)[0].abs().sum()), 0.)



if __name__ == '__main__':
    unittest.main()
