"""诊断脚本的纯 CPU 测试，不导入仿真环境或修改正式算法。"""
import importlib.util
from pathlib import Path
import unittest
import torch

HERE = Path(__file__).resolve().parent

def load(name):
    spec = importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

diag = load('diagnose_tracking')
summary = load('summarize_tracking')


class DiagnosticTests(unittest.TestCase):
    def test_windows_include_final_terminal_but_never_cross_reset(self):
        done = torch.tensor([[False],[True],[False],[False]])
        self.assertEqual(diag.valid_windows(done,2).tolist(),[[0,0],[2,0]])
        self.assertEqual(diag.valid_windows(done,3).tolist(),[])
        self.assertEqual(len(diag.valid_windows(done,1)),4)

    def test_pd_limit_fraction_and_per_joint_names(self):
        action = torch.tensor([[[0.,8.]],[[0.,0.]]])
        target = action*.25
        hard = torch.tensor([[[-1.,1.],[-1.,1.]]])
        soft = hard*.9
        result = diag.target_statistics(action,target,hard,soft,['a','b'])
        self.assertEqual(result['target_hard_limit_fraction'],.25)
        self.assertEqual(result['per_joint'][1]['target_hard_limit_fraction'],.5)
        self.assertEqual(result['raw_action_abs']['maximum'],8.)

    def test_log_summary_separates_tracking_and_penalty(self):
        rows = [dict(iteration=i,**{'policy/imag_reward_mean':-.2,
             'policy/imag_r/motion_body_pos':.01,'policy/imag_r/joint_limit':-.21}) for i in range(1,31)]
        result = summary.summarize(rows)
        self.assertEqual(result['sections'][0]['first_iteration'],1)
        self.assertEqual(result['sections'][1]['first_iteration'],11)
        self.assertAlmostEqual(result['sections'][1]['tracking_positive_sum'],.01)
        self.assertAlmostEqual(result['sections'][1]['penalties']['joint_limit'],-.21)

    def test_steps_too_short_have_no_windows(self):
        self.assertEqual(tuple(diag.valid_windows(torch.zeros(2,3,dtype=torch.bool),8).shape),(0,2))


if __name__=='__main__':unittest.main()
