"""想象轨迹诊断的汇总口径必须区分正负奖励、终止概率和动作饱和。"""
import importlib.util
from pathlib import Path
import unittest

import torch

path = Path(__file__).resolve().parents[1] / 'checks/audit_getup_imagination.py'
spec = importlib.util.spec_from_file_location('_imagination_audit_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ImaginationAuditTests(unittest.TestCase):
    def test_term_totals_excludes_style_and_splits_sign(self):
        terms = {'standing': torch.tensor([[1.0, 2.0], [1.0, 1.0]]),
                 'style': torch.full((2, 2), 5.0),
                 'head_contact': torch.tensor([[-0.5, -1.0], [-0.5, -0.5]])}
        result = module.term_totals(terms)
        self.assertNotIn('style', result['terms'])
        self.assertAlmostEqual(result['terms']['standing'], 2.5)
        self.assertAlmostEqual(result['terms']['head_contact'], -1.25)
        self.assertAlmostEqual(result['positive_total'], 2.5)
        self.assertAlmostEqual(result['negative_total'], -1.25)
        self.assertAlmostEqual(result['negative_fraction'], 1.25 / 3.75)

    def test_term_totals_handles_zero_gross(self):
        result = module.term_totals({'standing': torch.zeros(2, 3)})
        self.assertEqual(result['negative_fraction'], 0.0)

    def test_termination_summary_reports_first_step_and_fraction(self):
        probabilities = torch.tensor([[0.1, 0.9], [0.2, 0.3], [0.6, 0.1]])
        result = module.termination_summary(probabilities)
        self.assertAlmostEqual(result['first_step_mean'], 0.5)
        self.assertAlmostEqual(result['maximum'], 0.9)
        self.assertAlmostEqual(result['fraction_above_half'], 2 / 6)

    def test_action_summary_detects_saturation_and_change(self):
        lower, upper = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])
        actions = torch.tensor([[[1.0, -1.0]], [[0.0, 0.0]]])
        result = module.action_summary(actions, lower, upper)
        self.assertAlmostEqual(result['saturated_fraction'], 0.5)
        self.assertAlmostEqual(result['change_rms'], 1.0)
        self.assertAlmostEqual(result['maximum'], 1.0)

    def test_task_return_accumulates_every_transition(self):
        class Task:
            def reward(self, before, action, following, goals, style):
                return torch.ones(before.shape[0]), {}
        states = torch.zeros(2, 4, 1)
        actions = torch.zeros(2, 3, 1)
        total = module.task_return(Task(), states, actions, torch.zeros(2, 0))
        torch.testing.assert_close(total, torch.full((2,), 3.0))


if __name__ == '__main__':
    unittest.main()
