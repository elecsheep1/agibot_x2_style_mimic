"""动作方向归一化、边界与回放噪声过滤测试。"""
import unittest
import torch
from check_action_direction import action_direction,bounded_perturbation,paired_statistics


class ActionDirectionTests(unittest.TestCase):
    def test_direction_is_independent_per_environment_and_zero_safe(self):
        g=torch.zeros(8,2,3);g[:,0]=2
        d=action_direction(g)
        torch.testing.assert_close(d[:,0],torch.ones(8,3))
        torch.testing.assert_close(d[:,1],torch.zeros(8,3))
        self.assertTrue(torch.isfinite(d).all())

    def test_perturbation_respects_asymmetric_bounds(self):
        a=torch.tensor([[[-.2,.5]]]);d=torch.tensor([[[-2.,2.]]])
        out=bounded_perturbation(a,d,.1,torch.tensor([-.3,-.5]),torch.tensor([.2,.6]))
        torch.testing.assert_close(out,torch.tensor([[[-.3,.6]]]))

    def test_noise_excludes_unresolved_signs_and_invalid_samples(self):
        base=torch.zeros(3);plus=torch.tensor([.1,.1,.1]);minus=-plus
        real_plus=torch.tensor([.1,-.1,1.]);real_minus=-real_plus
        repeat=torch.tensor([.1,0.,0.]);valid=torch.tensor([True,True,False])
        r=paired_statistics(base,base,plus,real_plus,minus,real_minus,repeat,valid)
        self.assertEqual(r['valid_windows'],2)
        self.assertEqual(r['resolved_windows'],1)
        self.assertEqual(r['sign_agreement_fraction'],0.)
        r=paired_statistics(base,base,plus,real_plus,minus,real_minus,repeat,torch.zeros(3,dtype=torch.bool))
        self.assertIsNone(r['sign_agreement_fraction'])


if __name__=='__main__':unittest.main()
