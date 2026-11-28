"""失败头真实状态监督、梯度隔离和分类统计的 CPU 回归测试。"""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
import torch
from torch import nn

SOURCE = Path(__file__).resolve().parents[3]/'source/x2_style_mimic/x2_style_mimic'
PREFIX = '_failure_supervision_tests'
for name,path in ((PREFIX,SOURCE),(PREFIX+'.algorithms',SOURCE/'algorithms'),
                  (PREFIX+'.algorithms.supermimic_tracking',SOURCE/'algorithms/supermimic_tracking')):
    package=ModuleType(name)
    package.__path__=[str(path)]
    sys.modules[name]=package

wm=importlib.import_module(PREFIX+'.algorithms.supermimic_tracking.world_model')
heads=importlib.import_module(PREFIX+'.algorithms.supermimic_tracking.heads')


def state(batch=4):
    s=torch.zeros(batch,75)
    s[:,2]=.6
    s[:,3]=1.
    return s


def reference(batch=4):
    return heads.ReferenceFrame(None,None,None,None,state(batch)[:,:3],state(batch)[:,3:7])


class FailureSupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        self.model=wm.WorldModel(wm.WorldModelCfg(hidden=(16,16),term_hidden=(16,)),
            SimpleNamespace(termination_features=heads.AnalyticHeads.termination_features),'cpu')
        self.s=state()
        self.true=state();self.true[:,2]=torch.tensor([.3,.9,.6,.6])
        self.action=torch.zeros(4,29)
        self.context=torch.zeros(4,35)
        self.context[:,0]=.75
        self.done=torch.tensor([1.,1.,0.,0.])
        self.ref=reference()

    def loss_step(self,s=None,target=None):
        return self.model._loss_step(self.s if s is None else s,self.action,self.context,
            self.true if target is None else target,self.done,self.ref.root_pos_w,self.ref.root_quat_w)

    def test_classification_uses_true_state_not_predicted_state(self):
        first=self.loss_step()
        changed=self.s.clone();changed[:,2]+=2.
        second=self.loss_step(s=changed)
        self.assertFalse(torch.equal(first[0],second[0]))
        torch.testing.assert_close(first[2],second[2])
        torch.testing.assert_close(first[3],second[3])
        expected=self.model.term_head(self.model.heads.termination_features(self.true[:,:3],self.true[:,3:7],self.ref))
        torch.testing.assert_close(first[3],expected)

    def test_bce_updates_head_without_dynamics_or_target_gradient(self):
        target=self.true.clone().requires_grad_()
        _,_,bce,_=self.loss_step(target=target)
        bce.mean().backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in self.model.term_head.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.dynamics.parameters()))
        self.assertIsNone(target.grad)

    def test_state_loss_still_updates_dynamics(self):
        _,error,_,_=self.loss_step()
        error.mean().backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in self.model.dynamics.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.term_head.parameters()))

    def test_imagination_keeps_gradient_to_predicted_state(self):
        self.model.requires_grad_(False)
        s=self.true.clone().requires_grad_()
        self.model.termination_logits(s,self.ref).sum().backward()
        self.assertIsNotNone(s.grad)
        self.assertTrue(bool(torch.isfinite(s.grad).all()))
        self.assertGreater(float(s.grad[:,:3].abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_window_counts_and_probability_sums(self):
        class FixedHead(nn.Module):
            def forward(self,features):return features[...,0]*10
        self.model.term_head=FixedHead()
        states=torch.stack((self.s,self.true,self.true))
        _,metrics=self.model.window_loss(states,self.action.unsqueeze(0).expand(2,-1,-1),self.context,
                                         self.done.unsqueeze(0).expand(2,-1),[self.ref,self.ref])
        expected={'term_positive_count':4,'term_negative_count':4,'term_tp':2,'term_fn':2,'term_fp':0,'term_tn':4}
        for name,count in expected.items():self.assertEqual(float(metrics[name]),count)
        self.assertAlmostEqual(float(metrics['term_positive_probability_sum']),2.,places=5)
        self.assertAlmostEqual(float(metrics['term_negative_probability_sum']),2.,places=5)
        self.assertTrue(all(not v.requires_grad for v in metrics.values()))

    def test_no_positive_class_remains_finite_and_explicit(self):
        states=self.s.unsqueeze(0).expand(3,-1,-1)
        _,metrics=self.model.window_loss(states,self.action.unsqueeze(0).expand(2,-1,-1),self.context,
                                         torch.zeros(2,4),[self.ref,self.ref])
        self.assertEqual(float(metrics['term_positive_count']),0.)
        self.assertEqual(float(metrics['term_tp']),0.)
        self.assertEqual(float(metrics['term_fn']),0.)
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in metrics.values()))


if __name__=='__main__':unittest.main()
