"""固定数据补训的环境隔离与终止边界测试。"""
import unittest
import torch
from refit_fixed_dynamics import split_data, gather
from diagnose_tracking import valid_windows
from test_error_audit import make_data
from test_failure_supervision import wm
from types import SimpleNamespace
from unittest.mock import patch
import sys
from refit_fixed_dynamics import initialize_model


class FixedDynamicsTests(unittest.TestCase):
    def test_capacity_change_requires_scratch_and_preserves_head(self):
        original=wm.WorldModel(wm.WorldModelCfg(hidden=(16,)),SimpleNamespace(),'cpu')
        checkpoint={'world':original.state_dict()}
        with patch.dict(sys.modules,{'x2_style_mimic.algorithms.supermimic_tracking.world_model':wm}):
            with self.assertRaises(ValueError):
                initialize_model(wm.WorldModelCfg(),checkpoint,SimpleNamespace(),'cpu',hidden=[32,32])
            larger=initialize_model(wm.WorldModelCfg(),checkpoint,SimpleNamespace(),'cpu',scratch=True,hidden=[32,32])
        for k,v in original.term_head.state_dict().items():
            torch.testing.assert_close(v,larger.term_head.state_dict()[k])
        self.assertGreater(sum(p.numel() for p in larger.dynamics.parameters()),sum(p.numel() for p in original.dynamics.parameters()))
        torch.testing.assert_close(larger.dynamics(torch.zeros(2,75),torch.zeros(2,29),torch.zeros(2,35)),torch.zeros(2,74))

    def test_split_is_disjoint_and_reproducible(self):
        data=make_data(envs=16)
        groups,ids=split_data(data,43)
        _,repeat=split_data(data,43)
        self.assertEqual(ids,repeat)
        self.assertEqual(sorted(sum(ids.values(),[])),list(range(16)))
        self.assertEqual(len(groups['test']['state'][0]),2)

    def test_sampling_preserves_terminal_and_true_next(self):
        data=make_data(envs=8)
        data['terminated'][10,:]=True
        options=valid_windows(data['terminated'],8)
        self.assertFalse(bool(((options[:,0]<=10)&(options[:,0]+6>=10)).any()))
        s,true,actions,context=gather(data,options,32,8,torch.Generator().manual_seed(3))
        self.assertEqual(tuple(true.shape),(32,8,75))
        torch.testing.assert_close(true[:,0,0],s[:,0]+.02)


if __name__=='__main__':unittest.main()
