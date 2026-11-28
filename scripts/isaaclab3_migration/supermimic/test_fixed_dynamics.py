"""固定数据补训的环境隔离与终止边界测试。"""
import unittest
import torch
from refit_fixed_dynamics import split_data, gather
from diagnose_tracking import valid_windows
from test_error_audit import make_data


class FixedDynamicsTests(unittest.TestCase):
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
