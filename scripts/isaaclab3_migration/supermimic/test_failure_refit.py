"""失败头独立补训的回合隔离与留出验证测试。"""
import contextlib
import copy
import io
import unittest
import torch
from refit_failure_head import split_episodes,refit
from test_failure_supervision import wm


class FailureRefitTests(unittest.TestCase):
    def test_episode_split_is_disjoint_and_drops_partial_tails(self):
        done=torch.zeros(10,4,dtype=torch.bool)
        done[[2,5,8]]=True
        indices,groups,info=split_episodes(done)
        self.assertEqual(info['complete_episodes'],12)
        self.assertEqual(info['discarded_partial_samples'],4)
        self.assertEqual(sum(len(index) for index in indices.values()),36)
        names=list(groups)
        for i in range(len(names)):
            for j in range(i+1,len(names)):
                self.assertFalse(set(groups[names[i]])&set(groups[names[j]]))
                self.assertFalse(set(indices[names[i]].tolist())&set(indices[names[j]].tolist()))

    def test_real_head_can_fit_and_select_using_validation_only(self):
        torch.set_num_threads(2);torch.manual_seed(43)
        head=wm.TerminationHead((16,16),.25,.8)
        x=(torch.rand(1000,2)*2-1)*torch.tensor([.6,1.6])
        y=((x[:,0].abs()>.25)|(x[:,1].abs()>.8)).float()
        indices={'train':torch.arange(700),'validation':torch.arange(700,850),'test':torch.arange(850,1000)}
        before=copy.deepcopy(head.state_dict())
        with contextlib.redirect_stdout(io.StringIO()):
            report=refit(head,x,y,indices,updates=300,batch_size=128,lr=.005,seed=43,log_interval=100)
        self.assertLess(report['after']['validation']['bce'],report['before']['validation']['bce']*.6)
        self.assertTrue(any(not torch.equal(value,head.state_dict()[key]) for key,value in before.items()))
        self.assertTrue(all(set(item)=={'update','validation'} for item in report['history']))


if __name__=='__main__':unittest.main()
