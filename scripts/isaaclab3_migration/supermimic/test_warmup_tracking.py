"""预热脚本的留出隔离、失败标签和启动检查测试。"""
import importlib
import unittest
import torch
from test_failure_supervision import PREFIX
from warmup_tracking import PretrainingCfg,ReplayEnvironmentView,readiness

replay=importlib.import_module(PREFIX+'.algorithms.supermimic_tracking.replay_buffer')


class WarmupTests(unittest.TestCase):
    def test_environment_view_isolates_and_keeps_target_done_labels(self):
        b=replay.ReplayBuffer(2,20,75,29,35,'cpu');ids=torch.arange(2)
        for row in range(4):
            s=torch.zeros(2,75);s[:,3]=1;s[:,0]=torch.tensor([1.,2.]);s[:,2]=.6 if row<3 else .2
            b.add(ids,s,torch.zeros(2,29),torch.zeros(2,29),torch.zeros(2,35),torch.zeros(2),
                torch.full((2,),float(row>=2)),torch.full((2,),row),ids,
                torch.full((2,),row==3),torch.full((2,),row==0))
        b.end_episode(ids)
        view=ReplayEnvironmentView(b,[1])
        self.assertEqual(view.candidates(3).tolist(),[[1,0]])
        batch=view.sample_windows(3,3)
        torch.testing.assert_close(batch.states[...,0],torch.full((4,3),2.))
        torch.testing.assert_close(batch.done,torch.tensor([[0.,0.,0.],[0.,0.,0.],[1.,1.,1.]]))

    def test_fixed_windows_do_not_duplicate_starts(self):
        b=replay.ReplayBuffer(2,32,75,29,35,'cpu');ids=torch.arange(2)
        for row in range(12):
            s=torch.zeros(2,75);s[:,3]=1;s[:,0]=row
            b.add(ids,s,torch.zeros(2,29),torch.zeros(2,29),torch.zeros(2,35),torch.zeros(2),
                  torch.zeros(2),torch.full((2,),row),ids,torch.zeros(2,dtype=torch.bool),torch.full((2,),row==0))
        batch,count=ReplayEnvironmentView(b,[0]).fixed_windows(100,3)
        self.assertEqual(count,9)
        self.assertEqual(len(batch.states[0,:,0].unique()),count)

    def test_gate_rejects_bad_model_and_missing_failure_coverage(self):
        cfg=PretrainingCfg()
        world={'8':{'root_position_m':.01,'root_rotation_deg':1.},'32':{'root_position_m':.05,'root_rotation_deg':5.}}
        head={'positive_count':5,'recall':1.,'false_positive_rate':0.,'bce':.01,'constant_train_prior_bce':.1}
        self.assertTrue(readiness(world,head,cfg)[0])
        world['32']['root_position_m']=.4
        self.assertFalse(readiness(world,head,cfg)[0])
        world['32']['root_position_m']=float('nan')
        self.assertFalse(readiness(world,head,cfg)[0])
        world['32']['root_position_m']=.05;head['positive_count']=0;head['recall']=None
        self.assertFalse(readiness(world,head,cfg)[0])

    def test_capacity_protects_terminal_rows(self):
        cfg=PretrainingCfg()
        cfg.validate(32,1024)
        with self.assertRaises(ValueError):cfg.validate(32,256)
        with self.assertRaises(ValueError):cfg.validate(8,1024)


if __name__=='__main__':unittest.main()
