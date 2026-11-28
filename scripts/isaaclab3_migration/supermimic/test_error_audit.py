"""误差来源诊断的同批窗口、速度 oracle 和单步误差分解测试。"""
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from test_failure_supervision import wm
from audit_model_errors import sample_cohort,variant_rollout,one_step_budget,validate_data


class FixedDynamics(nn.Module):
    def forward(self,feat,action,context):
        delta=feat.new_zeros(len(feat),74)
        delta[:,0]=.3
        delta[:,6:37]=.1
        return delta


def make_data(steps=40,envs=2):
    states=torch.zeros(steps+1,envs,75);states[:,:,3]=1;states[:,:,2]=.6;states[:,:,38]=1
    states[:,:,0]=torch.arange(steps+1)[:,None]*.02
    return dict(state=states[:-1],next_state=states[1:],executed=torch.zeros(steps,envs,29),
                context=torch.zeros(steps,envs,35),frame=torch.arange(steps)[:,None].expand(-1,envs).clone(),
                terminated=torch.zeros(steps,envs,dtype=torch.bool),truncated=torch.zeros(steps,envs,dtype=torch.bool))


class ErrorAuditTests(unittest.TestCase):
    def setUp(self):
        self.model=wm.WorldModel(wm.WorldModelCfg(hidden=(16,)),SimpleNamespace(),'cpu')
        self.model.dynamics=FixedDynamics()
        self.data=make_data()
        self.cohort=sample_cohort(self.data,8,43)

    def test_windows_have_unique_starts_and_do_not_cross_resets(self):
        self.data['terminated'][5,0]=True
        c=sample_cohort(self.data,100,43)
        pairs=list(zip(c['starts'].tolist(),c['envs'].tolist()))
        self.assertEqual(len(set(pairs)),len(pairs))
        self.assertTrue(all(not (env==0 and start<=5<start+31) for start,env in pairs))

    def test_teacher_forcing_and_velocity_oracle_separate_feedback_error(self):
        baseline=variant_rollout(self.model,self.cohort,'baseline')
        teacher=variant_rollout(self.model,self.cohort,'teacher_forced')
        oracle=variant_rollout(self.model,self.cohort,'oracle_root_velocity')
        true=self.cohort['following']
        error=(baseline[:,-1,:3]-true[:,-1,:3]).norm(dim=-1).mean()
        single=(teacher[:,-1,:3]-true[:,-1,:3]).norm(dim=-1).mean()
        self.assertGreater(float(error),float(single)*10)
        torch.testing.assert_close(oracle[...,:3],true[...,:3],atol=1e-6,rtol=0)
        joint=variant_rollout(self.model,self.cohort,'oracle_joint_velocity')
        torch.testing.assert_close(joint[...,7:38],true[...,7:38])

    def test_single_step_budget_reconstructs_actual_position_error(self):
        result=one_step_budget(self.model,self.cohort)
        self.assertLess(result['root_position_budget_residual_max_m'],1e-6)
        reconstructed=(result['root_position_velocity_term_mse']+result['root_position_correction_term_mse']
                       +result['root_position_cross_term'])
        self.assertAlmostEqual(reconstructed,result['root_position_total_mse'],places=9)

    def test_continuity_check_rejects_bad_data(self):
        self.assertEqual(validate_data(self.data)['frame_mismatches'],0)
        self.data['state']=self.data['state'].clone();self.data['state'][2,0,0]+=1
        with self.assertRaises(ValueError):validate_data(self.data)


if __name__=='__main__':unittest.main()
