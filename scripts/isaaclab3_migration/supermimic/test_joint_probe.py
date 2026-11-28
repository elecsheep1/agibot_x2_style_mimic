"""短程联合入口的模块加载隔离、固定评估与状态恢复测试。"""
import copy
from types import SimpleNamespace as NS
import unittest
import torch
from joint_probe import (ProbeCfg, restore_modules, evaluation_frames, evaluate_real,
                         configure_failure_head, assert_head_unchanged)


class ConstantPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
    def forward(self,observation,goal):return observation*0+self.weight


class JointProbeTests(unittest.TestCase):
    def test_freeze_preserves_input_gradient_and_blocks_optimizer_update(self):
        from test_failure_supervision import wm
        from importlib import import_module
        frozen_parameters=import_module(wm.__package__+'.gradients').frozen_parameters
        training=NS(world=NS(term_head=wm.TerminationHead((16,))))
        optimizer=torch.optim.Adam(training.world.term_head.parameters(),lr=.01)
        initial=configure_failure_head(training,True)
        features=torch.randn(8,2,requires_grad=True)
        with frozen_parameters(training.world.term_head):
            loss=training.world.term_head(features).sum()
        optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertGreater(float(features.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in training.world.term_head.parameters()))
        assert_head_unchanged(training,initial)
        with torch.no_grad():next(training.world.term_head.parameters()).add_(1.)
        with self.assertRaises(RuntimeError):assert_head_unchanged(training,initial)

    def test_restore_only_parameters_and_head_from_base(self):
        training=NS(contract={'asset':'x2'},policy=torch.nn.Linear(2,1),critic=torch.nn.Linear(2,1),
                    world=torch.nn.Module())
        training.world.dynamics=torch.nn.Linear(2,1);training.world.term_head=torch.nn.Linear(2,1)
        base=dict(kind='x2_reproduced_tracking_v1',contract=training.contract,
                  policy=copy.deepcopy(training.policy.state_dict()),critic=copy.deepcopy(training.critic.state_dict()),
                  world=copy.deepcopy(training.world.state_dict()))
        dynamics=dict(kind='fixed_dynamics_refit_v1',world=copy.deepcopy(training.world.state_dict()))
        dynamics['world']['dynamics.weight'].fill_(2.)
        dynamics['world']['term_head.weight'].fill_(99.)
        training.world_trainer=NS(opt=torch.optim.Adam(training.world.parameters()))
        training.policy_trainer=NS(opt=torch.optim.Adam(training.policy.parameters()),
                                   critic_opt=torch.optim.Adam(training.critic.parameters()))
        restore_modules(training,base,dynamics)
        torch.testing.assert_close(training.world.dynamics.weight,torch.full_like(training.world.dynamics.weight,2.))
        torch.testing.assert_close(training.world.term_head.weight,base['world']['term_head.weight'])
        base['contract']={'asset':'other'}
        with self.assertRaises(ValueError):restore_modules(training,base,dynamics)

    def test_fixed_frame_selection_is_reproducible(self):
        lib=NS(starts=torch.tensor([0,50,100]),states=torch.zeros(150,75),
               last_frame_of=lambda f:(f//50+1)*50-1)
        a=evaluation_frames(lib,8,42);b=evaluation_frames(lib,8,42)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(a[:4],torch.tensor([0,50,100,0]))
        self.assertTrue(bool((a<=lib.last_frame_of(a)-2).all()))

    def test_real_evaluation_counts_first_episode_and_restores_state(self):
        command=NS(fixed_reset_frames=None,length=torch.ones(2),bin_return=torch.ones(3),bin_memory=torch.ones(3),
                   heads=NS(reward_cfg=NS(max_step_reward=1.)))
        term=NS(cfg=NS(noise_std=.1));steps=[0]
        def reset():
            steps[0]=0;command.bin_return.add_(5)
            return {'policy':torch.zeros(2,1),'goal':torch.zeros(2,1)},{}
        def step(action):
            steps[0]+=1;torch.rand(3)
            command.bin_memory.add_(1)
            return dict(policy=torch.zeros(2,1),goal=torch.zeros(2,1)),torch.tensor([1.,2.]),torch.tensor([True,False]),torch.tensor([False,False]),{}
        env=NS(num_envs=2,device='cpu',step_dt=.02,reset=reset,step=step,
               action_manager=NS(get_term=lambda name:term))
        training=NS(env=env,command=command,policy=ConstantPolicy(),cfg=NS(policy=NS(action_lower=(-1.,),action_upper=(1.,))))
        rng=torch.get_rng_state().clone()
        result=evaluate_real(training,torch.tensor([0,10]),.06)
        self.assertEqual(result['failure_rate'],.5)
        self.assertEqual(result['censored_rate'],.5)
        self.assertAlmostEqual(result['normalized_return'],1.5)
        self.assertAlmostEqual(result['episodes'][0]['seconds'],.02)
        torch.testing.assert_close(command.bin_return,torch.ones(3))
        torch.testing.assert_close(command.bin_memory,torch.ones(3))
        self.assertEqual(term.cfg.noise_std,.1)
        self.assertIsNone(command.fixed_reset_frames)
        self.assertTrue(training.policy.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_probe_budget_validation(self):
        for cfg in (ProbeCfg(updates=0),ProbeCfg(evaluation_seconds=float('nan'))):
            with self.assertRaises(ValueError):cfg.validate()


if __name__=='__main__':unittest.main()
