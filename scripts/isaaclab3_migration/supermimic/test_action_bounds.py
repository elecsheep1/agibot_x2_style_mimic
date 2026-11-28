"""动作边界修改的 CPU 合同测试；源文件语法检查单独保留，不启动仿真。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT/'source/x2_style_mimic/x2_style_mimic'
CORE = SOURCE/'algorithms/supermimic_tracking'
MDP = SOURCE/'tasks/manager_based/x2_supermimic_tracking/mdp.py'


def extract(path,names,scope):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)


def decoder():
    # 只隔离检查修改的数学逻辑；完整文件能否导入由 syntax 测试检查。
    scope = dict(torch=torch,nn=nn,ACT_DIM=29,PolicyCfg=SimpleNamespace,_obs_scale_vector=lambda cfg:torch.ones(930))
    extract(CORE/'dynamics.py',{'MLP'},scope)
    extract(CORE/'nets.py',{'Decoder'},scope)
    cfg = SimpleNamespace(latent_dim=32,history=10,decoder_hidden=(16,),decoder_last_gain=.01,
                          action_lower=(-2.,)*29,action_upper=(3.,)*29)
    return scope['Decoder'](cfg)


class MockJointAction:
    def __init__(self,cfg,env):
        self.cfg,self._joint_ids = cfg,torch.tensor([2,0])
        self.raw_actions = torch.zeros(2,2)
        self._raw_actions = self.raw_actions
        self._scale,self._offset = .25,torch.tensor([[.05,-.1]])
        self._clip = torch.tensor([[[-.1,.2],[-.7,.8]]]).expand(2,-1,-1).clone()
    def reset(self,env_ids=None):self._raw_actions[env_ids]=0


def action_term(clip=None,noise=.1):
    scope = dict(torch=torch,JointPositionAction=MockJointAction,tensor=lambda value:value)
    extract(MDP,{'NoisyJointPositionAction'},scope)
    limits = torch.tensor([[[-1.,1.],[-2.,2.],[-.3,.3]]]).expand(2,-1,-1).clone()
    env = SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(joint_pos_limits=limits))})
    return scope['NoisyJointPositionAction'](SimpleNamespace(asset_name='robot',noise_std=noise,clip=clip),env)


class ActionBoundTests(unittest.TestCase):
    def test_source_syntax(self):
        for name in ('compiled.py','dynamics.py','nets.py','training.py'):
            with self.subTest(file=name):compile((CORE/name).read_text(),str(CORE/name),'exec')
        compile(MDP.read_text(),str(MDP),'exec')

    def test_decoder_respects_asymmetric_bounds(self):
        d = decoder()
        class Constant(nn.Module):
            def forward(self,x):return torch.linspace(-100,100,29).expand(x.shape[0],-1)
        d.mlp = Constant()
        output = d(torch.zeros(2,32),torch.zeros(2,930))
        self.assertTrue(bool((output>=-2).all() and (output<=3).all()))
        self.assertAlmostEqual(float(output[0,0]),-2.,places=6)
        self.assertAlmostEqual(float(output[0,-1]),3.,places=6)

    def test_zero_output_has_unit_gradient(self):
        d = decoder()
        class Raw(nn.Module):
            def __init__(self):super().__init__();self.value=nn.Parameter(torch.zeros(1,29))
            def forward(self,x):return self.value.expand(x.shape[0],-1)
        d.mlp = Raw()
        output=d(torch.zeros(1,32),torch.zeros(1,930))
        torch.testing.assert_close(output,torch.zeros_like(output))
        output.sum().backward()
        torch.testing.assert_close(d.mlp.value.grad,torch.ones_like(output))

    def test_noisy_execution_is_clipped_and_recorded_exactly(self):
        term=action_term(noise=10.)
        torch.manual_seed(4)
        original=torch.tensor([[-100.,100.],[100.,-100.]])
        term.process_actions(original)
        self.assertTrue(bool((term._processed_actions>=term.target_lower).all()))
        self.assertTrue(bool((term._processed_actions<=term.target_upper).all()))
        torch.testing.assert_close(term.last_executed*term._scale+term._offset,term._processed_actions)
        torch.testing.assert_close(term.raw_actions,original)
        before=term.last_executed.clone()
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term.last_executed,before)

    def test_clip_intersection_and_joint_reordering(self):
        term=action_term(clip={'configured':True},noise=0.)
        torch.testing.assert_close(term.target_lower,torch.tensor([[-.1,-.7],[-.1,-.7]]))
        torch.testing.assert_close(term.target_upper,torch.tensor([[.2,.8],[.2,.8]]))

    def test_runtime_calibration_reconstructs_target_bounds(self):
        node=next(n for n in ast.parse((CORE/'training.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='TrackingTraining')
        init=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        start=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='action_term' for t in n.targets))
        end=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='policy' for t in n.targets))
        term=action_term(noise=0.)
        env=SimpleNamespace(device='cpu',action_manager=SimpleNamespace(get_term=lambda name:term))
        cfg=SimpleNamespace(policy=SimpleNamespace(action_lower=None,action_upper=None))
        exec(compile(ast.Module(body=init.body[start:end],type_ignores=[]),'calibration','exec'),dict(torch=torch,env=env,cfg=cfg))
        torch.testing.assert_close(torch.tensor(cfg.policy.action_lower)*term._scale+term._offset[0],term.target_lower[0])
        torch.testing.assert_close(torch.tensor(cfg.policy.action_upper)*term._scale+term._offset[0],term.target_upper[0])


if __name__=='__main__':unittest.main()
