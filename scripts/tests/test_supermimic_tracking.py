"""独立跟踪范式的 CPU 测试：真实数据加载、时序、监督与 BPTT。"""
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'source/x2_style_mimic/x2_style_mimic'
PREFIX = '_tracking_port_tests'
for name,path in ((PREFIX,SOURCE),(PREFIX+'.algorithms',SOURCE/'algorithms'),
                  (PREFIX+'.algorithms.supermimic_tracking',SOURCE/'algorithms/supermimic_tracking'),
                  (PREFIX+'.algorithms.supermimic',SOURCE/'algorithms/supermimic')):
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def load(name):
    return importlib.import_module(PREFIX+'.algorithms.supermimic_tracking.'+name)


rotation = load('rotations')
state_spec = load('state_spec')
heads = load('heads')
nets = load('nets')
world = load('world_model')
replay = load('replay_buffer')
motion = load('motion_library')
wm_trainer = load('wm_trainer')
policy_trainer = load('policy_trainer')


class FakeFK(nn.Module):
    """仅用于训练图连通性测试；不作为 X2 仿真精度证据。"""
    def __init__(self):
        super().__init__()
        self.register_buffer('control_ids',torch.arange(29))

    def body_indices(self,names): return torch.arange(len(names))
    def body_index(self,name): return motion.TRACKED_BODIES.index(name)

    def forward(self,s):
        count = len(motion.TRACKED_BODIES)
        pos = s[...,:3].unsqueeze(-2).expand(*s.shape[:-1],count,3)
        q = s[...,3:7].unsqueeze(-2).expand(*s.shape[:-1],count,4)
        v = s[...,38:41].unsqueeze(-2).expand_as(pos)
        w = s[...,41:44].unsqueeze(-2).expand_as(pos)
        return dict(pos=pos,quat=q,lin_vel=v,ang_vel=w)


def library():
    import json
    import numpy as np
    manifest = ROOT/'data/motions/prone_getup2/motions_birth.json'
    first = json.loads(manifest.read_text())['motions'][0]['file']
    with np.load(manifest.parent/first,allow_pickle=False) as data:
        names = data['joint_names'].tolist()
    return motion.MotionLibrary(manifest,names,torch.zeros(31),FakeFK())


class TrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.lib = library()

    def make_world(self):
        fk = FakeFK()
        h = heads.AnalyticHeads(fk,motion.TRACKED_BODIES,'torso_link','base_link',
            torch.tensor([[-3.,3.]]).repeat(31,1),torch.zeros(31))
        return world.WorldModel(world.WorldModelCfg(hidden=(32,32),term_hidden=(16,16)),h,'cpu')

    def test_x2_dimensions_and_zero_delta_physical_prior(self):
        spec = state_spec.StateSpec(state_spec.StateSpecCfg(),'cpu')
        s = torch.zeros(2,75);s[:,3]=1;s[:,38]=1;s[:,44:]=2
        pred = spec.integrate(s,torch.zeros(2,74))
        self.assertEqual(spec.canonical(s).shape,(2,75))
        torch.testing.assert_close(pred[:,0],torch.full((2,),.02))
        torch.testing.assert_close(pred[:,7:38],torch.full((2,31),.04))
        torch.testing.assert_close(spec.error_vector(s,s),torch.zeros(2,74))

    def test_quaternion_boundary_and_gradient_at_identity(self):
        q = torch.tensor([[1.,0.,0.,0.]])
        torch.testing.assert_close(rotation.to_wxyz(rotation.to_xyzw(q)),q)
        delta = torch.zeros(2,3,requires_grad=True)
        rotation.rotvec_from_quat(rotation.quat_from_rotvec(delta)).sum().backward()
        torch.testing.assert_close(delta.grad,torch.ones_like(delta))
        qyaw = rotation.quat_from_rotvec(torch.tensor([[0.,0.,torch.pi/2]]))
        torch.testing.assert_close(rotation.quat_apply(qyaw,torch.tensor([[1.,0.,0.]])),
                                   torch.tensor([[0.,1.,0.]]),atol=1e-6,rtol=0)

    def test_motion_loading_names_and_goal_clamp(self):
        lib = self.lib
        self.assertEqual(lib.states.shape[-1],75)
        frame = lib.ends[:2]
        goal = lib.goal_window(frame,10).reshape(2,10,72)
        torch.testing.assert_close(goal,goal[:,:1].expand_as(goal))
        self.assertEqual(lib.body_data['pos'].shape[1],14)
        self.assertTrue(torch.isfinite(goal).all())
        torch.testing.assert_close(goal[:,0,:31],lib.states[frame,7:38]-lib.default_joint_pos)

    def test_reference_tracking_has_max_reward_and_dt_once(self):
        w = self.make_world()
        s = self.lib.states[:3]
        ref = wm_trainer.reference_at(self.lib,torch.arange(3))
        # 使用同一个 FK 与参考构造完全匹配的数据。
        w.heads.soft_limits = torch.tensor([[-100.,100.]]).repeat(31,1)
        terms = w.heads.reward(s,torch.zeros(3,29),torch.zeros(3,29),ref)
        torch.testing.assert_close(terms['total'],torch.full((3,),.1))
        changed = w.heads.reward(s,torch.ones(3,29),torch.zeros(3,29),ref)
        torch.testing.assert_close(changed['action_rate_l2'],torch.full((3,),-.1*.02*29))

    def test_replay_terminal_row_and_window_unit(self):
        b = replay.ReplayBuffer(1,32,75,29,35,'cpu')
        ids = torch.tensor([0]); s = torch.zeros(1,75);s[:,3]=1
        for i in range(9):
            state = s.clone();state[:,0]=i
            b.add(ids,state,torch.zeros(1,29),torch.full((1,29),.25),torch.zeros(1,35),torch.zeros(1),
                  torch.tensor([float(i==7)]),torch.tensor([i]),ids,torch.tensor([i==8]),torch.tensor([i==0]))
        b.end_episode(ids)
        b.add(ids,s+100,torch.zeros(1,29),torch.zeros(1,29),torch.zeros(1,35),torch.zeros(1),
              torch.zeros(1),ids,ids,torch.tensor([False]),torch.tensor([True]))
        window = b.sample_windows(4,8)
        self.assertEqual(window.states.shape,(9,4,75))
        torch.testing.assert_close(window.states[-1,:,0],torch.full((4,),8.))
        torch.testing.assert_close(window.a_exec,torch.full((8,4,29),.25))

    def test_world_model_supervision_has_termination_gradient(self):
        w = self.make_world()
        s = self.lib.states[:4]
        states = s.unsqueeze(0).repeat(9,1,1)
        actions = torch.randn(8,4,29)*.1
        context = torch.zeros(4,35);context[:,0]=.75
        refs = [wm_trainer.reference_at(self.lib,torch.arange(4)) for _ in range(8)]
        done = torch.zeros(8,4);done[-1,0]=1
        loss,metrics = w.window_loss(states,actions,context,done,refs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in w.term_head.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in w.parameters() if p.grad is not None))

    def test_imagined_actor_update_freezes_world_and_updates_critic(self):
        torch.manual_seed(9)
        w = self.make_world()
        # 模拟已拟合后的动作导数；零输出初始模型本来没有动作导数。
        nn.init.normal_(w.dynamics.mlp.net[-1].weight,std=.002)
        pcfg = nets.PolicyCfg(encoder_hidden=(32,),decoder_hidden=(32,))
        actor = nets.Policy(pcfg)
        critic = nets.Critic(nets.CriticCfg(hidden=(32,)),pcfg=pcfg)
        cfg = policy_trainer.PolicyTrainerCfg(horizon=4,batch_size=4,critic_epochs=1,critic_minibatches=2,
                                               compile=None,obs_noise=False)
        trainer = policy_trainer.PolicyTrainer(w,actor,critic,self.lib,cfg)
        b = replay.ReplayBuffer(2,40,75,29,35,'cpu')
        ids = torch.arange(2)
        for i in range(15):
            frames = torch.tensor([i,i+30])
            b.add(ids,self.lib.states[frames],torch.zeros(2,29),torch.zeros(2,29),torch.zeros(2,35),
                  torch.zeros(2),torch.zeros(2),frames,torch.zeros(2,dtype=torch.long),
                  torch.zeros(2,dtype=torch.bool),torch.full((2,),i==0))
        before_world = [p.clone() for p in w.parameters()]
        before_actor = [p.clone() for p in actor.parameters()]
        before_critic = [p.clone() for p in critic.parameters()]
        metrics = trainer.update(b)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before_actor,actor.parameters())))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before_critic,critic.parameters())))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before_world,w.parameters())))
        self.assertTrue(all(p.requires_grad for p in w.parameters()))

    def test_reward_gradient_reaches_state(self):
        w = self.make_world()
        s = self.lib.states[:2].clone().requires_grad_()
        ref = wm_trainer.reference_at(self.lib,torch.arange(2))
        offset = s + torch.nn.functional.pad(torch.full((2,1),.05),(2,72))
        reward = w.heads.reward(offset,torch.zeros(2,29),torch.zeros(2,29),ref)['total']
        reward.sum().backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertGreater(float(s.grad[:,:3].abs().sum()),0)

    def test_default_cfg_and_smoke_cfg(self):
        import importlib.util
        config = load('config').TrackingTrainCfg()
        config.validate()
        spec = importlib.util.spec_from_file_location('_tracking_entry',ROOT/'scripts/supermimic_tracking/train.py')
        entry = importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        import tomllib
        entry.update_config(config,tomllib.loads((ROOT/'scripts/supermimic_tracking/configs/smoke.toml').read_text()))
        config.validate()
        self.assertIsNone(config.wm_trainer.compile)
        with self.assertRaises(ValueError):entry.update_config(config,{'bad_key':1})

    def test_real_x2_fk_state_adapter_and_velocity(self):
        import json
        import numpy as np
        manifest = ROOT/'data/motions/prone_getup2/motions_birth.json'
        first = json.loads(manifest.read_text())['motions'][0]['file']
        with np.load(manifest.parent/first,allow_pickle=False) as data:
            names = data['joint_names'].tolist()
        urdf = Path('/home/panda/robotics_ws/agibot_model/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra.urdf')
        if not urdf.is_file():self.skipTest('本机未安装 X2 URDF')
        fk = motion.X2Kinematics(urdf,names,torch.arange(29))
        s = self.lib.states[100:102].double()
        fk = fk.double()
        predicted = fk(s)
        spec = state_spec.StateSpec(state_spec.StateSpecCfg(dt=1e-5),'cpu',dtype=torch.float64)
        following = fk(spec.integrate(s,torch.zeros(2,74,dtype=torch.float64)))
        measured = (following['pos']-predicted['pos'])/1e-5
        torch.testing.assert_close(measured,predicted['lin_vel'],atol=2e-4,rtol=2e-4)
        self.assertEqual(predicted['pos'].shape,(2,14,3))

    def test_step_captures_terminal_before_reset_and_advances_only_survivors(self):
        import ast
        from types import SimpleNamespace
        path = SOURCE/'tasks/manager_based/x2_supermimic_tracking/env.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef))
        class Base:
            def step(self,action):
                self.action_term.last_executed = action+.2
                self.state = self.state+1
                self.reward_buf = torch.tensor([.1,.2])
                self._reset_idx(torch.tensor([0]))
                return {'goal':torch.zeros(2,720)},self.reward_buf,torch.tensor([True,False]),torch.zeros(2,dtype=torch.bool),{}
            def _reset_idx(self,ids):
                self.state[ids] = -100
                self.command.frame[ids] = 40
                self.action_term.raw_actions[ids] = 0
        scope = dict(torch=torch,ManagerBasedRLEnv=Base)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
        env = scope['X2SuperMimicTrackingEnv'].__new__(scope['X2SuperMimicTrackingEnv'])
        env.num_envs,env.device = 2,'cpu'
        env.state = torch.zeros(2,75)
        env.read_tracking_state = lambda:env.state.clone()
        env.action_term = SimpleNamespace(raw_actions=torch.ones(2,29),last_executed=torch.zeros(2,29))
        calls = []
        env.command = SimpleNamespace(frame=torch.tensor([2,5]),xi=torch.zeros(2,35),previous_action=torch.zeros(2,29),
            library=SimpleNamespace(last_frame_of=lambda frames:torch.full_like(frames,100)),
            command=torch.zeros(2,720),log_rewards=lambda ids:calls.extend(ids.tolist()))
        env.action_manager = SimpleNamespace(get_term=lambda name:env.action_term)
        env.command_manager = SimpleNamespace(get_term=lambda name:env.command)
        _,_,_,_,extras = env.step(torch.zeros(2,29))
        t = extras['tracking_transition']
        torch.testing.assert_close(t['next_state'],torch.ones(2,75))
        torch.testing.assert_close(t['frame'],torch.tensor([2,5]))
        torch.testing.assert_close(env.command.frame,torch.tensor([40,6]))
        torch.testing.assert_close(t['executed'],torch.full((2,29),.2))
        self.assertEqual(calls,[0,1])

    def test_sampler_uses_discounted_return_to_go_per_visited_bin(self):
        import ast
        from types import SimpleNamespace
        path = SOURCE/'tasks/manager_based/x2_supermimic_tracking/mdp.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='TrackingCommand')
        method = next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='_finalize_returns')
        scope = {'torch':torch}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
        obj = SimpleNamespace(device='cpu',length=torch.tensor([3.]),
            episode_rewards=torch.tensor([[.1,.1,.1,0.]]),episode_bins=torch.tensor([[0,1,2,0]]),
            gamma_powers=.99**torch.arange(4,dtype=torch.float64),bin_return=torch.zeros(3),frame=torch.tensor([3]),
            library=SimpleNamespace(last_frame_of=lambda frame:torch.tensor([3])),
            _env=SimpleNamespace(termination_manager=SimpleNamespace(terminated=torch.tensor([False]))),
            cfg=SimpleNamespace(min_remaining_steps=100,ema_alpha=1.))
        scope['_finalize_returns'](obj,torch.tensor([0]))
        torch.testing.assert_close(obj.bin_return,torch.tensor([.1+.99*.1+.99**2*.1,.1+.99*.1,.1]))


if __name__=='__main__': unittest.main()
