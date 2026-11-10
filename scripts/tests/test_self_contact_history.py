"""Exercise production tensor update/reset methods without launching SimulationApp."""
import ast,math,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2]
MDP=ROOT/'source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic/mdp'
class Base:
    is_initialized=True
    def reset(self,**kwargs):pass

def load_nodes(path,predicate,namespace):
    tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if predicate(n)]
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),namespace)
ns={'torch':torch,'math':math,'wp':SimpleNamespace(to_torch=lambda x:x),'SensorBase':Base}
load_nodes(MDP/'self_contact.py',lambda n:isinstance(n,ast.ClassDef),ns)
reward_ns={'torch':torch};load_nodes(MDP/'rewards.py',lambda n:isinstance(n,ast.FunctionDef) and n.name in ('self_contact_history_count', 'self_contact_pressure_cost', 'self_contact_control_cost'),reward_ns)
count=reward_ns['self_contact_history_count']
ns['self_contact_pressure_cost']=reward_ns['self_contact_pressure_cost']
ns['self_contact_control_cost']=reward_ns['self_contact_control_cost']
def sensor():
    s=ns['SelfContactSensor'].__new__(ns['SelfContactSensor']);s.device='cpu';s._num_envs=2;s._body_count=3;s._sim_physics_dt=.002;s._window_open=True
    s.cfg=SimpleNamespace(force_threshold=10.,duration_start=.1,validate_contact_buffer=False,pressure_force_threshold=150.,pressure_force_scale=300.,pressure_grace_seconds=.05,pressure_ramp_seconds=.2,pressure_release_threshold=100.,pressure_release_rate=.5,impact_force_threshold=800.,impact_force_scale=800.,impact_weight=.1)
    s._i,s._j=torch.triu_indices(3,3,1);s.duration=torch.zeros(2,3);s.peak_force=torch.zeros(2,3);s._persistent=torch.zeros(2,3,dtype=torch.bool);s._max_duration=torch.zeros(2);s.strong_duration=torch.zeros(2,3);s._strong_active=torch.zeros(2,3,dtype=torch.bool);s.pressure_cost_sum=torch.zeros(2);s._updates=torch.zeros(2,dtype=torch.long)
    s.matrix=torch.zeros(2,3,2,3);s._view=SimpleNamespace(get_contact_force_matrix=lambda dt:s.matrix)
    s._resolve_indices_and_mask=lambda env_ids,env_mask:env_mask if env_mask is not None else torch.tensor([i in env_ids for i in range(2)])
    return s
class TestHistory(unittest.TestCase):
    def test_directional_pair_merge(self):
        s=sensor();s.matrix[0,0,0,0]=11;s.matrix[0,1,0,0]=20;s.matrix[0,0,1,0]=30
        s._update_buffers_impl(torch.tensor([True,True]));self.assertEqual(s.peak_force[0].tolist(),[20.,30.,0.]);self.assertEqual(s.pressure_cost_sum.tolist(),[0.,0.])
    def test_substep_count_and_peak(self):
        s=sensor()
        for force in [100.,100.,100.,100.,0.,0.,10.,11.,12.,0.]:
            s.matrix.zero_();s.matrix[0,0,0,0]=force;s._update_buffers_impl(torch.tensor([True,True]))
        self.assertEqual(s.pressure_cost_sum.tolist(),[0.,0.]);self.assertEqual(s._updates.tolist(),[10,10]);self.assertEqual(float(s.peak_force[0].max()),100.)
    def test_masked_update_and_reset_do_not_touch_other_env(self):
        s=sensor();s.strong_duration[1].fill_(.5);s.pressure_cost_sum[1]=2.;s.matrix[0,0,0,0]=25
        s._update_buffers_impl(torch.tensor([True,False]));self.assertEqual(s.strong_duration[1].tolist(),[.5]*3);s.reset(env_ids=[0]);self.assertEqual(s.strong_duration[0].tolist(),[0.]*3);self.assertEqual(s.pressure_cost_sum.tolist(),[0.,2.])
    def test_control_boundary_keeps_duration(self):
        s=sensor();s.strong_duration.fill_(.05);s.pressure_cost_sum.fill_(2.);s.duration.fill_(.05);s.end_control_window();s.begin_control_window();self.assertTrue(torch.allclose(s.strong_duration,torch.full((2,3),.05)));self.assertEqual(s.pressure_cost_sum.tolist(),[0.,0.]);self.assertTrue(torch.allclose(s.duration,torch.full((2,3),.05)));self.assertEqual(s._updates.tolist(),[0,0])
    def test_nonfinite_force_rejected(self):
        s=sensor();s.matrix[0,0,0,0]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'非有限'):s._update_buffers_impl(torch.tensor([True,True]))
    def test_wrong_time_step_rejected(self):
        s=sensor()
        with self.assertRaisesRegex(RuntimeError,'physics_dt'):s.update(.02)
class Pressure(unittest.TestCase):

    def test_formula(self):
        f = reward_ns['self_contact_pressure_cost']
        out = f(torch.tensor([[100.0, 300.0, 450.0, 1400.0]]), torch.tensor([[1.0, 0.2, 0.3, 0.05]]), 150.0, 300.0, 0.1, 0.2)
        self.assertTrue(torch.allclose(out, torch.tensor([1.25])))

    def test_memory_decay_and_reset(self):
        s = sensor()
        mask = torch.tensor([True, True])
        s.matrix[0, 0, 0, 0] = 450.0
        for i in range(20):
            s._update_buffers_impl(mask)
        self.assertEqual(s.pressure_cost_sum[0].item(), 0.0)
        s.end_control_window()
        s.begin_control_window()
        self.assertAlmostEqual(s.strong_duration[0, 0].item(), 0.04, places=6)
        for i in range(110):
            s._update_buffers_impl(mask)
        self.assertGreater(s.pressure_cost_sum[0].item(), 0.0)
        s.matrix.zero_()
        s._update_buffers_impl(mask)
        self.assertAlmostEqual(s.strong_duration.max().item(), 0.249, places=6)
        s.strong_duration[1] = 0.5
        s.pressure_cost_sum[1] = 2.0
        s.reset(env_ids=[0])
        self.assertEqual(s.pressure_cost_sum.tolist(), [0.0, 2.0])
        self.assertEqual(s.strong_duration[1, 0].item(), 0.5)

    def test_mask_and_pair_switch(self):
        s = sensor()
        mask = torch.tensor([True, False])
        s.matrix[:, 0, 0, 0] = 450.0
        for i in range(20):
            s._update_buffers_impl(mask)
        s.matrix.zero_()
        s.matrix[:, 0, 1, 0] = 450.0
        for i in range(20):
            s._update_buffers_impl(mask)
        self.assertEqual(s.pressure_cost_sum.sum().item(), 0.0)
        self.assertEqual(s.strong_duration[1].sum().item(), 0.0)

class MemoryAndImpact(unittest.TestCase):
    def test_hysteresis_recontact_and_release(self):
        s = sensor()
        mask = torch.tensor([True, False])
        def step(force, count=1):
            s.matrix.zero_()
            s.matrix[0, 0, 0, 0] = force
            for _ in range(count):
                s._update_buffers_impl(mask)
        step(450., 40)
        self.assertGreater(s.pressure_cost_sum[0].item(), 0.)
        step(120.)
        self.assertTrue(s._strong_active[0, 0])
        before = s.strong_duration[0, 0].item()
        step(0.)
        self.assertFalse(s._strong_active[0, 0])
        self.assertAlmostEqual(s.strong_duration[0, 0].item(), before - .001, places=6)
        step(450.)
        self.assertGreater(s.strong_duration[0, 0].item(), before)
        step(450., 200)
        self.assertAlmostEqual(s.strong_duration[0, 0].item(), .25, places=6)
        step(0., 260)
        self.assertEqual(s.strong_duration.sum().item(), 0.)
        self.assertFalse(s._strong_active[1].any())

    def test_peak_impact_snapshot_and_reset(self):
        s = sensor()
        mask = torch.tensor([True, True])
        s.matrix[0, 0, 0, 0] = 1600.
        s._update_buffers_impl(mask)
        s.matrix.zero_()
        for _ in range(9):
            s._update_buffers_impl(mask)
        expected = torch.tensor([.1, 0.])
        self.assertTrue(torch.allclose(s.snapshot()['penalty_raw'], expected))
        self.assertTrue(torch.equal(s.snapshot()['penalty_raw'], reward_ns['self_contact_control_cost'](s)))
        s._strong_active[0] = True
        s.reset(env_ids=[0])
        self.assertFalse(s._strong_active[0].any())
        self.assertEqual(s.snapshot()['penalty_raw'].sum().item(), 0.)

if __name__=='__main__':unittest.main()
