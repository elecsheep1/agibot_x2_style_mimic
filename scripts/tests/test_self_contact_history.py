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
reward_ns={'torch':torch};load_nodes(MDP/'rewards.py',lambda n:isinstance(n,ast.FunctionDef) and n.name=='self_contact_history_count',reward_ns)
count=reward_ns['self_contact_history_count']
def sensor():
    s=ns['SelfContactSensor'].__new__(ns['SelfContactSensor']);s.device='cpu';s._num_envs=2;s._body_count=3;s._sim_physics_dt=.002;s._window_open=True
    s.cfg=SimpleNamespace(force_threshold=10.,duration_start=.1,validate_contact_buffer=False)
    s._i,s._j=torch.triu_indices(3,3,1);s.duration=torch.zeros(2,3);s.peak_force=torch.zeros(2,3);s._persistent=torch.zeros(2,3,dtype=torch.bool);s._max_duration=torch.zeros(2);s.force_history=torch.zeros(2,4);s._updates=torch.zeros(2,dtype=torch.long)
    s.matrix=torch.zeros(2,3,2,3);s._view=SimpleNamespace(get_contact_force_matrix=lambda dt:s.matrix)
    s._resolve_indices_and_mask=lambda env_ids,env_mask:env_mask if env_mask is not None else torch.tensor([i in env_ids for i in range(2)])
    return s
class TestHistory(unittest.TestCase):
    def test_directional_pair_merge_and_multiple_contacts_count_once(self):
        s=sensor();s.matrix[0,0,0,0]=11;s.matrix[0,1,0,0]=20;s.matrix[0,0,1,0]=30
        s._update_buffers_impl(torch.tensor([True,True]));self.assertEqual(s.peak_force[0].tolist(),[20.,30.,0.]);self.assertEqual(count(s.force_history).tolist(),[1.,0.])
    def test_boundary_and_last_four_physical_samples(self):
        s=sensor()
        for force in [100.,100.,100.,100.,0.,0.,10.,11.,12.,0.]:
            s.matrix.zero_();s.matrix[0,0,0,0]=force;s._update_buffers_impl(torch.tensor([True,True]))
        self.assertEqual(s.force_history[0].tolist(),[0.,12.,11.,10.]);self.assertEqual(count(s.force_history).tolist(),[2.,0.]);self.assertEqual(s._updates.tolist(),[10,10]);self.assertEqual(float(s.peak_force[0].max()),100.)
    def test_masked_update_and_reset_do_not_touch_other_env(self):
        s=sensor();s.force_history[1]=torch.tensor([1.,2.,30.,40.]);s.matrix[0,0,0,0]=25
        s._update_buffers_impl(torch.tensor([True,False]));self.assertEqual(s.force_history[1].tolist(),[1.,2.,30.,40.]);s.reset(env_ids=[0]);self.assertEqual(s.force_history[0].tolist(),[0.]*4);self.assertEqual(s.force_history[1].tolist(),[1.,2.,30.,40.])
    def test_control_boundary_keeps_history_and_duration(self):
        s=sensor();s.force_history.fill_(20);s.duration.fill_(.05);s.end_control_window();s.begin_control_window();self.assertEqual(count(s.force_history).tolist(),[4.,4.]);self.assertTrue(torch.allclose(s.duration,torch.full((2,3),.05)));self.assertEqual(s._updates.tolist(),[0,0])
    def test_nonfinite_force_rejected(self):
        s=sensor();s.matrix[0,0,0,0]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'非有限'):s._update_buffers_impl(torch.tensor([True,True]))
    def test_wrong_time_step_rejected(self):
        s=sensor()
        with self.assertRaisesRegex(RuntimeError,'physics_dt'):s.update(.02)
if __name__=='__main__':unittest.main()
