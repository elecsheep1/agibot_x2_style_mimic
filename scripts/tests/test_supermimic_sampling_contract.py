"""世界模型按帧计数，模型/策略/AMP 的批量互不串用。"""
from types import SimpleNamespace
import unittest
import torch
from test_supermimic_getup import runner_module
import test_supermimic_training_entry as entry_tests


@unittest.skipUnless(hasattr(runner_module.SuperMimicRunner, 'SAMPLING_MODE'), '等待 source 拆分批量与窗口约定')
class SamplingContractTests(unittest.TestCase):
    def make_runner(self):
        calls = []; runner = runner_module.SuperMimicRunner.__new__(runner_module.SuperMimicRunner)
        runner.amp_steps = runner.history_length = 5
        runner.collect = lambda steps: (calls.append(('collect', steps)) or {})
        def world(**kwargs):
            calls.append(('world', kwargs['batch_size'], kwargs['horizon']))
            return dict(states=torch.zeros(kwargs['batch_size'], kwargs['horizon']+1, 2))
        runner.sample_world_batch = world
        runner.sample_policy_windows = lambda batch: (calls.append(('amp_policy',batch)) or torch.zeros(batch,5,2))
        counter = []
        def update_world(batch):
            counter.append(1); return {'world_loss': float(len(counter))}
        def update_discriminator(reference, policy):
            self.assertEqual(reference.shape, policy.shape); calls.append(('discriminator',len(policy))); return {}
        def histories(**kwargs):
            size=kwargs['batch_size'];calls.append(('policy',size))
            return dict(history=torch.zeros(size,5,2), context=torch.zeros(size,1), commands=torch.empty(size,0))
        def actor(**kwargs):
            calls.append(('actor',len(kwargs['history']))); return kwargs['history'], {}
        def critic(trajectory,context):
            self.assertEqual(len(trajectory),len(context)); calls.append(('critic',len(context))); return {}
        runner.replay = SimpleNamespace(sample_policy_histories=histories)
        runner.trainer = SimpleNamespace(update_world_model=update_world, update_discriminator=update_discriminator,
                                         update_actor=actor, update_critic=critic)
        def reference(**kwargs):
            calls.append(('amp_reference',kwargs['batch_size']))
            return {'states':torch.zeros(kwargs['batch_size'],kwargs['num_steps'],2)}
        return runner, SimpleNamespace(sample_amp_windows=reference), calls

    def test_independent_batches_seven_transitions_and_mean_world_loss(self):
        runner,reference,calls=self.make_runner()
        result=runner.train_iteration(reference,world_batch_size=3,policy_batch_size=7,amp_batch_size=2,
            world_window_frames=8,world_updates=3,discriminator_updates=2,
            offline_world_replay=object(),offline_world_fraction=.4)
        self.assertEqual(calls.count(('world',3,7)),3)
        self.assertEqual(calls.count(('amp_policy',2)),2)
        self.assertEqual(calls.count(('amp_reference',2)),2)
        self.assertIn(('actor',7),calls);self.assertIn(('critic',7),calls)
        self.assertEqual(result['world_loss'],2.)
        self.assertEqual(result['world/offline_fraction'],1/3)
        self.assertEqual(result['world/transitions_per_window'],7)

    def test_invalid_window_or_batch_never_collects(self):
        for kwargs in ({'world_window_frames':1},{'policy_batch_size':0},{'amp_batch_size':True}):
            runner,reference,calls=self.make_runner()
            with self.assertRaises(ValueError):runner.train_iteration(reference,**kwargs)
            self.assertEqual(calls,[])

    def test_two_frames_mean_one_transition_and_old_shared_argument_rejected(self):
        runner,reference,calls=self.make_runner()
        runner.train_iteration(reference,world_batch_size=2,policy_batch_size=2,amp_batch_size=2,
                               world_window_frames=2,world_updates=1,discriminator_updates=1)
        self.assertIn(('world',2,1),calls)
        with self.assertRaises(TypeError):runner.train_iteration(reference,batch_size=2)


class WarmupWindowContractTests(unittest.TestCase):
    def test_warmup_uses_world_batch_and_converts_frames_once(self):
        runtime,args,_,_=entry_tests.TrainingEntryTests().make_warmup_runtime()
        samples=[]
        runtime.runner.sample_world_batch=lambda *values:(samples.append(values) or {})
        entry_tests.warmup.warmup_world_model(runtime,args,lambda *args:None)
        self.assertTrue(samples)
        self.assertTrue(all(values[:2]==(4,7) for values in samples))


if __name__=='__main__':unittest.main()
