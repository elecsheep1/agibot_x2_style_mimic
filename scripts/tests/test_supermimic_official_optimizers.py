"""官方 Adam/Muon 分组、CPU 更新和完整优化器状态恢复。"""
import copy
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from test_supermimic_world_model import load, SOURCE
from test_supermimic_world_model import state, wm
from test_supermimic_training_entry import checkpointing

optimizer_module = load('optimizers') if (SOURCE / 'optimizers.py').is_file() else None


def network():
    return nn.Sequential(nn.Linear(3, 8), nn.LayerNorm(8), nn.GELU(),
                         nn.Linear(8, 8), nn.LayerNorm(8), nn.GELU(), nn.Linear(8, 2))


@unittest.skipUnless(optimizer_module is not None, '等待 source 接入官方分组优化器')
class OfficialOptimizerTests(unittest.TestCase):
    def test_trainer_world_update_steps_both_optimizers(self):
        trainer_module = load('trainer')
        spec = state.StateSpec(2, 2)
        world = wm.WorldModel(spec, 1, wm.TargetLimiter(torch.zeros(2), .25, -torch.ones(2), torch.ones(2), .1),
                              (8, 8, 8), dt=.02, decimation=10)
        states = torch.zeros(2, 3, spec.dim); states[:, :, 6] = 1.
        batch = dict(states=states, actions=torch.ones(2, 2, 2)*.1, context=torch.ones(2, 1))
        world.fit_input_stats(batch)
        task = nn.Identity()
        task.max_step_reward = .4
        trainer = trainer_module.SuperMimicTrainer(network(), network(), world, nn.Identity(),
                                                  nn.Identity(), nn.Identity(), task)
        self.assertAlmostEqual(trainer.value_max, 40.)
        metrics = trainer.update_world_model(batch)
        self.assertTrue(torch.isfinite(torch.tensor(metrics['world_loss'])))
        self.assertEqual(len(trainer.world_optimizers), 2)
        self.assertTrue(all(len(optimizer.state) > 0 for optimizer in trainer.world_optimizers))
        self.assertTrue(all(p.grad is None for p in trainer.actor.parameters()))

    def test_partition_is_disjoint_complete_and_keeps_edges_in_adam(self):
        model = network(); optimizers = optimizer_module.build_optimizers(model, lr=2e-4)
        self.assertIsInstance(optimizers[0], torch.optim.Adam)
        self.assertIsInstance(optimizers[1], torch.optim.Muon)
        sets = [{id(p) for group in optimizer.param_groups for p in group['params']} for optimizer in optimizers]
        self.assertFalse(sets[0] & sets[1])
        self.assertEqual(sets[0] | sets[1], {id(p) for p in model.parameters()})
        self.assertEqual(sets[1], {id(model[3].weight)})
        self.assertIn(id(model[0].weight), sets[0]); self.assertIn(id(model[-1].weight), sets[0])
        self.assertIn(id(model[4].weight), sets[0]); self.assertIn(id(model[3].bias), sets[0])

    def test_no_middle_linear_uses_only_adam(self):
        model = nn.Sequential(nn.Linear(3, 8), nn.ELU(), nn.Linear(8, 2))
        optimizers = optimizer_module.build_optimizers(model, lr=2e-4)
        self.assertEqual(len(optimizers), 1)
        self.assertIsInstance(optimizers[0], torch.optim.Adam)

    def test_cpu_step_and_resume_preserve_both_optimizer_states(self):
        torch.manual_seed(2); model = network(); x = torch.randn(4, 3)
        optimizers = optimizer_module.build_optimizers(model, lr=2e-4, betas=(.9, .95))
        def step(model, optimizers):
            for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            for optimizer in optimizers: optimizer.step()
        step(model, optimizers)
        restored = copy.deepcopy(model)
        restored_optimizers = optimizer_module.build_optimizers(restored, lr=2e-4, betas=(.9, .95))
        for actual, saved in zip(restored_optimizers, optimizers):
            actual.load_state_dict(copy.deepcopy(saved.state_dict()))
            self.assertGreater(len(actual.state), 0)
        step(model, optimizers); step(restored, restored_optimizers)
        for a, b in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertTrue(torch.isfinite(a).all())

    def test_invalid_learning_rate_and_betas_rejected(self):
        for kwargs in ({'lr':0.}, {'lr':float('nan')}, {'lr':.1,'betas':(.9,1.)}):
            with self.assertRaises(ValueError): optimizer_module.build_optimizers(network(), **kwargs)


class OptimizerCheckpointMappingTests(unittest.TestCase):
    def test_grouped_and_legacy_optimizers_are_all_registered(self):
        def adam(): return torch.optim.Adam(nn.Linear(2, 1).parameters())
        grouped = SimpleNamespace(world_optimizers=[adam(), adam()], actor_optimizers=[adam(), adam()], critic_optimizers=[adam()])
        discriminator = SimpleNamespace(optimizer=adam())
        result = checkpointing.collect_optimizers(grouped, discriminator)
        self.assertEqual(set(result), {'world/0','world/1','actor/0','actor/1','critic/0','discriminator'})
        self.assertIs(result['world/1'], grouped.world_optimizers[1])
        legacy = SimpleNamespace(world_optimizer=adam(), actor_optimizer=adam(), critic_optimizer=adam())
        self.assertEqual(set(checkpointing.collect_optimizers(legacy, discriminator)), {'world','actor','critic','discriminator'})


if __name__ == '__main__': unittest.main()
