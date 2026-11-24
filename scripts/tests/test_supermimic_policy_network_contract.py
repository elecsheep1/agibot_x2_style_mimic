"""配置宽度、动作输入梯度和 Adam/Muon 参数分组的集成检查。"""
import unittest
import torch
from torch import nn
from test_supermimic_world_model import load
from test_supermimic_entry_config import config

networks = load('actor_critic')
optimizers = load('optimizers')


class PolicyNetworkContractTests(unittest.TestCase):
    def test_configured_widths_and_actor_backward(self):
        cfg = config.runner_defaults()
        actor = networks.GoalConditionedActor(
            465, 0, 29, action_lower=-torch.ones(29),
            action_upper=2 * torch.ones(29), hidden_dims=cfg.actor_hidden_dims)
        widths = tuple(layer.out_features for layer in actor.network
                       if isinstance(layer, nn.Linear))
        self.assertEqual(widths, cfg.actor_hidden_dims + (29,))
        self.assertEqual(sum(isinstance(layer, nn.LayerNorm) for layer in actor.network),
                         len(cfg.actor_hidden_dims))
        observation = torch.randn(2, 465, requires_grad=True)
        action = actor(observation, torch.empty(2, 0))
        self.assertTrue(bool(((action >= -1) & (action <= 2)).all()))
        action.square().sum().backward()
        self.assertGreater(float(observation.grad.abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(observation.grad).all()))
        grouped = optimizers.build_optimizers(actor, lr=2e-4)
        adam_ids = {id(p) for group in grouped[0].param_groups for p in group['params']}
        muon_ids = {id(p) for group in grouped[1].param_groups for p in group['params']}
        self.assertFalse(adam_ids & muon_ids)
        self.assertEqual(adam_ids | muon_ids, {id(p) for p in actor.parameters()})
        for layer in actor.network:
            if isinstance(layer, nn.LayerNorm):
                self.assertTrue(all(id(p) in adam_ids for p in layer.parameters()))

    def test_critic_configured_widths_and_zero_output(self):
        cfg = config.runner_defaults()
        critic = networks.GoalConditionedCritic(133, 472, 0, hidden_dims=cfg.critic_hidden_dims)
        widths = tuple(layer.out_features for layer in critic.network if isinstance(layer, nn.Linear))
        self.assertEqual(widths, cfg.critic_hidden_dims + (1,))
        value = critic(torch.randn(2, 133), torch.randn(2, 472), torch.empty(2, 0))
        torch.testing.assert_close(value, torch.zeros(2))
        (value - 1).square().mean().backward()
        self.assertGreater(float(critic.network[-1].weight.grad.abs().sum()), 0)

    def test_invalid_widths_are_rejected_by_network(self):
        for widths in ((), (0,), (-1,), (True,), (1.5,)):
            with self.assertRaises(ValueError):
                networks.build_mlp(3, widths, 2)


if __name__ == '__main__':
    unittest.main()
