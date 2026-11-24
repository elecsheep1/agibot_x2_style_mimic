"""只用真实采样更新运行统计量，前向及想象过程保持统计量不变。"""
import unittest
import torch
from test_supermimic_world_model import state, wm


@unittest.skipUnless(hasattr(wm.WorldModel, 'NORMALIZATION_MODE'), '等待 source 接入运行归一化')
class RunningNormalizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.spec = state.StateSpec(31, 29)
        limiter = wm.TargetLimiter(torch.zeros(29), .25, -torch.ones(29), torch.ones(29), .1)
        self.model = wm.WorldModel(self.spec, 4, limiter, (16, 16), dt=.02, decimation=10)
        self.states = torch.randn(8, self.spec.dim) * .1
        self.states[:, 3:7] = torch.tensor([0., 0., 0., 1.])
        self.actions = torch.randn(8, 29) * .2
        self.context = torch.randn(8, 4)

    def test_chunked_statistics_equal_all_samples(self):
        self.model.update_normalization(self.states[:3], self.actions[:3], self.context[:3])
        self.model.update_normalization(self.states[3:], self.actions[3:], self.context[3:])
        rep = self.model.representation.represent(self.states)
        _, targets = self.model.features(self.states, self.actions, self.context)
        torch.testing.assert_close(self.model.state_normalizer.mean, rep.mean(0))
        torch.testing.assert_close(self.model.action_normalizer.mean, targets.mean(0))
        torch.testing.assert_close(self.model.state_normalizer.std, rep.std(0, unbiased=False), atol=1e-5, rtol=1e-4)
        self.assertEqual(int(self.model.state_normalizer.count), 8)
        self.assertEqual(int(self.model.action_normalizer.count), 8)

    def test_action_statistics_use_executed_target(self):
        self.model.update_normalization(self.states, self.actions, self.context)
        _, target = self.model.features(self.states, self.actions, self.context)
        torch.testing.assert_close(self.model.action_normalizer.mean, target.mean(0))
        self.assertFalse(torch.allclose(self.model.action_normalizer.mean, self.actions.mean(0)))

    def test_forward_train_and_eval_never_change_statistics(self):
        self.model.update_normalization(self.states, self.actions, self.context)
        before = {k:v.clone() for k,v in self.model.state_dict().items()}
        for training in (True, False):
            self.model.train(training)
            action = self.actions.clone().requires_grad_()
            out = self.model(self.states, action, self.context)
            out[:, 2].sum().backward()
            self.assertTrue(torch.isfinite(action.grad).all())
        for k,v in self.model.state_dict().items():
            torch.testing.assert_close(v, before[k], atol=0, rtol=0)

    def test_fixed_context_scaling_and_normalizer_equivalence(self):
        self.model.update_normalization(self.states, self.actions, self.context)
        width = self.model.representation.rep_state_dim
        end = width + self.spec.num_actions
        context_mean = self.model.input_mean[end:].clone()
        self.model.eval()
        self.model.update_normalization(self.states, self.actions, self.context + 100)
        torch.testing.assert_close(self.model.input_mean[end:], context_mean)
        features, target = self.model.features(self.states, self.actions, self.context)
        normalized = (features - self.model.input_mean) / self.model.input_std
        torch.testing.assert_close(normalized[:, :width], self.model.state_normalizer(features[:, :width]))
        torch.testing.assert_close(normalized[:, width:end], self.model.action_normalizer(target))
        self.assertFalse(self.model.state_normalizer.training)

    def test_replay_cannot_silently_refit_initialized_statistics(self):
        batch = dict(states=self.states[:, None].expand(-1, 2, -1), actions=self.actions[:, None], context=self.context)
        self.model.fit_input_stats(batch)
        with self.assertRaises(RuntimeError):
            self.model.fit_input_stats(batch)

    def test_nonfinite_statistics_rejected_before_mutation(self):
        bad = self.states.clone(); bad[0, 2] = float('nan')
        with self.assertRaises(ValueError):
            self.model.update_normalization(bad, self.actions, self.context)
        self.assertEqual(int(self.model.state_normalizer.count), 0)
        self.assertFalse(bool(self.model.stats_ready))


if __name__ == '__main__':
    unittest.main()
