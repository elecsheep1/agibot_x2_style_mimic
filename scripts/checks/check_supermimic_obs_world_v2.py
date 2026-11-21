"""检查当前正式实现；可选择在临时副本应用候选补丁，绝不改写 source。"""
import argparse
import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest

sys.dont_write_bytecode = True
import torch

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--candidate-patch', action='store_true')
options, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining
RELATIVE = Path('source/x2_style_mimic/x2_style_mimic/algorithms/supermimic')
TEMPORARY = tempfile.TemporaryDirectory(prefix='supermimic_obs_world_v2_')
DIRECTORY = Path(TEMPORARY.name) / RELATIVE
DIRECTORY.mkdir(parents=True)
for name in ('state.py', 'observations.py', 'world_model.py', 'runner.py'):
    shutil.copy2(ROOT / RELATIVE / name, DIRECTORY / name)
if options.candidate_patch:
    subprocess.run(['git', 'apply', str(ROOT / 'scripts/patches/supermimic_obs_world_v2.patch')], cwd=TEMPORARY.name, check=True)
for name, path in [('proposal', ROOT / RELATIVE.parent.parent), ('proposal.algorithms', DIRECTORY.parent),
                   ('proposal.algorithms.supermimic', DIRECTORY)]:
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
state = importlib.import_module('proposal.algorithms.supermimic.state')
observations = importlib.import_module('proposal.algorithms.supermimic.observations')
world = importlib.import_module('proposal.algorithms.supermimic.world_model')


class ProposalTests(unittest.TestCase):
    def test_running_statistics_match_all_samples(self):
        normalizer = observations.ObservationNormalizer(4)
        torch.manual_seed(9)
        values = torch.randn(31, 4)
        for part in (values[:7], values[7:19], values[19:]):
            normalizer.update(part)
        torch.testing.assert_close(normalizer.mean, values.mean(0))
        torch.testing.assert_close(normalizer.variance, values.var(0, unbiased=False))
        self.assertEqual(int(normalizer.count), 31)

    def test_read_only_normalization_retains_input_gradient(self):
        normalizer = observations.ObservationNormalizer(3)
        normalizer.update(torch.tensor([[-1., -1., -1.], [1., 1., 1.]]))
        saved = {k: v.clone() for k, v in normalizer.state_dict().items()}
        x = torch.tensor([[.2, .3, .4]], requires_grad=True)
        normalizer(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        for k, v in normalizer.state_dict().items():
            torch.testing.assert_close(v, saved[k])
        self.assertLessEqual(float(normalizer(torch.full((1, 3), 1e8)).abs().max()), 10.)

    def make_model(self):
        spec = state.StateSpec(3, 2)
        limiter = world.TargetLimiter(torch.zeros(2), torch.ones(2), -torch.ones(2), torch.ones(2), torch.full((2,), .1))
        model = world.WorldModel(spec, 4, limiter, (16, 16))
        model.stats_ready.fill_(True)
        x = torch.zeros(2, spec.dim)
        x[:, 6] = 1.
        return spec, model, x, torch.zeros(2, 4)

    def test_same_executed_target_same_physical_prediction(self):
        spec, model, x, context = self.make_model()
        a = torch.full((2, 2), 100., requires_grad=True)
        b = a.detach() * 2
        y1, y2 = model(x, a, context), model(x, b, context)
        physical = 13 + 2 * spec.num_joints
        torch.testing.assert_close(y1[:, :physical], y2[:, :physical], atol=0, rtol=0)
        grad = torch.autograd.grad(y1[:, :physical].sum(), a)[0]
        torch.testing.assert_close(grad, torch.zeros_like(grad))
        torch.testing.assert_close(state.unpack_state(y1, spec)['previous_action'], a)
        x2 = x.clone()
        x2[:, physical:physical + spec.num_actions] = 999.
        torch.testing.assert_close(model(x2, a, context)[:, :physical], y1[:, :physical], atol=0, rtol=0)

    def test_unsaturated_action_keeps_dynamics_gradient(self):
        _, model, x, context = self.make_model()
        a = torch.zeros(2, 2, requires_grad=True)
        gradient = torch.autograd.grad(model(x, a, context)[:, :19].sum(), a)[0]
        self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertEqual(model.input_mean.numel(), 15 + 2 * 3 + 2 + 4)

    def test_nonfinite_statistics_rejected_without_mutation(self):
        normalizer = observations.ObservationNormalizer(2)
        with self.assertRaises(ValueError):
            normalizer.update(torch.tensor([[float('nan'), 0.]]))
        self.assertEqual(int(normalizer.count), 0)


if __name__ == '__main__':
    try:
        unittest.main(verbosity=2)
    finally:
        TEMPORARY.cleanup()
