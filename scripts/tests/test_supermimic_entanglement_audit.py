"""候选筛查不得把站立、接触换配对或跨回合计时当成自缠。"""
import importlib.util
from pathlib import Path
import unittest
import torch

path = Path(__file__).resolve().parents[1] / 'checks/audit_supermimic_entanglement.py'
spec = importlib.util.spec_from_file_location('entanglement_audit', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CandidateTests(unittest.TestCase):
    def data(self):
        s = torch.zeros(12, 1, 13)
        s[:, :, 2] = .1
        s[:, :, 3] = 2 ** -.5
        s[:, :, 6] = 2 ** -.5
        return dict(history=s, episode_keys=torch.zeros(12, 2, dtype=torch.long),
                    task_memory=dict(episode_steps=torch.arange(12),
                                     self_contact_strong_active=torch.ones(12, 2, dtype=torch.bool)))

    def run_audit(self, d):
        return module.find_candidates(d, dt=.1, window_seconds=.5, contact_seconds=.3,
            height_progress=.03, upright_progress=.1, standing_height=.55, standing_up=.85)

    def test_stalled_contact_detected_and_merged(self):
        result = self.run_audit(self.data())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['start_step'], 5)
        self.assertEqual(result[0]['end_step'], 11)

    def test_standing_excluded(self):
        d = self.data()
        d['history'][:, :, 2] = .65
        d['history'][:, :, 3] = 0
        d['history'][:, :, 6] = 1
        self.assertEqual(self.run_audit(d), [])

    def test_progress_excluded(self):
        d = self.data()
        d['history'][:, 0, 2] = torch.arange(12) * .02
        self.assertEqual(self.run_audit(d), [])

    def test_alternating_pairs_not_persistent(self):
        d = self.data()
        a = d['task_memory']['self_contact_strong_active']
        a.zero_()
        a[::2, 0] = True
        a[1::2, 1] = True
        self.assertEqual(self.run_audit(d), [])

    def test_episode_boundaries_and_gaps_not_joined(self):
        d = self.data()
        d['episode_keys'][:, 1] = torch.arange(12) // 3
        self.assertEqual(self.run_audit(d), [])
        d = self.data()
        d['task_memory']['episode_steps'] *= 2
        self.assertEqual(self.run_audit(d), [])


if __name__ == '__main__':
    unittest.main()
