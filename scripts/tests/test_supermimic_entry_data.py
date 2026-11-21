"""正式训练脚本的数据辅助接口，不再依赖 checks 下的实现。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

DIRECTORY = Path(__file__).resolve().parents[1] / 'supermimic'
spec = importlib.util.spec_from_file_location('getup_entry_data_tests', DIRECTORY / 'getup_data.py')
data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_module)


class EntryDataTests(unittest.TestCase):
    def make_data(self):
        states = torch.tensor([0., 1., 100., 101.]).reshape(4, 1, 1).expand(-1, -1, 3).clone()
        return dict(state=states, next_state=states + 1, action=torch.zeros(4, 1, 2),
                    contexts=torch.zeros(1, 2), terminated=torch.tensor([[False], [True], [False], [True]]),
                    truncated=torch.zeros(4, 1, dtype=torch.bool), episode_ids=torch.tensor([[0], [0], [1], [1]]),
                    step_ids=torch.tensor([[0], [1], [0], [1]]), limiter={'scale': torch.ones(2)}, metadata={'test': 1})

    def test_windows_include_terminal_successor_without_crossing_episode(self):
        windows = data_module.valid_windows(self.make_data(), 2)
        self.assertEqual(windows.tolist(), [[0, 0], [2, 0]])

    def test_shared_vector_rejects_different_environment_calibration(self):
        torch.testing.assert_close(data_module.shared_vector(.5, 2, 'scale', 'cpu'), torch.full((2,), .5))
        with self.assertRaises(RuntimeError):
            data_module.shared_vector(torch.tensor([[1., 1.], [1., 2.]]), 2, 'scale', 'cpu')

    def test_offline_replay_samples_training_episodes_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            data_path, report_path = path / 'data.pt', path / 'report.json'
            torch.save(self.make_data(), data_path)
            report_path.write_text(json.dumps(dict(data=str(data_path), train_episodes=[[0, 0]], validation_episodes=[[0, 1]])))
            replay = data_module.OfflineWorldReplay(data_path, report_path, {'test': 1}, {'scale': torch.ones(2)}, 'cpu')
            batch = replay.sample_windows(20, 2)
            self.assertEqual(batch['states'].shape, (20, 3, 3))
            self.assertEqual(batch['actions'].shape, (20, 2, 2))
            self.assertTrue(bool((batch['states'][:, 0, 0] == 0).all()))
            self.assertTrue(bool((batch['states'][:, -1, 0] == 2).all()))


if __name__ == '__main__':
    unittest.main()
