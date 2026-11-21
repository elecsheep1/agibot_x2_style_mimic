"""世界模型批次合并与嵌套监督的回归检查。"""
import importlib.util
from pathlib import Path
import unittest
import torch

PATH = Path(__file__).resolve().parents[2] / 'source/x2_style_mimic/x2_style_mimic/algorithms/supermimic/batches.py'
spec = importlib.util.spec_from_file_location('_world_batches_test', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WorldBatchTests(unittest.TestCase):
    fields = {('task_memory', 'hold_steps'): ((), torch.long),
              ('assist_wrench_w',): ((6,), torch.float32)}

    def batch(self, marker):
        return dict(states=torch.full((2, 4, 3), float(marker)),
                    actions=torch.full((2, 3, 1), float(marker)),
                    context=torch.full((2, 2), float(marker)),
                    task_memory={'hold_steps': torch.full((2, 3), marker, dtype=torch.long)},
                    assist_wrench_w=torch.full((2, 3, 6), float(marker)))

    def merge(self, batches):
        return module.merge_world_batches(batches, 4, 3, 3, 1, 2, self.fields, 'cpu')

    def test_nested_fields_follow_same_sample_permutation(self):
        output = self.merge([self.batch(0), self.batch(1)])
        marker = output['states'][:, 0, 0]
        self.assertEqual(int(marker.sum()), 2)
        torch.testing.assert_close(marker, output['actions'][:, 0, 0])
        torch.testing.assert_close(marker, output['context'][:, 0])
        torch.testing.assert_close(marker.long(), output['task_memory']['hold_steps'][:, 0])
        torch.testing.assert_close(marker, output['assist_wrench_w'][:, 0, 2])

    def test_old_or_invalid_auxiliary_data_is_rejected(self):
        for failure in ('missing', 'shape', 'dtype', 'nan'):
            old = self.batch(1)
            if failure == 'missing':
                del old['task_memory']
            elif failure == 'shape':
                old['assist_wrench_w'] = torch.zeros(2, 2, 6)
            elif failure == 'dtype':
                old['task_memory']['hold_steps'] = torch.zeros(2, 3)
            else:
                old['assist_wrench_w'][0, 0, 0] = float('nan')
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.merge([self.batch(0), old])

    def test_device_preparation_keeps_nested_fields_and_detaches(self):
        batch = self.batch(1)
        batch['assist_wrench_w'].requires_grad_()
        output = module.detach_world_batch(batch, 'cpu')
        self.assertFalse(output['assist_wrench_w'].requires_grad)
        torch.testing.assert_close(output['task_memory']['hold_steps'], batch['task_memory']['hold_steps'])


if __name__ == '__main__':
    unittest.main()
