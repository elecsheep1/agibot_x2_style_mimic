"""注释编辑不应把完成的训练标为失败，代码逻辑变化仍应拒绝。"""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

folder = Path(__file__).resolve().parents[1] / 'supermimic'
sys.path.insert(0, str(folder))
spec = importlib.util.spec_from_file_location('_source_guard_test', folder / 'getup_training.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class SourceGuardTests(unittest.TestCase):
    def run_finish(self, initial, final):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root / 'source').mkdir()
            p = root / 'source/config.py'; p.write_text(initial)
            run = training.GetupTrainingRun.__new__(training.GetupTrainingRun)
            run.source_before = training.file_hashes([p])
            run.source_semantic_before = training.semantic_file_hashes([p])
            run.cfg = SimpleNamespace(iterations=200, output=root)
            run.runtime = SimpleNamespace(runner=SimpleNamespace(total_env_steps=10),
                                          env=SimpleNamespace(num_envs=2), replay=SimpleNamespace(total_steps=10))
            reports = {}
            run.logger = SimpleNamespace(write_json=lambda name, value: reports.update({name: value}), flush=lambda: None)
            p.write_text(final)
            with patch.object(training, 'PROJECT', root):
                run._finish()
            return reports

    def test_comments_and_format_changes_complete_without_claiming_byte_identity(self):
        reports = self.run_finish('weight = -0.005\n', '# weight = -0.010\nweight=-0.005\n')
        self.assertFalse(reports['summary.json']['formal_source_unchanged'])
        self.assertTrue(reports['summary.json']['formal_source_semantics_unchanged'])
        self.assertEqual(len(reports['source_changes.json']['files']), 1)

    def test_logic_change_still_rejected(self):
        with self.assertRaisesRegex(RuntimeError, '代码逻辑发生变化'):
            self.run_finish('weight = -0.005\n', 'weight = -0.010\n')

    def test_unchanged_source_keeps_original_claim(self):
        reports = self.run_finish('weight = -0.005\n', 'weight = -0.005\n')
        self.assertTrue(reports['summary.json']['formal_source_unchanged'])
        self.assertTrue(reports['summary.json']['formal_source_semantics_unchanged'])


if __name__ == '__main__':
    unittest.main()
