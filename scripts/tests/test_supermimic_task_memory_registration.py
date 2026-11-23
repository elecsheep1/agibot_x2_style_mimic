"""任务记忆按实际张量注册，新增配对状态不能丢失或改变布尔类型。"""
import ast
from pathlib import Path
import unittest
import torch

source = Path(__file__).resolve().parents[1] / 'supermimic/getup_runtime.py'
fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'extend_task_memory_fields')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
register = namespace['extend_task_memory_fields']



class TaskMemoryRegistrationTests(unittest.TestCase):

    def test_pair_memory_registered_for_both_times(self):
        fields = {('task_memory', 'assist_withdrawn'): ((), torch.bool)}
        memory = dict(assist_withdrawn=torch.zeros(3, dtype=torch.bool),
                      self_contact_strong_duration=torch.zeros(3, 17),
                      self_contact_strong_active=torch.zeros(3, 17, dtype=torch.bool))
        result = register(fields, memory, 3)
        for group in ('task_memory', 'next_task_memory'):
            self.assertEqual(result[(group, 'self_contact_strong_duration')], ((17,), torch.float32))
            self.assertEqual(result[(group, 'self_contact_strong_active')], ((17,), torch.bool))
        self.assertEqual(len(fields), 1)

    def test_wrong_environment_dimension_rejected(self):
        with self.assertRaises(ValueError):
            register({}, dict(contact=torch.zeros(2, 17)), 3)

    def test_conflicting_old_schema_rejected(self):
        with self.assertRaises(ValueError):
            register({('task_memory', 'contact'): ((), torch.float32)}, dict(contact=torch.zeros(3, 17)), 3)


if __name__ == '__main__':
    unittest.main()
