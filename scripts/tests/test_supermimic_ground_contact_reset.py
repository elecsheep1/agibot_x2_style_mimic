"""执行正式重置函数与 AMP 父类，验证按环境清理及终止捕获顺序。

使用 AST 隔离仿真依赖；不验证 PhysX 或 Warp 的实际执行。
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2] / 'source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic'


def production_reset():
    tree = ast.parse((ROOT / 'mdp/self_contact.py').read_text())
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name == 'reset_ground_contact_sensors' or
             isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'GROUND_CONTACT_SENSOR_NAMES' for t in node.targets)]
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / 'mdp/self_contact.py'), 'exec'), namespace)
    return namespace['reset_ground_contact_sensors'], namespace['GROUND_CONTACT_SENSOR_NAMES']


class GroundContactResetTests(unittest.TestCase):
    def test_production_function_clears_only_selected_environments(self):
        reset, names = production_reset()
        calls = []
        sensors = {}
        buffers = {}
        for name in names:
            history = torch.ones(3, 3, 1, 1, 3)
            buffers[name] = history
            sensors[name] = SimpleNamespace(
                reset=lambda ids, name=name: calls.append((name, ids.clone())),
                data=SimpleNamespace(force_matrix_w_history=SimpleNamespace(torch=history)))
        env = SimpleNamespace(num_envs=3, device='cpu', scene=SimpleNamespace(sensors=sensors))
        reset(env, torch.tensor([1]))
        for history in buffers.values():
            torch.testing.assert_close(history[1], torch.zeros_like(history[1]))
            torch.testing.assert_close(history[[0, 2]], torch.ones_like(history[[0, 2]]))
        self.assertEqual(len(calls), 2 * len(names))
        calls.clear()
        reset(env, torch.empty(0, dtype=torch.long))
        self.assertFalse(calls)

    def test_amp_terminal_capture_precedes_parent_reset_and_cleanup(self):
        self.check_order(capture=True)

    def test_manual_reset_still_cleans_without_terminal_capture(self):
        self.check_order(capture=False)

    def check_order(self, capture):
        events = []
        class Base:
            def _reset_idx(self, ids):
                events.append('parent_reset')
                self.current[ids] = -10
        def cleanup(env, ids):
            events.append('cleanup')
            torch.testing.assert_close(env.current[ids], torch.tensor([[-10.]]))
            if capture:
                torch.testing.assert_close(env._amp_terminal_state[ids], torch.tensor([[2.]]))
        tree = ast.parse((ROOT / 'x2_getup_env.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'X2AMPGetUpEnv')
        namespace = dict(ManagerBasedRLEnv=Base, torch=torch, reset_ground_contact_sensors=cleanup,
                         amp_observation=lambda env, cfg: env.current.clone())
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ROOT / 'x2_getup_env.py'), 'exec'), namespace)
        env = namespace['X2AMPGetUpEnv']()
        env.current = torch.tensor([[1.], [2.]])
        env._amp_capture_active = capture
        env._amp_asset_cfg = None
        env._amp_terminal_state = torch.zeros(2, 1)
        env._amp_terminal_mask = torch.zeros(2, dtype=torch.bool)
        env._reset_idx(torch.tensor([1]))
        self.assertEqual(events, ['parent_reset', 'cleanup'])
        self.assertEqual(env._amp_terminal_mask.tolist(), [False, capture])
        torch.testing.assert_close(env.current[0], torch.tensor([1.]))


if __name__ == '__main__':
    unittest.main()
