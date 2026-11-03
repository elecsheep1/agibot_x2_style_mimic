"""Verify replay extraction against current production reward without a simulator."""
import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
NAME = "x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.getup"


class StandingRewardDiagnosticsTests(unittest.TestCase):
    def test_current_reward_without_default_pose_multiplier(self):
        source = ROOT / "source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic/mdp/getup.py"
        tree = ast.parse(source.read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                    node.name in {"smoothstep", "standing_quality", "standing_quality_reward"}]
        production = ModuleType(NAME)
        production.__dict__.update(torch=torch, math=math, SceneEntityCfg=object)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), production.__dict__)
        production.ground_contact_force = lambda env, cfg: torch.full((2,), cfg.force)
        tensor = lambda value: SimpleNamespace(torch=torch.tensor(value, dtype=torch.float32))
        data = SimpleNamespace(
            body_link_pose_w=tensor([[[0, 0, .63, 0, 0, 0, 1]]] * 2),
            root_link_vel_w=tensor([[0] * 6] * 2),
            root_pos_w=tensor([[0, 0, .63]] * 2),
            projected_gravity_b=tensor([[.342, 0, -.94], [.1, 0, -.995]]),
            joint_pos=tensor([[1.] * 31] * 2), default_joint_pos=tensor([[0.] * 31] * 2),
        )
        scene = {"robot": SimpleNamespace(data=data)}
        class Scene(dict):
            env_origins = torch.zeros(2, 3)
        env = SimpleNamespace(scene=Scene(scene))
        params = dict(torso_cfg=SimpleNamespace(body_ids=[0]),
                      ground_sensor_cfgs=(SimpleNamespace(force=200.),) * 2,
                      wrist_sensor_cfgs=(SimpleNamespace(force=0.),) * 2)
        path = ROOT / "scripts/rsl_rl/multimotion/diagnostics.py"
        spec = importlib.util.spec_from_file_location("standing_diagnostics", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(sys.modules, {NAME: production}):
            result = module.standing_reward_comparison(env, params)
        torch.testing.assert_close(result[:, 0], production.standing_quality_reward(env, **params))
        self.assertTrue(torch.all(result[:, 1] >= result[:, 0]))
        torch.testing.assert_close(result[:, 2], result[:, 1] - result[:, 0])


if __name__ == "__main__":
    unittest.main()
