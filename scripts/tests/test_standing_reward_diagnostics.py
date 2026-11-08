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
    def check_reward(self, with_limbs=False, yaw=0.0):
        source = ROOT / "source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic/mdp/getup.py"
        tree = ast.parse(source.read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                    node.name in {"smoothstep", "standing_quality", "standing_quality_reward", "standing_limb_quality"}]
        production = ModuleType(NAME)
        production.__dict__.update(torch=torch, math=math, SceneEntityCfg=object, ManagerBasedRLEnv=object)
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
        if with_limbs:
            c, s = math.cos(yaw), math.sin(yaw)
            poses = []
            for side in (1, -1):
                angle = yaw + side * math.radians(5)
                poses.append([-s * side * .17, c * side * .17, .08,
                              0, 0, math.sin(angle / 2), math.cos(angle / 2)])
            data.body_link_pose_w = tensor([
                [[0, 0, .63, 0, 0, 0, 1]] + poses
            ] * 2)
            data.root_link_pose_w = tensor(
                [[0, 0, .63, 0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]] * 2)
            data.joint_pos = tensor([[0.] * 14, [.8] * 14])
        scene = {"robot": SimpleNamespace(data=data)}
        class Scene(dict):
            env_origins = torch.zeros(2, 3)
        env = SimpleNamespace(scene=Scene(scene))
        params = dict(torso_cfg=SimpleNamespace(body_ids=[0]),
                      ground_sensor_cfgs=(SimpleNamespace(force=200.),) * 2,
                      wrist_sensor_cfgs=(SimpleNamespace(force=0.),) * 2)
        if with_limbs:
            params.update(limb_joint_cfg=SimpleNamespace(joint_ids=list(range(14))),
                          feet_cfg=SimpleNamespace(body_ids=[1, 2]),
                          limb_joint_targets=(0.,) * 14, limb_fraction=.10)
        path = ROOT / "scripts/rsl_rl/multimotion/diagnostics.py"
        spec = importlib.util.spec_from_file_location("standing_diagnostics", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(sys.modules, {NAME: production}):
            result = module.standing_reward_comparison(env, params)
        torch.testing.assert_close(result[:, 0], production.standing_quality_reward(env, **params))
        self.assertTrue(torch.all(result[:, 1] >= result[:, 0]))
        torch.testing.assert_close(result[:, 2], result[:, 1] - result[:, 0])

        if with_limbs:
            quality = production.standing_limb_quality(
                env, params["limb_joint_cfg"], params["feet_cfg"], params["limb_joint_targets"])
            torch.testing.assert_close(quality[0], torch.tensor(1.))
            self.assertLess(quality[1], quality[0])
            inputs = dict(height=torch.tensor([.3, .63]), root_up=torch.ones(2),
                          torso_up=torch.ones(2), linear_velocity=torch.zeros(2, 3),
                          angular_velocity=torch.zeros(2, 3), feet_support=torch.ones(2),
                          wrist_load=torch.zeros(2))
            baseline = production.standing_quality(**inputs)
            penalized = production.standing_quality(**inputs, limb_quality=torch.zeros(2))
            torch.testing.assert_close(baseline[0], penalized[0])
            torch.testing.assert_close(baseline[1] - penalized[1], torch.tensor(.065))

    def test_legacy_reward(self):
        self.check_reward()

    def test_limb_reward(self):
        self.check_reward(with_limbs=True)

    def test_limb_reward_rotated_heading(self):
        self.check_reward(with_limbs=True, yaw=2.1)


if __name__ == "__main__":
    unittest.main()
