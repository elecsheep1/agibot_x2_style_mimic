"""离线验证 USD 凸包坐标、身体选择及几何检查点隔离。"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

path = Path(__file__).resolve().parents[1] / 'supermimic/contact_assets.py'
spec = importlib.util.spec_from_file_location('_geometry_assets_tests', path)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)

try:
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
except ImportError:
    Usd = None


@unittest.skipIf(Usd is None, '当前 Python 没有 USD；需要 Isaac Lab Python')
class CollisionAssetTests(unittest.TestCase):
    def make_stage(self, folder, sphere=False, finite=True):
        path = Path(folder) / 'robot.usda'
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        self.points = np.array([(x, y, z) for x in (-1., 1.)
                                for y in (-1., 1.) for z in (-1., 1.)])
        for name, offset in (('left', (1., 2., 3.)), ('right', (-1., 0., 2.))):
            body = UsdGeom.Xform.Define(stage, '/robot/' + name)
            body.AddTranslateOp().Set(Gf.Vec3d(10., 20., 30.))
            body.AddRotateZOp().Set(90.)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            if sphere:
                prim = UsdGeom.Sphere.Define(stage, body.GetPath().AppendChild('collision')).GetPrim()
            else:
                mesh = UsdGeom.Mesh.Define(stage, body.GetPath().AppendChild('collision'))
                points = self.points.copy()
                if not finite:
                    points[0, 0] = float('nan')
                mesh.GetPointsAttr().Set(points.tolist())
                mesh.AddTranslateOp().Set(Gf.Vec3d(*offset))
                prim = mesh.GetPrim()
                UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr().Set('convexHull')
            UsdPhysics.CollisionAPI.Apply(prim).GetCollisionEnabledAttr().Set(True)
        stage.GetRootLayer().Save()
        return path

    def test_relative_body_coordinates_and_requested_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_stage(folder)
            vertices, records = assets.load_collision_vertices(path, ('right', 'left'))
        self.assertEqual(tuple(vertices), ('right', 'left'))
        self.assertEqual(len(records), 2)
        torch.testing.assert_close(vertices['left'].amin(0), torch.tensor([0., 1., 2.]))
        torch.testing.assert_close(vertices['left'].amax(0), torch.tensor([2., 3., 4.]))
        torch.testing.assert_close(vertices['right'].amin(0), torch.tensor([-2., -1., 1.]))

    def test_missing_duplicate_and_unsupported_shapes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_stage(folder)
            for names in ((), ('left', 'left'), ('missing',)):
                with self.assertRaises(ValueError):
                    assets.load_collision_vertices(path, names)
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_stage(folder, sphere=True)
            with self.assertRaisesRegex(ValueError, '未支持'):
                assets.load_collision_vertices(path, ('left',))

    def test_invalid_points_and_units_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_stage(folder, finite=False)
            with self.assertRaisesRegex(ValueError, '顶点无效'):
                assets.load_collision_vertices(path, ('left',))
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_stage(folder)
            stage = Usd.Stage.Open(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, .01)
            stage.GetRootLayer().Save()
            with self.assertRaisesRegex(ValueError, '米制'):
                assets.load_collision_vertices(path, ('left',))


class GeometryContractTests(unittest.TestCase):
    def module(self):
        module = nn.Module()
        module.kinematics = nn.Module()
        module.kinematics.body_names = ('left', 'right')
        module.kinematics.joint_names = ('joint',)
        module.register_buffer('vertices', torch.zeros(4, 3))
        return module

    def test_contract_covers_vertices_and_body_order(self):
        module = self.module()
        contract = assets.contact_geometry_contract(module)
        self.assertEqual(contract, assets.contact_geometry_contract(module))
        module.vertices[0, 0] = 1.
        self.assertNotEqual(contract, assets.contact_geometry_contract(module))
        module.vertices.zero_()
        module.kinematics.body_names = ('right', 'left')
        self.assertNotEqual(contract, assets.contact_geometry_contract(module))

    def test_old_or_different_geometry_checkpoint_rejected(self):
        contract = assets.contact_geometry_contract(self.module())
        metadata = dict(transition_schema_version=4, contact_mode='geometry_proxy',
                        contact_geometry_contract=contract)
        assets.validate_geometry_checkpoint(dict(metadata=metadata), contract)
        for bad in ({}, dict(metadata, transition_schema_version=3),
                    dict(metadata, contact_mode='force_prediction'),
                    dict(metadata, contact_geometry_contract={})):
            with self.assertRaises(ValueError):
                assets.validate_geometry_checkpoint(dict(metadata=bad), contract)


class GroundGeometryTests(unittest.TestCase):
    def test_six_clearances_and_gradient_follow_root_height(self):
        from test_supermimic_world_model import load, state
        geometry = load('contact_geometry')

        class Kinematics(nn.Module):
            body_names = geometry.CONTACT_BODY_NAMES
            joint_names = ('joint',)

            def forward(self, root_pose, root_velocity, joint_position, joint_velocity):
                return (root_pose[:, None].expand(-1, 6, -1),
                        root_velocity[:, None].expand(-1, 6, -1))

        spec = state.StateSpec(1, 1)
        vertices = torch.tensor([(x, y, z) for x in (-.01, .01)
                                 for y in (-.01, .01) for z in (-.01, .01)])
        module = geometry.GroundContactGeometry(spec, Kinematics(),
            {name: vertices for name in geometry.CONTACT_BODY_NAMES})
        samples = torch.zeros(3, spec.dim)
        samples[:, 2] = torch.tensor([.1, .2, .3])
        samples[:, 6] = 1.
        samples.requires_grad_()
        clearance = module(samples)
        torch.testing.assert_close(clearance, (samples[:, 2:3] - .01).expand(-1, 6))
        clearance.sum().backward()
        self.assertTrue(bool(torch.isfinite(samples.grad).all()))
        torch.testing.assert_close(samples.grad[:, 2], torch.full((3,), 6.))


if __name__ == '__main__':
    unittest.main()
