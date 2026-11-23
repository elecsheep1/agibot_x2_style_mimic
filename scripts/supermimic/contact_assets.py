"""从实际 USD 读取碰撞凸包，组装可微离地几何。"""

import hashlib
import json

import numpy as np
import torch

def load_collision_vertices(usd_path, body_names):
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull, QhullError
    body_names = tuple(body_names)
    if not body_names or len(set(body_names)) != len(body_names):
        raise ValueError('接触身体名称必须非空且不重复')
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None or UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or str(UsdGeom.GetStageUpAxis(stage)) != 'Z':
        raise ValueError('接触几何需要米制、Z 向上的有效 USD')
    cache = UsdGeom.XformCache()
    groups = {name: [] for name in body_names}
    body_paths = {}
    records = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        body = prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body or body.GetName() not in groups:
            continue
        name = body.GetName()
        path = str(body.GetPath())
        if name in body_paths and body_paths[name] != path:
            raise ValueError(f'USD 存在同名接触刚体，无法唯一定位：{name}')
        body_paths[name] = path
        if not prim.IsA(UsdGeom.Mesh) or str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()) != 'convexHull':
            raise ValueError(f'接触部位存在未支持的碰撞形状：{prim.GetPath()}')
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
            raise ValueError(f'碰撞网格顶点无效：{prim.GetPath()}')
        transform = np.asarray(cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse())
        points = points @ transform[:3, :3] + transform[3, :3]
        try:
            vertices = points[ConvexHull(points).vertices]
        except QhullError as exc:
            raise ValueError(f'碰撞网格无法构造三维凸包：{prim.GetPath()}') from exc
        groups[name].append(torch.tensor(vertices, dtype=torch.float32))
        records.append(dict(body=name, path=str(prim.GetPath()), source_vertices=len(points), hull_vertices=len(vertices)))
    missing = [name for name, values in groups.items() if not values]
    if missing:
        raise ValueError(f'USD 缺少有效碰撞凸包：{missing}')
    return {name: torch.cat(groups[name]) for name in body_names}, records


def build_contact_geometry(spec, urdf_path, usd_path, joint_names):
    from x2_style_mimic.algorithms.supermimic.kinematics import DifferentiableKinematics
    from x2_style_mimic.algorithms.supermimic.contact_geometry import CONTACT_BODY_NAMES, GroundContactGeometry
    vertices, _ = load_collision_vertices(usd_path, CONTACT_BODY_NAMES)
    kinematics = DifferentiableKinematics(urdf_path, joint_names, CONTACT_BODY_NAMES)
    kinematics.align_joint_origins_from_usd(usd_path)
    return GroundContactGeometry(spec, kinematics, vertices)


def contact_geometry_contract(module):
    """防止恢复时以旧检查点覆盖当前资产推导出的几何。"""
    names = dict(body_names=list(module.kinematics.body_names),
                 joint_names=list(module.kinematics.joint_names))
    digest = hashlib.sha256(json.dumps(names, sort_keys=True).encode())
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return dict(version=1, **names, sha256=digest.hexdigest())


def validate_geometry_checkpoint(checkpoint, contract):
    metadata = checkpoint['metadata']
    if metadata.get('transition_schema_version') != 4 or metadata.get('contact_mode') != 'geometry_proxy':
        raise ValueError('旧检查点使用辅助力或接触预测结构，不能直接恢复几何接触版本')
    if metadata.get('contact_geometry_contract') != contract:
        raise ValueError('接触几何或运动学与检查点不一致，不能直接恢复')
