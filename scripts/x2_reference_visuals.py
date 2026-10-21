"""零物理步回放的显示层同步；只写当前内存 stage，不保存机器人资产。"""

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


class ReferenceVisualSync:
    def __init__(self, stage, robot_path, body_names):
        root = stage.GetPrimAtPath(robot_path)
        if not root.IsValid():
            raise ValueError(f"回放机器人 Prim 不存在：{robot_path}")
        bodies = {}
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                name = prim.GetName()
                if name in bodies:
                    raise ValueError(f"刚体 Prim 名称重复：{name}")
                bodies[name] = prim
        if set(bodies) != set(body_names):
            raise ValueError("显示层刚体和 PhysX 刚体集合不一致")
        self.operations = []
        for name in body_names:
            xform = UsdGeom.Xformable(bodies[name])
            xform.ClearXformOpOrder()
            translate = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "reference")
            orient = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "reference")
            # 每个嵌套刚体用世界位姿，防止父节点转动被重复叠加。
            xform.SetResetXformStack(True)
            self.operations.append((translate, orient))

    def update(self, body_poses):
        if body_poses.shape != (len(self.operations), 7) or not np.isfinite(body_poses).all():
            raise ValueError("显示层刚体位姿不合法")
        for (translate, orient), pose in zip(self.operations, body_poses):
            translate.Set(Gf.Vec3d(*pose[:3].astype(float)))
            orient.Set(Gf.Quatd(float(pose[6]), Gf.Vec3d(*pose[3:6].astype(float))))
