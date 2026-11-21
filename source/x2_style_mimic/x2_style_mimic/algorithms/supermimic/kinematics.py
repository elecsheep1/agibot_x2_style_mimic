"""根据 URDF 计算可微刚体姿态和刚体原点速度。

四元数采用 XYZW。
根线速度、根角速度和输出刚体速度均采用世界坐标。
关节状态顺序由 joint_names 指定。
"""


import math
import xml.etree.ElementTree as ET

import torch
from torch import nn

from ...utils.rotation import (
    normalize_quaternion,
    quaternion_multiply,
    rotate_vector,
    rotation_vector_to_quaternion,
)

def _read_vector(element, attribute, default):
    """读取 URDF 中的三维向量。"""
    text = (
        default
        if element is None
        else element.get(attribute, default)
    )

    values = [
        float(value)
        for value in text.split()
    ]

    if (
        len(values) != 3
        or not all(math.isfinite(value) for value in values)
    ):
        raise ValueError(f"URDF {attribute} 三维向量无效: {text}")

    return values


class DifferentiableKinematics(nn.Module):
    """可微刚体运动学计算器。"""

    def __init__(
        self,
        urdf_path,
        joint_names,
        body_names,
    ):
        super().__init__()

        self.joint_names = tuple(joint_names)
        self.body_names = tuple(body_names)

        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names 存在重复名称")

        robot = ET.parse(urdf_path).getroot()

        links = {
            link.get("name")
            for link in robot.findall("link")
        }
        joints = robot.findall("joint")

        children = {
            joint.find("child").get("link")
            for joint in joints
        }

        roots = links - children

        if len(roots) != 1:
            raise ValueError("URDF 必须包含一个根刚体")

        self.root_name = roots.pop()
        if self.root_name != "base_link":
            raise ValueError(
                "当前 X2 世界状态使用 base_link 作为根刚体，"
                f"但 URDF 根刚体是 {self.root_name}"
            )

        moving_joint_names = {
            joint.get("name")
            for joint in joints
            if joint.get("type") != "fixed"
        }
        supplied_joint_names = set(self.joint_names)
        if moving_joint_names != supplied_joint_names:
            missing = (
                moving_joint_names - supplied_joint_names
            )
            extra = (
                supplied_joint_names - moving_joint_names
            )

            raise ValueError(f"关节名称不一致，缺失={missing}，多余={extra}")

        unknown_bodies = set(self.body_names) - links
        if unknown_bodies:
            raise ValueError(f"URDF 中找不到刚体: {unknown_bodies}")

        # 根刚体占据第零个位置。
        visited = {
            self.root_name: 0,
        }

        self.parents = []
        self.joint_indices = []

        origin_positions = []
        origin_quaternions = []
        joint_axes = []

        pending = list(joints)

        # 按父刚体先于子刚体的顺序整理关节树。
        """
        pending：这一轮要检查的关节列表。
        remaining：这一轮暂时处理不了、留到下一轮的关节列表。
        """
        while pending:
            remaining = []

            for joint in pending:
                parent_name = joint.find("parent").get("link")
                child_name = joint.find("child").get("link")
                # 如果父刚体还没被访问过，就先跳过这个关节，留到下一轮。
                if parent_name not in visited:
                    remaining.append(joint)
                    continue

                joint_type = joint.get("type")
                if (
                    joint_type
                    not in ("fixed", "revolute", "continuous")
                    or joint.find("mimic") is not None
                ):
                    raise ValueError(f"不支持的关节: {joint.get('name')}")
                if child_name in visited:
                    raise ValueError("URDF 关节关系不是树结构")

                origin = joint.find("origin")
                position = _read_vector(
                    origin,
                    "xyz",
                    "0 0 0",
                )
                roll, pitch, yaw = _read_vector(
                    origin,
                    "rpy",
                    "0 0 0",
                )
                # URDF 固定轴 RPY 对应 Rz(yaw) Ry(pitch) Rx(roll)。
                # 欧拉角对应的三个单轴旋转分别转换为四元数
                quaternion_x = torch.tensor(
                    [
                        math.sin(roll / 2.0),
                        0.0,
                        0.0,
                        math.cos(roll / 2.0),
                    ],
                    dtype=torch.float64,
                )
                quaternion_y = torch.tensor(
                    [
                        0.0,
                        math.sin(pitch / 2.0),
                        0.0,
                        math.cos(pitch / 2.0),
                    ],
                    dtype=torch.float64,
                )
                quaternion_z = torch.tensor(
                    [
                        0.0,
                        0.0,
                        math.sin(yaw / 2.0),
                        math.cos(yaw / 2.0),
                    ],
                    dtype=torch.float64,
                )

                # URDF 固定轴 RPY 对应 Rz(yaw) Ry(pitch) Rx(roll)，所以四元数乘法顺序为 ZYX。
                origin_quaternion = quaternion_multiply(
                    quaternion_z,
                    quaternion_multiply(
                        quaternion_y,
                        quaternion_x,
                    ),
                )

                joint_index = -1
                axis = torch.zeros(3, dtype=torch.float64)

                if joint_type != "fixed":
                    axis = torch.tensor(
                        _read_vector(
                            joint.find("axis"),
                            "xyz",
                            "1 0 0",
                        ),
                        dtype=torch.float64,
                    )

                    if axis.norm() < 1e-12:
                        raise ValueError(f"关节轴为零: {joint.get('name')}")

                    axis /= axis.norm()

                    joint_index = self.joint_names.index(
                        joint.get("name")
                    )

                self.parents.append(
                    visited[parent_name]
                )
                self.joint_indices.append(
                    joint_index
                )

                origin_positions.append(position)
                origin_quaternions.append(origin_quaternion)
                joint_axes.append(axis)
                visited[child_name] = len(visited)

            if len(remaining) == len(pending):
                raise ValueError("URDF 存在断开的关节或循环关系")

            pending = remaining

        if set(visited) != links:
            raise ValueError("URDF 存在未连接的刚体")

        # 各关节原点相对于父 link 的位置偏移
        self.register_buffer(
            "origin_positions",
            torch.tensor(
                origin_positions,
                dtype=torch.float64,
            ),
        )
        # 各关节原点相对于父 link 的旋转姿态
        self.register_buffer(
            "origin_quaternions",
            torch.stack(origin_quaternions),
        )
        # 各关节在关节坐标系中的旋转轴；这里固定关节存零向量
        self.register_buffer(
            "joint_axes",
            torch.stack(joint_axes),
        )
        # 所需刚体在遍历编号中的索引
        self.register_buffer(
            "body_indices",
            torch.tensor(
                [
                    visited[name]
                    for name in self.body_names
                ],
                dtype=torch.long,
            ),
        )

    @torch.no_grad()
    def align_joint_origins_from_usd(self, usd_path):
        """使用实际 USD 中的活动关节变换对齐运动学。"""
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(usd_path))

        if stage is None:
            raise ValueError(f"无法打开 USD: {usd_path}")

        usd_joints = {}

        for prim in stage.TraverseAll():
            if prim.IsA(UsdPhysics.RevoluteJoint):
                name = prim.GetName()

                if name in usd_joints:
                    raise ValueError(f"USD 存在重复关节名称: {name}")
                usd_joints[name] = (
                    UsdPhysics.RevoluteJoint(prim)
                )

        positions = self.origin_positions.clone()
        quaternions = self.origin_quaternions.clone()
        axes = self.joint_axes.clone()

        axis_vectors = {
            "X": (1.0, 0.0, 0.0),
            "Y": (0.0, 1.0, 0.0),
            "Z": (0.0, 0.0, 1.0),
        }

        for edge_index, joint_index in enumerate(
            self.joint_indices
        ):
            if joint_index < 0:
                continue

            name = self.joint_names[joint_index]

            if name not in usd_joints:
                raise ValueError(f"USD 中找不到活动关节: {name}")

            joint = usd_joints[name]

            child_position = tuple(
                joint.GetLocalPos1Attr().Get()
            )
            child_rotation = (
                joint.GetLocalRot1Attr().Get()
            )

            if max(
                abs(value)
                for value in child_position
            ) > 1e-8:
                raise ValueError(
                    f"{name} 的子侧关节原点存在偏移，"
                    "当前运动学递推需要扩展"
                )

            child_rotation_xyz = tuple(
                child_rotation.GetImaginary()
            )

            if (
                abs(
                    abs(child_rotation.GetReal()) - 1.0
                ) > 1e-8
                or max(
                    abs(value)
                    for value in child_rotation_xyz
                ) > 1e-8
            ):
                raise ValueError(
                    f"{name} 的子侧关节坐标系存在旋转，"
                    "当前运动学递推需要扩展"
                )

            parent_position = tuple(
                joint.GetLocalPos0Attr().Get()
            )
            parent_rotation = (
                joint.GetLocalRot0Attr().Get()
            )

            rotation_xyzw = (
                *parent_rotation.GetImaginary(),
                parent_rotation.GetReal(),
            )

            axis_name = str(
                joint.GetAxisAttr().Get()
            )

            if axis_name not in axis_vectors:
                raise ValueError(f"{name} 的 USD 关节轴无效: {axis_name}")

            positions[edge_index] = (
                positions.new_tensor(parent_position)
            )
            quaternions[edge_index] = (
                normalize_quaternion(
                    quaternions.new_tensor(
                        rotation_xyzw
                    )
                )
            )
            axes[edge_index] = (
                axes.new_tensor(
                    axis_vectors[axis_name]
                )
            )

        # 全部校验完成后再更新模块，避免部分更新。
        self.origin_positions.copy_(positions)
        self.origin_quaternions.copy_(quaternions)
        self.joint_axes.copy_(axes)



    def forward(
        self,
        root_pose,
        root_velocity,
        joint_position,
        joint_velocity,
    ):
        batch_size = root_pose.shape[0]

        if (
            root_pose.shape != (batch_size, 7)
            or root_velocity.shape != (batch_size, 6)
        ):
            raise ValueError("根姿态和根速度必须为 [B,7]、[B,6]")

        expected_joint_shape = (
            batch_size,
            len(self.joint_names),
        )

        if (
            joint_position.shape != expected_joint_shape
            or joint_velocity.shape != expected_joint_shape
        ):
            raise ValueError("关节状态形状必须与 joint_names 对应")

        positions = [
            root_pose[:, :3]
        ]
        rotations = [
            normalize_quaternion(
                root_pose[:, 3:7]
            )
        ]
        linear_velocities = [
            root_velocity[:, :3],
        ]
        angular_velocities = [
            root_velocity[:, 3:6],
        ]
        # 把张量转换成与 root_pose 相同的设备和数据类型，方便后续一起计算。
        offsets = self.origin_positions.to(root_pose)
        origin_quaternions = (
            self.origin_quaternions.to(root_pose)
        )
        axes = self.joint_axes.to(root_pose)

        edges = zip(
            self.parents,
            self.joint_indices,
        )
        # 已知父刚体的状态和关节角度、速度，计算子刚体的世界位置、姿态和速度。
        for edge_index, (parent, joint_index) in enumerate(edges):
            # 计算子刚体原点相对于父刚体的世界坐标偏移
            offset_world = rotate_vector(
                rotations[parent],
                offsets[edge_index],
            )
            position = (
                positions[parent] + offset_world
            )
            # 计算子刚体的世界坐标姿态
            rotation = quaternion_multiply(
                rotations[parent],
                origin_quaternions[edge_index],
            )
            # v_child = v_parent + ω_parent × r
            linear_velocity = (
                linear_velocities[parent]
                + torch.cross(
                    angular_velocities[parent],
                    offset_world,
                    dim=-1,
                )
            )
            # 先继承父刚体的角速度，如果是固定关节，子刚体与父刚体保持固定连接，角速度就相同。
            angular_velocity = angular_velocities[parent]

            # 如果是活动关节，计算关节对子刚体的影响
            if joint_index >= 0:
                # 先把关节旋转轴转换到世界坐标系
                axis_world = rotate_vector(
                    rotation,
                    axes[edge_index],
                )
                # 再加入关节转动产生的角速度
                # ω_child = ω_parent + 世界旋转轴 × 关节角速度
                angular_velocity = (
                    angular_velocity
                    + axis_world
                    * joint_velocity[
                        :, joint_index:joint_index + 1
                    ]
                )
                # 计算关节旋转产生的姿态变化
                # 旋转轴乘旋转角度 [0, 0, 1] × 0.5 = [0, 0, 0.5]
                rotation_vector = (
                    axes[edge_index]
                    * joint_position[
                        :, joint_index:joint_index + 1
                    ]
                )
                # 旋转向量转换为四元数增量
                increment = rotation_vector_to_quaternion(
                    rotation_vector
                )
                # 计算子刚体的世界坐标姿态
                rotation = normalize_quaternion(
                    quaternion_multiply(
                        rotation,
                        increment,
                    )
                )

            positions.append(position)
            rotations.append(rotation)
            linear_velocities.append(linear_velocity)
            angular_velocities.append(angular_velocity)

        poses = torch.cat(
            (
                torch.stack(positions, dim=1),
                torch.stack(rotations, dim=1),
            ),
            dim=-1,
        )

        velocities = torch.cat(
            (
                torch.stack(linear_velocities, dim=1),
                torch.stack(angular_velocities, dim=1),
            ),
            dim=-1,
        )

        return (
            poses.index_select(
                dim=1,
                index=self.body_indices,
            ),
            velocities.index_select(
                dim=1,
                index=self.body_indices,
            ),
        )


