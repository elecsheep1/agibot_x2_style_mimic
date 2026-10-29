"""为 X2 嵌套刚体补充 PhysX 接触报告，修改本地 USD 入口层。"""

# Support direct execution: python scripts/<category>/<tool>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Enable contact reports on every X2 rigid body.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 必须在启动 Isaac Sim 后导入。
from pxr import PhysxSchema, Usd, UsdPhysics


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    usd_path = (
        project_root
        / "source"
        / "x2_style_mimic"
        / "x2_style_mimic"
        / "assets"
        / "robots"
        / "x2_ultra"
        / "x2_ultra"
        / "x2_ultra.usda"
    )

    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Cannot open USD: {usd_path}")

    # 将新增配置写到入口层，配套 payload 文件保持原样。
    stage.SetEditTarget(stage.GetRootLayer())

    rigid_body_count = 0

    # 完整遍历，不在遇到父刚体时停止。
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        contact_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        contact_api.CreateThresholdAttr().Set(0.0)

        rigid_body_count += 1
        print(f"[CONTACT] {prim.GetPath()}")

    if rigid_body_count == 0:
        raise RuntimeError("No rigid bodies found; USD was not saved.")


    # 备份修改前的入口层。
    from shutil import copy2

    backup_path = usd_path.with_name(
        "x2_ultra.before_assembly_filters.usda"
    )
    if not backup_path.exists():
        copy2(usd_path, backup_path)

    # 原头部过滤，加上四对已验证的装配近邻。
    collision_pairs = [
        ("head_pitch_link", "torso_link"),
        ("left_ankle_roll_link", "left_knee_link"),
        ("right_ankle_roll_link", "right_knee_link"),
        ("left_wrist_roll_link", "left_wrist_yaw_link"),
        ("right_wrist_roll_link", "right_wrist_yaw_link"),
    ]

    required_names = {
        name
        for pair in collision_pairs
        for name in pair
    }

    bodies = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        name = prim.GetName()
        if name not in required_names:
            continue

        if name in bodies:
            raise RuntimeError(f"Duplicate rigid body: {name}")

        bodies[name] = prim

    missing = required_names - bodies.keys()
    if missing:
        raise RuntimeError(
            f"Cannot find rigid bodies: {sorted(missing)}"
        )

    for first, second in collision_pairs:
        filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(
            bodies[first]
        )
        filtered_pairs.CreateFilteredPairsRel().AddTarget(
            bodies[second].GetPath()
        )

        print(f"[FILTER] {first} <-> {second}")

    print(f"[BACKUP] {backup_path}")


    stage.GetRootLayer().Save()

    print(f"\nEnabled contact reports on {rigid_body_count} rigid bodies.")
    print(f"Saved: {usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
