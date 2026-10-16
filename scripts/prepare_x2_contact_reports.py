"""为 X2 嵌套刚体补充 PhysX 接触报告，修改本地 USD 入口层。"""

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
    project_root = Path(__file__).resolve().parents[1]

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

    stage.GetRootLayer().Save()

    print(f"\nEnabled contact reports on {rigid_body_count} rigid bodies.")
    print(f"Saved: {usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
