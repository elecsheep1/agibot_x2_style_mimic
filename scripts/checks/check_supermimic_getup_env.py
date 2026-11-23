"""验证重建后的起身环境、传感器匹配和自动重置状态捕获。"""
import argparse
import json
from pathlib import Path
import re
import sys


def main():
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    add_launcher_args(parser)
    args, rest = setup_preset_cli(parser)
    if rest:
        parser.error(str(rest))
    sys.argv = [sys.argv[0]]
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_getup_env_cfg import X2SuperMimicGetupEnvCfg
    cfg = X2SuperMimicGetupEnvCfg()
    cfg.scene.num_envs = 2
    cfg.episode_length_s = 0.12
    with launch_simulation(cfg, args):
        import torch
        import omni.usd
        from pxr import UsdPhysics
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv
        env = X2SuperMimicEnv(cfg)
        try:
            env.reset()
            sensor_matches = {}
            resolved_bodies = {}
            prims = list(omni.usd.get_context().get_stage().Traverse())
            for name in ('left_hand_ground_contact', 'right_hand_ground_contact', 'left_foot_ground_contact', 'right_foot_ground_contact'):
                expression = env.scene.sensors[name].cfg.prim_path.replace('{ENV_REGEX_NS}', '/World/envs/env_.*')
                sensor_matches[name] = [dict(path=str(p.GetPath()), rigid_body=p.HasAPI(UsdPhysics.RigidBodyAPI))
                                        for p in prims if re.fullmatch(expression, str(p.GetPath()))]
                sensor = env.scene.sensors[name]
                resolved_bodies[name] = list(sensor.body_physx_view.prim_paths)
                assert sensor.num_sensors == 1, f'{name} 未按每环境匹配唯一刚体'
                assert all(omni.usd.get_context().get_stage().GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI)
                           for path in resolved_bodies[name])
            resets = 0
            for _ in range(18):
                before = env.read_world_state()
                before_steps = env.episode_length_buf.clone()
                _, reward, terminated, truncated, extras = env.step(torch.zeros(2, env.world_state_spec.num_actions, device=env.device))
                captured = extras['world_model']
                torch.testing.assert_close(captured['state'], before)
                torch.testing.assert_close(captured['reward'], reward)
                assert torch.isfinite(captured['next_state']).all()
                assert torch.equal(terminated | truncated, env._wm_terminal_mask)
                assert captured['schema_version'] == 4
                torch.testing.assert_close(captured['task_memory']['episode_steps'], before_steps)
                torch.testing.assert_close(captured['next_task_memory']['episode_steps'], before_steps + 1)
                resets += int((terminated | truncated).sum())
            assert resets > 0
            report = dict(automatic_resets=resets,
                          sensor_matches=sensor_matches, resolved_bodies=resolved_bodies)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            env.close()


if __name__ == '__main__':
    main()
