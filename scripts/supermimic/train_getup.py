"""SuperMimic + AMP 起身训练入口；默认算法配置见 agents/x2_supermimic_cfg.py。"""
import json
import random
import sys

sys.dont_write_bytecode = True

from getup_config import PROJECT, parse_config


def main():
    # 查看配置和帮助时不加载 Isaac Sim。
    if '--print-config' in sys.argv or '--help' in sys.argv or '-h' in sys.argv:
        cfg, arguments = parse_config()
        if arguments.print_config:
            print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        return

    import numpy as np
    import torch
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
    from getup_runtime import build_runtime
    from getup_training import GetupTrainingRun, file_hashes, semantic_file_hashes

    cfg, launcher = parse_config(launcher_hooks=(add_launcher_args, setup_preset_cli))
    cfg.validate()
    cfg.prepare_output()
    sys.argv = [sys.argv[0]]
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_getup_env_cfg import X2SuperMimicGetupEnvCfg

    source_before = file_hashes((PROJECT / 'source').rglob('*.py'))
    source_semantic_before = semantic_file_hashes((PROJECT / 'source').rglob('*.py'))
    env_cfg = X2SuperMimicGetupEnvCfg()
    env_cfg.scene.num_envs, env_cfg.seed, env_cfg.sim.device = cfg.num_envs, cfg.seed, cfg.device
    with launch_simulation(env_cfg, launcher):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        checkpoint = torch.load(cfg.resume, map_location=cfg.device, weights_only=False) if cfg.resume else None
        env = X2SuperMimicEnv(env_cfg)
        try:
            runtime = build_runtime(env, cfg, checkpoint)
            GetupTrainingRun(runtime, cfg, checkpoint, source_before, source_semantic_before).run()
        finally:
            env.close()


if __name__ == '__main__':
    main()
