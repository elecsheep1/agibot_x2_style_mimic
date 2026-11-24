"""只读加载检查点，用新采集动作检查 1／8／32 步预测；不更新网络或统计量。"""
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
from getup_config import parse_config


def main():
    if any(flag in sys.argv for flag in ('--help', '-h', '--print-config')):
        cfg, arguments = parse_config()
        if arguments.print_config:
            print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        return
    import numpy as np
    import torch
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
    from getup_runtime import build_runtime
    cfg, launcher = parse_config(launcher_hooks=(add_launcher_args, setup_preset_cli))
    cfg.validate()
    if cfg.resume is None:
        raise ValueError('模型诊断必须通过 --resume 指定检查点；不会执行恢复训练')
    cfg.prepare_output()
    sys.argv = [sys.argv[0]]
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_getup_env_cfg import X2SuperMimicGetupEnvCfg
    env_cfg = X2SuperMimicGetupEnvCfg()
    env_cfg.scene.num_envs, env_cfg.seed, env_cfg.sim.device = cfg.num_envs, cfg.seed, cfg.device
    with launch_simulation(env_cfg, launcher):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        checkpoint = torch.load(cfg.resume, map_location=cfg.device, weights_only=False)
        env = X2SuperMimicEnv(env_cfg)
        try:
            runtime = build_runtime(env, cfg, checkpoint)
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'checks'))
            from audit_supermimic_training import audit_predictions
            report = audit_predictions(env, runtime.trainer, runtime.context,
                                       batch_size=cfg.world_batch_size, seed=cfg.eval_seed)
            report.update(checkpoint=str(cfg.resume.resolve()), iteration=checkpoint['iteration'],
                          optimizer_updates=0, state_frames_in_training=list(cfg.world_horizons))
            cfg.output.mkdir(parents=True, exist_ok=False)
            (cfg.output / 'prediction_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            print('[预测诊断] ' + json.dumps(report['endpoint_errors'], ensure_ascii=False), flush=True)
            print(f'[完成] {cfg.output}', flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
