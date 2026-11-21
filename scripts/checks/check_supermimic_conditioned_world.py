"""仅检查带辅助力和步末事件的世界模型监督更新，不运行 actor。"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'supermimic'))


def main():
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
    from getup_config import parse_config
    cfg, launcher = parse_config(launcher_hooks=(add_launcher_args, setup_preset_cli))
    cfg.validate()
    cfg.prepare_output()
    sys.argv = [sys.argv[0]]
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_getup_env_cfg import X2SuperMimicGetupEnvCfg
    env_cfg = X2SuperMimicGetupEnvCfg()
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed, env_cfg.sim.device = cfg.seed, cfg.device
    env_cfg.assist_zero_probability = 0.0
    env_cfg.assist_curriculum_enabled = False
    env_cfg.episode_length_s = .12
    with launch_simulation(env_cfg, launcher):
        import torch
        from getup_runtime import build_runtime
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import X2SuperMimicEnv
        env = X2SuperMimicEnv(env_cfg)
        try:
            runtime = build_runtime(env, cfg, None)
            runtime.runner.collect(cfg.warmup_steps)
            batch = runtime.runner.sample_world_batch(cfg.batch_size, cfg.world_horizons[0], offline_world_fraction=0.)
            assert (batch['assist_wrench_w'][:, :, 2] > 0).any()
            metrics = runtime.trainer.update_world_model(batch)
            assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
            from x2_style_mimic.algorithms.supermimic.imagined_env import ImaginedGetupEnv
            imagined = ImaginedGetupEnv(runtime.trainer.world_model, runtime.trainer.assistance_cfg)
            imagined.reset(batch['states'][:, 0], batch['context'],
                           {key: value[:, 0] for key, value in batch['task_memory'].items()})
            imagined.step(batch['actions'][:, 0])
            torch.testing.assert_close(imagined.applied_wrench_w, batch['assist_wrench_w'][:, 0], atol=1e-5, rtol=1e-6)
            metrics['first_step_assistance_matches_recording'] = True
            cfg.output.mkdir(parents=True)
            (cfg.output / 'report.json').write_text(json.dumps(metrics, indent=2))
            print('[带辅助世界模型监督检查通过]', metrics, flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
