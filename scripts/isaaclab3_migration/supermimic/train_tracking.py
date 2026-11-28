"""IsaacLab 3.0 有界参考跟踪：冻结策略预热，通过留出检查后才做短训。"""
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import random
import sys
import json
import tomllib
from zoneinfo import ZoneInfo
from diagnose_tracking import load_file,PROJECT
from warmup_tracking import PretrainingCfg,prepare

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'configs/bounded_warmup_smoke.toml')
    parser.add_argument('--head-init',type=Path)
    parser.add_argument('--urdf',type=Path,default=Path.home()/'robotics_ws/agibot_model/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra.urdf')
    parser.add_argument('--run-name')
    parser.add_argument('--print-config',action='store_true')
    if '--print-config' in sys.argv:
        parser.add_argument('--device')
        parser.add_argument('--viz')
    else:
        from isaaclab.app import add_launcher_args
        add_launcher_args(parser)
    args=parser.parse_args()
    legacy=load_file('_tracking_legacy_config',PROJECT/'scripts/supermimic_tracking/train.py')
    migration=load_file('_tracking_lab3_compat',HERE.parent/'train_supermimic_tracking.py')
    from x2_style_mimic.algorithms.supermimic_tracking.config import TrackingTrainCfg
    cfg=TrackingTrainCfg();warmup=PretrainingCfg()
    settings=tomllib.loads(args.config.read_text())
    legacy.update_config(warmup,settings.pop('pretraining',{}))
    legacy.update_config(cfg,settings)
    if args.head_init:warmup.head_init=str(args.head_init.expanduser().resolve())
    if args.run_name:cfg.run_name=args.run_name
    if args.device:cfg.device=args.device
    cfg.policy_trainer.obs_noise=cfg.obs_noise
    cfg.validate();warmup.validate(cfg.num_envs,cfg.replay_capacity)
    if cfg.resume:parser.error('此预热验证入口不接受旧检查点续训')
    if args.print_config:
        print(json.dumps({'training':asdict(cfg),'pretraining':asdict(warmup)},indent=2,ensure_ascii=False));return
    if not warmup.head_init or not (PROJECT/warmup.head_init).is_file():
        raise FileNotFoundError('找不到独立补训后的失败头，请设置 pretraining.head_init 或 --head-init')
    from x2_style_mimic.tasks.manager_based.x2_supermimic_tracking.env_cfg import X2SuperMimicTrackingEnvCfg
    env_cfg=X2SuperMimicTrackingEnvCfg()
    env_cfg.scene.num_envs,env_cfg.seed,env_cfg.sim.device=cfg.num_envs,cfg.seed,cfg.device
    env_cfg.domain_randomization,env_cfg.reset_noise,env_cfg.obs_noise=cfg.domain_randomization,cfg.reset_noise,cfg.obs_noise
    env_cfg.actions.joint_pos.noise_std=cfg.action_noise_std
    env_cfg.observations.policy.history_length=cfg.policy.history
    env_cfg.commands.motion.future_frames=cfg.policy_trainer.future_frames
    env_cfg.root_height_threshold,env_cfg.root_tilt_threshold=cfg.world_model.height_threshold,cfg.world_model.tilt_threshold
    env_cfg.urdf=str(args.urdf.expanduser().resolve())
    if not Path(env_cfg.urdf).is_file():raise FileNotFoundError(env_cfg.urdf)
    migration._replace_null_manager_configs(env_cfg)
    output=PROJECT/'logs/supermimic_tracking'/f'{datetime.now(ZoneInfo("Asia/Shanghai")):%Y-%m-%d_%H-%M-%S}_{cfg.run_name}'
    sys.argv=[sys.argv[0]]
    from isaaclab.app import launch_simulation
    with launch_simulation(env_cfg,args):
        import numpy as np
        import torch
        from x2_style_mimic.tasks.manager_based.x2_supermimic_tracking.env import X2SuperMimicTrackingEnv
        from x2_style_mimic.algorithms.supermimic_tracking.training import TrackingTraining
        random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
        env=X2SuperMimicTrackingEnv(env_cfg)
        training=None
        try:
            training=TrackingTraining(env,cfg,output)
            (output/'pretraining_config.json').write_text(json.dumps(asdict(warmup),indent=2,ensure_ascii=False))
            print(f'[输出目录] {output}',flush=True)
            report=prepare(training,warmup,PROJECT)
            if report['ready_for_joint_training'] and warmup.joint_if_ready:
                print('[预热通过] 开始短程联合更新',flush=True)
                training.run()
            else:
                print('[停止于预热] 未执行 Actor／Critic 更新，详见 warmup_report.json',flush=True)
        finally:
            if training is not None:training.writer.close()
            env.close()


if __name__=='__main__':main()
