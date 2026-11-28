"""独立 X2 SuperMimic 参考跟踪训练／验收入口。"""
import argparse
from dataclasses import fields, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import tomllib

PROJECT = Path(__file__).resolve().parents[2]


def update_config(target, values):
    """严格递归配置：未知键立即报错，防止配置被静默忽略。"""
    allowed = {f.name for f in fields(target)}
    for key,value in values.items():
        if key not in allowed:
            raise ValueError(f'未知训练配置：{key}')
        current = getattr(target,key)
        if is_dataclass(current):
            if not isinstance(value,dict):
                raise ValueError(f'配置组必须是表：{key}')
            update_config(current,value)
        else:
            if isinstance(current,tuple): value = tuple(value)
            if key=='compile' and value=='none': value=None
            setattr(target,key,value)


def main():
    from isaaclab_tasks.utils import add_launcher_args,setup_preset_cli,launch_simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/train.toml')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--num-envs',type=int)
    parser.add_argument('--iterations',type=int)
    parser.add_argument('--run-name')
    parser.add_argument('--motion-manifest',type=Path)
    parser.add_argument('--urdf',type=Path)
    parser.add_argument('--print-config',action='store_true')
    add_launcher_args(parser)
    args,unknown = setup_preset_cli(parser)
    if unknown: parser.error(str(unknown))
    from x2_style_mimic.algorithms.supermimic_tracking.config import TrackingTrainCfg
    cfg = TrackingTrainCfg()
    update_config(cfg,tomllib.loads(args.config.read_text()))
    if args.evaluate and not args.resume: parser.error('验收需要 --resume 指定该入口保存的检查点')
    if args.evaluate:
        import torch
        checkpoint = torch.load(args.resume,map_location='cpu',weights_only=False)
        update_config(cfg,checkpoint['config'])
    for name in ('num_envs','iterations','run_name','device'):
        value = getattr(args,name,None)
        if value is not None: setattr(cfg,name,value)
    if args.resume: cfg.resume = str(args.resume.resolve())
    cfg.policy_trainer.obs_noise = cfg.obs_noise
    cfg.validate()
    from dataclasses import asdict
    if args.print_config:
        print(json.dumps(asdict(cfg),indent=2,ensure_ascii=False))
        return
    from x2_style_mimic.tasks.manager_based.x2_supermimic_tracking.env_cfg import X2SuperMimicTrackingEnvCfg
    env_cfg = X2SuperMimicTrackingEnvCfg()
    env_cfg.scene.num_envs,env_cfg.seed,env_cfg.sim.device = cfg.num_envs,cfg.seed,cfg.device
    env_cfg.domain_randomization = cfg.domain_randomization and not args.evaluate
    env_cfg.reset_noise = cfg.reset_noise and not args.evaluate
    env_cfg.obs_noise = cfg.obs_noise and not args.evaluate
    env_cfg.actions.joint_pos.noise_std = 0. if args.evaluate else cfg.action_noise_std
    env_cfg.observations.policy.history_length = cfg.policy.history
    env_cfg.root_height_threshold = cfg.world_model.height_threshold
    env_cfg.root_tilt_threshold = cfg.world_model.tilt_threshold
    env_cfg.commands.motion.future_frames = cfg.policy_trainer.future_frames
    env_cfg.commands.motion.sampling_mode = 'start' if args.evaluate else 'return'
    if args.motion_manifest: env_cfg.motion_manifest = str(args.motion_manifest.resolve())
    if args.urdf: env_cfg.urdf = str(args.urdf.resolve())
    if not Path(env_cfg.urdf).is_file(): raise FileNotFoundError(f'找不到 X2 URDF：{env_cfg.urdf}')
    if args.evaluate:
        import numpy as np
        manifest = Path(env_cfg.motion_manifest)
        entries = json.loads(manifest.read_text())['motions']
        longest = 0
        for entry in entries:
            with np.load(manifest.parent/entry['file'],allow_pickle=False) as data:
                longest = max(longest,len(data['time_s']))
        env_cfg.episode_length_s = (longest+1)*env_cfg.sim.dt*env_cfg.decimation
    output = PROJECT/'logs/supermimic_tracking'/f'{datetime.now():%Y-%m-%d_%H-%M-%S}_{cfg.run_name}'
    sys.argv = [sys.argv[0]]
    with launch_simulation(env_cfg,args):
        import random
        import numpy as np
        import torch
        from x2_style_mimic.tasks.manager_based.x2_supermimic_tracking.env import X2SuperMimicTrackingEnv
        from x2_style_mimic.algorithms.supermimic_tracking.training import TrackingTraining,evaluate
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        env = X2SuperMimicTrackingEnv(env_cfg)
        try:
            if args.evaluate: evaluate(env,cfg,args.resume,output)
            else: TrackingTraining(env,cfg,output).run()
        finally:
            env.close()


if __name__=='__main__':
    main()
