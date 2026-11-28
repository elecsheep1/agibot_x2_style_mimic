"""独立 SuperMimic 参考跟踪任务注册。"""
import gymnasium as gym

gym.register(id='X2-SuperMimic-Tracking-v0',
    entry_point=__name__+'.env:X2SuperMimicTrackingEnv',disable_env_checker=True,
    kwargs={'env_cfg_entry_point':__name__+'.env_cfg:X2SuperMimicTrackingEnvCfg'})
