"""独立参考跟踪实验的训练配置；环境与奖励参数在 tracking/env_cfg.py。"""
from dataclasses import dataclass, field
from .world_model import WorldModelCfg
from .wm_trainer import WMTrainerCfg
from .nets import PolicyCfg, CriticCfg
from .policy_trainer import PolicyTrainerCfg


@dataclass
class TrackingTrainCfg:
    run_name: str = 'x2_reference_tracking_scratch'
    num_envs: int = 4096
    seed: int = 43
    device: str = 'cuda:0'
    iterations: int = 320000  # 一轮是一条向量环境步，不是旧入口的 32 步采集块。
    replay_capacity: int = 1000
    warmup_steps: int = 40
    updates_per_env_step: float = .4
    lr_final_ratio: float = .03
    lr_decay_fraction: float = 250000/320000
    log_interval: int = 50
    checkpoint_interval: int = 4000
    resume: str | None = None
    domain_randomization: bool = True
    reset_noise: bool = True
    obs_noise: bool = True
    action_noise_std: float = .1
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)

    def validate(self):
        if min(self.num_envs,self.iterations,self.log_interval,self.checkpoint_interval,
               self.wm_trainer.window,self.wm_trainer.batches_per_update,self.wm_trainer.batch_size,
               self.policy_trainer.batch_size,self.policy_trainer.horizon,self.policy_trainer.history) <= 0:
            raise ValueError('环境数、训练次数、窗口和批量必须为正数')
        if self.replay_capacity <= max(self.wm_trainer.window+2,self.policy_trainer.history+2):
            raise ValueError('回放容量不足以保留监督窗口和策略历史')
        if self.warmup_steps < self.wm_trainer.window+1:
            raise ValueError('预热必须足以采满一条世界模型监督窗口')
        if not 0 < self.updates_per_env_step <= 10:
            raise ValueError('每环境步更新比例必须处于 (0,10]')
        if self.policy.history != self.policy_trainer.history:
            raise ValueError('策略与想象起点历史长度必须一致')
        if self.policy.goal_dim != self.policy_trainer.future_frames*72:
            raise ValueError('X2 每个目标帧为 72 维，目标维度与未来帧数不一致')
        if self.policy_trainer.gamma != .99:
            raise ValueError("当前回报采样器与 Actor 均使用 gamma=.99；修改时需同时更新环境配置")
        if not 0 <= self.policy_trainer.gae_lambda <= 1:
            raise ValueError("lambda 必须处于 [0,1]")
        if self.policy_trainer.batch_size*self.policy_trainer.horizon < self.policy_trainer.critic_minibatches:
            raise ValueError("Critic minibatch 数不能超过想象样本数量")
        if self.action_noise_std < 0:
            raise ValueError('动作噪声标准差不能为负数')
        if not 0 < self.lr_final_ratio <= 1 or not 0 < self.lr_decay_fraction <= 1:
            raise ValueError('学习率衰减比例必须处于 (0,1]')
        if self.policy_trainer.anchor_obs:
            raise ValueError('首版使用论文的纯本体观测，请关闭额外锚点观测')
