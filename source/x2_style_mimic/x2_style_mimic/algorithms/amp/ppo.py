"""在 RSL-RL PPO 上加入 AMP 风格奖励和判别器更新。"""

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO

from x2_style_mimic.algorithms.amp import (
    AMPDiscriminator,
    AMPReplayBuffer,
)
from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.amp import (
    AMPReferenceMotion,
)

class AMPPPO(PPO):

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        step_dt = cfg["algorithm"]["amp_cfg"]["step_dt"]        # 从嵌套字典中读取配置

        if not math.isclose(
            env.unwrapped.step_dt,
            step_dt,
            abs_tol=1e-8 ,
        ):
            raise ValueError("AMP 参考间隔与环境控制周期不同")

        # 复用原有 Actor、Critic 和 PPO storage 的创建流程。
        return PPO.construct_algorithm(
            obs,
            env,
            cfg,
            device,
        )

    def __init__(self, *args, amp_cfg, **kwargs):
        if kwargs.get("multi_gpu_cfg") is not None:
            raise ValueError("AMP v0 当前只支持单 GPU 训练")

        super().__init__(*args, **kwargs)

        if self.rnd is not None:
            raise ValueError("AMP v0 暂不与 RND 同时启用")

        self.amp_cfg = deepcopy(amp_cfg)

        self.amp_reference = AMPReferenceMotion(
            motion_file=amp_cfg["motion_file"],
            joint_names=amp_cfg["joint_names"],
            device=self.device,
            step_dt=amp_cfg["step_dt"],
        )

        self.amp_discriminator = AMPDiscriminator(
            self.amp_reference.states,
            learning_rate=amp_cfg["learning_rate"],
            gradient_penalty=amp_cfg["gradient_penalty"],
        )

        self.amp_buffer = AMPReplayBuffer(
            capacity=amp_cfg["buffer_capacity"],
            device=self.device,
        )

        # 恢复训练时检查参考文件是否发生变化。
        self._amp_motion_hash = hashlib.sha256(
            Path(amp_cfg["motion_file"]).read_bytes()
        ).hexdigest()

        self._amp_current = None        # 初始化动作前状态缓存。

        self._amp_task_sum = 0.0        # 初始化任务奖励统计累计值。
        self._amp_style_sum = 0.0       # 初始化原始风格奖励统计累计值。
        self._amp_step_count = 0        # 初始化已统计的环境步数量。

    def act(self, obs):
        '''内部保存 AMP 状态 外部返回 PPO 动作'''
        # 保存动作前状态，防止环境刷新观测时覆盖。
        self._amp_current = (
            obs["amp"].to(self.device).clone()
        )

        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        '''处理一次环境交互'''
        if self._amp_current is None:
            raise RuntimeError("必须先调用 act 再记录 AMP 转移")

        # 这里包含终止前状态，而非自动重置后的出生状态。
        following = extras["amp_next_state"].to(self.device)

        terminal_mask = extras["amp_terminal_mask"].to(self.device)

        if not torch.equal(
            terminal_mask,
            dones.bool()
        ):
            raise RuntimeError("AMP 终止掩码与 PPO dones 不一致")

        raw_style = self.amp_discriminator.style_reward(
            self._amp_current,      # 动作前 AMP 状态。
            following,              # 真实动作后 AMP 状态。
        )

        # 终止步也已正确保存，所有这些转移均有效。
        self.amp_buffer.add(
            self._amp_current,
            following,
            torch.ones_like(
                dones,
                dtype=torch.bool,
            ),
        )

        self._amp_task_sum += rewards.mean().item()     # 先对本步所有环境的任务奖励求平均，再转为 Python 数字，累加到统计量中。
        self._amp_style_sum += raw_style.mean().item()  # 累计本步所有环境的平均原始风格奖励。
        self._amp_step_count += 1

        # RewardManager 已对任务奖励乘过 dt。
        # 风格奖励也乘 dt，避免其量级随控制频率改变
        scale = (
            self.amp_cfg["style_reward_weight"]
            * self.amp_cfg["step_dt"]
        )

        # 原位更新，使 runner 记录的回报与 PPO 使用值一致。
        # 计算每个环境实际加入的风格奖励。把风格奖励原位加到任务奖励上。
        rewards.add_(scale * raw_style)

        super().process_env_step(
            obs,
            rewards,
            dones,
            extras,
        )

        self._amp_current = None    # 清除已使用的动作前状态。

    def update(self):
        # PPO 使用本轮采集时已经计算好的混合奖励。
        metrics = super().update()      # 调用父类 PPO 更新 Actor/Critic，取得其返回的训练指标字典。

        totals = {}     # 创建空字典，稍后累计多次判别器更新的指标。
        updates = self.amp_cfg["updates_per_iteration"]     # 读取每轮判别器更新次数
        batch_size = self.amp_cfg["batch_size"]             # 读取每次更新的样本批量

        # 此处已经离开 runner 的 inference_mode。
        for _ in range(updates):
            first, second = self.amp_reference.sample(
                batch_size
            )

            result = self.amp_discriminator.update(
                reference_pairs=torch.cat(
                    (first, second),
                    dim=-1,
                ),
                policy_pairs=self.amp_buffer.sample(
                    batch_size
                ),
            )

            for key, value in result.items():       # 遍历返回字典中的每一组键值。
                totals[key] = (
                    totals.get(key, 0.0) + value    # 读取之前累计的值，再加上这一次的值。如果之前没有这个键，就从 0.0 开始。
                )

        for key, value in totals.items():
            metrics[f"amp/{key}"] = value / updates # 取平均值，并加入总指标字典。

        count = max(self._amp_step_count, 1)
        raw_style = self._amp_style_sum / count     # 计算本轮采集期间的平均原始风格奖励。

        metrics["amp/style_reward_raw"] = raw_style
        metrics["amp/style_reward_scaled"] = (
            raw_style
            * self.amp_cfg["style_reward_weight"]
            * self.amp_cfg["step_dt"]
        )
        metrics["amp/task_reward"] = (              # 记录任务奖励平均值。
            self._amp_task_sum / count
        )
        metrics["amp/buffer_size"] = float(         # 读取缓冲区有效转移数量。
            self.amp_buffer.size
        )

        self._amp_task_sum = 0.0
        self._amp_style_sum = 0.0
        self._amp_step_count = 0

        return metrics

    def save(self):
        checkpoint = super().save()

        checkpoint["amp"] = {
            "feature_version": (
                self.amp_reference.feature_version
            ),
            "motion_sha256": self._amp_motion_hash,
            "settings": {
                key: value
                for key, value in self.amp_cfg.items()      # 遍历 AMP 配置的全部键值。
                if key != "motion_file"                     # 只保留键不是 "motion_file" 的项。
            },
            "discriminator": (
                self.amp_discriminator.state_dict()
            ),
            "optimizer": (
                self.amp_discriminator.optimizer.state_dict()
            ),
        }

        return checkpoint

    def load(self, loaded_dict, load_cfg=None, strict=True):
        load_amp = (
            load_cfg is None                # 没有提供恢复选项时，结果为 True，默认加载 AMP。
            or load_cfg.get("amp", True)    # 如果提供了恢复选项，就读取 "amp" 的值；没有该键时，默认 True。
        )

        if load_amp:
            amp = loaded_dict.get("amp")

            if amp is None:
                raise ValueError("该 checkpoint 没有 AMP 状态，不能直接恢复 AMP 训练")

            settings = {
                key: value
                for key, value in self.amp_cfg.items()
                if key != "motion_file"
            }

            if (
                amp["feature_version"]
                != self.amp_reference.feature_version
                or amp["motion_sha256"]
                != self._amp_motion_hash
                or amp["settings"] != settings
            ):
                raise ValueError("AMP 参考数据、特征版本或训练配置与 checkpoint 不一致")

        load_iteration = super().load(
            loaded_dict,
            load_cfg,
            strict,
        )

        if load_amp:
            self.amp_discriminator.load_state_dict(
                amp["discriminator"],
                strict=strict,
            )

            if (
                load_cfg is None
                or load_cfg.get("optimizer", False)
            ):
                self.amp_discriminator.optimizer.load_state_dict(
                    amp["optimizer"]
                )

        # 不保存体积较大的回放缓冲区。
        # 恢复后使用新的 rollout 重新填充。
        self.amp_buffer.size = 0
        self.amp_buffer.position = 0
        self._amp_current = None

        return load_iteration