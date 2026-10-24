"""多动作、五帧 AMP：新版算法，独立于旧版 AMPPPO。"""

from copy import deepcopy

import hashlib
from pathlib import Path
import torch
from rsl_rl.algorithms import PPO

from .discriminator import AMPWindowDiscriminator
from .history import AMPHistory
from .replay_buffer import AMPWindowReplayBuffer
from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.motion_dataset import (
    MotionDataset,
)
from ...tasks.manager_based.x2_style_mimic.mdp.amp import (
    AMP_FEATURE_VERSION,
)

class AMPWindowPPO(PPO):

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        amp_cfg = cfg["algorithm"]["amp_cfg"]

        if abs(env.unwrapped.step_dt - amp_cfg["step_dt"]) > 1e-8:
            raise ValueError("AMP 参考间隔与环境控制周期不同")

        algorithm = PPO.construct_algorithm(
            obs, env, cfg, device
        )
        algorithm._getup_env = env.unwrapped
        return algorithm

    def __init__(self, *args, amp_cfg, **kwargs):
        if kwargs.get("multi_gpu_cfg") is not None:
            raise ValueError("新版 AMP 暂只支持单 GPU")

        super().__init__(*args, **kwargs)

        if self.rnd is not None:
            raise ValueError("新版 AMP 暂不与 RND 同时启用")

        self.amp_cfg = deepcopy(amp_cfg)
        self.amp_num_steps = int(amp_cfg.get("num_steps", 5))

        self.amp_reference = MotionDataset(
            manifest_file=amp_cfg["manifest_file"],
            robot_joint_names=amp_cfg["robot_joint_names"],
            policy_joint_names=amp_cfg["joint_names"],
            device=self.device,
            step_dt=amp_cfg["step_dt"],
        )

        # 按动作权重抽样，估计参考特征的归一化统计量。
        # 不直接拼接全部帧，避免长动作主导统计量。
        reference_windows = (
            self.amp_reference.sample_amp_windows(
                batch_size=4096,                    # 采样多少个滑动窗口
                num_steps=self.amp_num_steps,       # 滑动窗口长度
            )["states"]                             # 形状为 [批量, num_steps, state_dim]
        )
        # 参考状态的形状为 [批量, num_steps, state_dim]，展开时间维后为 [批量 * num_steps, state_dim]。
        reference_states = reference_windows.flatten(start_dim=0, end_dim=1)

        self.amp_discriminator = AMPWindowDiscriminator(
            reference_states=reference_states,
            num_steps=self.amp_num_steps,
            learning_rate=amp_cfg["learning_rate"],
            gradient_penalty=amp_cfg["gradient_penalty"],
        )

        self.amp_buffer = AMPWindowReplayBuffer(
            capacity=amp_cfg["buffer_capacity"],
            state_dim=reference_states.shape[-1],
            num_steps=self.amp_num_steps,
            device=self.device,
        )

        # 加载时记录，恢复训练时核对参考数据是否一致。
        self._amp_reference_hash = self._reference_fingerprint()

        # 首次 act 时才能从观测确定并行环境数量
        self.amp_history = None
        self._amp_waiting_for_step = False

        self._amp_task_sum = 0.0        # 初始化任务奖励统计累计值。
        self._amp_style_sum = 0.0       # 初始化原始风格奖励统计累计值。
        self._amp_valid_sum = 0.0       # 初始化有效窗口统计累计值。
        self._amp_step_count = 0        # 初始化已统计的环境步数量。

    def act(self, obs):
        """内部保存 AMP 状态，外部返回 PPO 动作。"""
        if self._amp_waiting_for_step:
            raise RuntimeError("上一次 act 后尚未调用 process_env_step")

        current = obs["amp"].to(self.device)

        if self.amp_history is None:
            # Runner 的采集阶段使用 inference_mode。
            # 显式创建普通张量，便于恢复或清空历史。
            with torch.inference_mode(False):
                self.amp_history = AMPHistory(
                    num_envs=current.shape[0],
                    state_dim=current.shape[-1],
                    num_steps=self.amp_num_steps,
                    device=self.device,
                )

        self.amp_history.begin(current)      # 仅为刚出生的环境写入第一帧。
        actions = super().act(obs)
        self._amp_waiting_for_step = True
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        if not self._amp_waiting_for_step:
            raise RuntimeError("必须先调用 act 再记录 AMP 转移")

        following = extras["amp_next_state"].to(self.device)
        terminal_mask = extras[
            "amp_terminal_mask"
        ].to(self.device).bool()
        done_mask = dones.to(self.device).bool()

        if not torch.equal(terminal_mask, done_mask):
            raise RuntimeError("AMP 终止掩码与 PPO dones 不一致")

        windows, valid_mask = self.amp_history.advance(
            following, done_mask
        )       # 追加最新帧，返回窗口和有效掩码。

        # 历史不足五帧时，风格奖励明确为零。
        raw_style = torch.zeros(
            windows.shape[0],
            dtype=rewards.dtype,
            device=self.device,
        )
        if bool(valid_mask.any()):
            raw_style[valid_mask] = self.amp_discriminator.style_reward(
                windows[valid_mask]
            )

        # 缓冲区只保存完整、未跨回合的窗口。
        self.amp_buffer.add(windows, valid_mask)

        # 在加入 AMP 前记录纯任务奖励。
        self._amp_task_sum += rewards.mean().item()     # 先对本步所有环境的任务奖励求平均，再转为 Python 数字，累加到统计量中。
        self._amp_style_sum += raw_style.mean().item()  # 累计本步所有环境的平均原始风格奖励。
        self._amp_valid_sum += valid_mask.float().mean().item()  # 累计本步所有环境的平均有效窗口比例
        self._amp_step_count += 1

        scale = (
            self.amp_cfg["style_reward_weight"]
            * self.amp_cfg["step_dt"]
        )
        rewards.add_(scale * raw_style.to(rewards.device))

        super().process_env_step(obs, rewards, dones, extras)
        self._amp_waiting_for_step = False

    def update(self):
        if self._amp_waiting_for_step:
            raise RuntimeError("环境步尚未处理完成，不能更新")

        # Actor/Critic 使用采集时已经混合好的奖励。
        metrics = super().update()      # 调用父类 PPO 更新 Actor/Critic，取得其返回的训练指标字典。

        updates = int(self.amp_cfg["updates_per_iteration"])   # 读取每轮判别器更新次数
        batch_size = int(self.amp_cfg["batch_size"])
        if updates <=0 or batch_size <= 0:
            raise ValueError("AMP 更新次数或批量大小无效")

        totals = {}
        completed_updates = 0

        # 极短回合可能无法积累完整窗口，缓冲区为空时跳过。
        if self.amp_buffer.size > 0:
            for _ in range(updates):
                reference_windows = (
                    self.amp_reference.sample_amp_windows(
                        batch_size=batch_size,
                        num_steps=self.amp_num_steps,
                    )["states"]
                )
                policy_windows = self.amp_buffer.sample(batch_size)

                # 旧类的参数名叫 pairs，新版实际传入五帧窗口。
                result = self.amp_discriminator.update(
                    reference_pairs=reference_windows,
                    policy_pairs=policy_windows,
                )

                for key, value in result.items():
                    totals[key] = (totals.get(key, 0.0) + value)
                completed_updates += 1

        for key, value in totals.items():
            metrics[f"amp/{key}"] = value / completed_updates

        count = max(self._amp_step_count, 1)
        raw_style = self._amp_style_sum / count

        metrics["amp/style_reward_raw"] = raw_style
        metrics["amp/style_reward_scaled"] = (
            raw_style
            * self.amp_cfg["style_reward_weight"]
            * self.amp_cfg["step_dt"]
        )
        metrics["amp/task_reward"] = (
            self._amp_task_sum / count
        )
        metrics["amp/valid_window_ratio"] = (
            self._amp_valid_sum / count
        )
        metrics["amp/buffer_size"] = float(
            self.amp_buffer.size
        )
        metrics["amp/discriminator_updates"] = float(
            completed_updates
        )

        self._amp_task_sum = 0.0
        self._amp_style_sum = 0.0
        self._amp_valid_sum = 0.0
        self._amp_step_count = 0

        # 不清空历史：同一回合允许跨 PPO 迭代继续积累。
        return metrics

    def _reference_fingerprint(self):
        """同时校验清单和所有参考动作的内容。"""
        digest = hashlib.sha256()
        paths = [
            Path(self.amp_cfg["manifest_file"]),
            *self.amp_reference.paths,
        ]

        for path in paths:
            content = path.read_bytes()
            # 写入长度，明确不同文件内容的边界。
            digest.update(
                len(content).to_bytes(8, "little")
            )
            digest.update(content)

        return digest.hexdigest()

    def save(self):
        checkpoint = super().save()

        checkpoint["amp_window"] = {
            "feature_version": AMP_FEATURE_VERSION + "_window_v1",
            "num_steps": self.amp_num_steps,
            "reference_sha256": self._amp_reference_hash,
            "settings": deepcopy(self.amp_cfg),
            "discriminator": (
                self.amp_discriminator.state_dict()
            ),
            "optimizer": (
                self.amp_discriminator.optimizer.state_dict()
            ),
        }

        checkpoint["assist_curriculum"] = (
            self._getup_env._getup_assist.state_dict()
        )
        # 不保存回放缓冲区和环境历史。
        return checkpoint

    def load(self, loaded_dict, load_cfg=None, strict=True):
        load_amp = (
            load_cfg is None
            or load_cfg.get("amp", True)
        )

        if load_amp:
            amp = loaded_dict.get("amp_window")
            if amp is None:
                raise ValueError("该 checkpoint 没有新版五帧 AMP 状态")

            if (
                amp["feature_version"]
                != AMP_FEATURE_VERSION + "_window_v1"
                or amp["num_steps"] != self.amp_num_steps
                or amp["reference_sha256"]
                != self._amp_reference_hash
                or amp["settings"] != self.amp_cfg
            ):
                raise ValueError("参考数据、窗口长度或 AMP 配置不一致")

        load_iteration = super().load(
            loaded_dict, load_cfg, strict
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

        restore_curriculum = (
            load_cfg is None
            or load_cfg.get("assist_curriculum", True)
        )
        controller = self._getup_env._getup_assist

        if restore_curriculum and controller.curriculum_enabled:
            curriculum = loaded_dict.get("assist_curriculum")
            if curriculum is None:
                raise ValueError(
                    "checkpoint 缺少辅助力课程状态"
                )
            controller.load_state_dict(curriculum)

        # 历史与回放缓冲区不存档，恢复后重新积累。
        self.amp_buffer.clear()
        if self.amp_history is not None:
            self.amp_history.clear()

        self._amp_waiting_for_step = False
        self._amp_task_sum = 0.0
        self._amp_style_sum = 0.0
        self._amp_valid_sum = 0.0
        self._amp_step_count = 0

        return load_iteration
