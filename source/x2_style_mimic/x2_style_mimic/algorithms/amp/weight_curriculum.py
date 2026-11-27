"""根据完整零辅助恢复回合调整 AMP 权重。"""

from collections import deque
import math


class AMPWeightCurriculum:
    def __init__(self, cfg, fixed_weight):
        self.cfg = dict(cfg)
        self.enabled = bool(self.cfg.get("enabled", False))

        self.minimum = float(self.cfg.get("min_weight", 2.0))
        self.maximum = float(self.cfg.get("max_weight", 10.0))
        self.increment = float(self.cfg.get("increment", 0.5))      # 每次调整 AMP 权重的步长。

        self.interval = int(self.cfg.get("check_interval", 100))    # 检查权重调整的间隔。
        self.window_size = int(self.cfg.get("window_size", 2000))   # 统计成功率时最多使用最近多少个回合的数据。
        self.min_episodes = int(self.cfg.get("min_episodes", 500))  # 开始调整前至少需要积累的回合数。

        # 成功率达到该值，满足“提高 AMP 权重”的条件之一。
        self.raise_success = float(self.cfg.get("raise_success_rate", 0.80))
        # 稳定成功率达到该值，满足“提高 AMP 权重”的条件之一。
        self.raise_stable = float(self.cfg.get("raise_stable_rate", 0.75))
        # 成功率低于该值，满足“降低 AMP 权重”的条件之一。
        self.lower_success = float(self.cfg.get("lower_success_rate", 0.60))
        # 稳定成功率低于该值，满足“降低 AMP 权重”的条件之一。
        self.lower_stable = float(self.cfg.get("lower_stable_rate", 0.55))

        values = (
            self.minimum,
            self.maximum,
            self.increment,
            float(fixed_weight),
        )
        if not all(math.isfinite(x) for x in values):
            raise ValueError("AMP 权重必须为有限数")
        if not 0.0 <= self.minimum <= self.maximum:
            raise ValueError("AMP 权重范围不合法")
        if self.increment <= 0.0:
            raise ValueError("AMP 权重增量必须大于零")
        if not 1 <= self.min_episodes <= self.window_size:
            raise ValueError("AMP 课程样本窗口不合法")
        if self.interval < 1:
            raise ValueError("AMP 课程检查间隔必须大于零")
        for lower, upper in (
            (self.lower_success, self.raise_success),
            (self.lower_stable, self.raise_stable),
        ):
            if not 0.0 <= lower < upper <= 1.0:
                raise ValueError("AMP 课程成功率阈值不合法")

        self.weight = (
            self.minimum if self.enabled else float(fixed_weight)
        )
        # 记录已经进行的训练迭代次数。通常每次调用更新函数时增加，用于判断是否到了 check_interval 检查时机
        self.iterations = 0
        # 记录自上次检查以来新增的回合数量。用于判断是否已经积累了足够的样本，达到 min_episodes 后才进行权重调整。
        self.new_episodes = 0
        # 记录表现不佳的统计窗口数量。一般用于防止因为一次短暂失败就立刻降低 AMP 权重，可能要求连续多个窗口表现不好后才降低。
        self.bad_windows = 0
        self.records = deque(maxlen=self.window_size)

    def observe(self, succeeded, stable):
        """输入已经筛选过的完整回合结果，使用 CPU Python 列表。"""
        if not self.enabled:
            return
        if len(succeeded) != len(stable):
            raise ValueError("成功与稳定统计长度不同")

        for success, end_stable in zip(succeeded, stable):
            self.records.append((bool(success), bool(end_stable)))

        self.new_episodes += len(succeeded)

    def advance(self):
        """每次 PPO update 结束调用一次。"""
        self.iterations += 1
        if not self.enabled:
            return

        if self.iterations % self.interval != 0:
            return
        if (
            len(self.records) < self.min_episodes
            or self.new_episodes < self.min_episodes
        ):
            return

        success_rate, stable_rate = self.rates()
        self.new_episodes = 0

        if (
            success_rate >= self.raise_success
            and stable_rate >= self.raise_stable
        ):
            self.weight = min(
                self.maximum, self.weight + self.increment
            )
            self.bad_windows = 0
        elif (
            success_rate < self.lower_success
            or stable_rate < self.lower_stable
        ):
            self.bad_windows += 1
            if self.bad_windows >= 2:
                self.weight = max(
                    self.minimum, self.weight - self.increment
                )
                self.bad_windows = 0
        else:
            self.bad_windows = 0

        # 使用独立的新统计窗口，避免旧成功回合推动连续升权。
        self.records.clear()

    def rates(self):
        if not self.records:
            return float("nan"), float("nan")
        count = len(self.records)
        return (
            sum(x[0] for x in self.records) / count,
            sum(x[1] for x in self.records) / count,
        )

    def state_dict(self):
        return {
            "weight": self.weight,
            "iterations": self.iterations,
            "new_episodes": self.new_episodes,
            "bad_windows": self.bad_windows,
            "records": list(self.records),
        }

    def load_state_dict(self, state):
        weight = float(state["weight"])
        if not math.isfinite(weight):
            raise ValueError("checkpoint AMP 权重不是有限数")
        if self.enabled and not self.minimum <= weight <= self.maximum:
            raise ValueError("checkpoint AMP 权重超出课程范围")

        self.weight = weight
        self.iterations = int(state["iterations"])
        self.new_episodes = int(state["new_episodes"])
        self.bad_windows = int(state["bad_windows"])
        self.records.clear()
        self.records.extend(
            (bool(a), bool(b)) for a, b in state["records"]
        )
