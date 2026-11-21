"""SuperMimic 真实转移的任务状态与接触监督读取。"""

import torch

from ..getup import ground_contact_force
from isaaclab.managers import SceneEntityCfg


GROUND_SENSOR_NAMES = (
    "left_foot_ground_contact",
    "right_foot_ground_contact",
    "left_hand_ground_contact",
    "right_hand_ground_contact",
    "head_yaw_ground_contact",
    "head_pitch_ground_contact",
)


@torch.no_grad()
def read_task_memory(env):
    """读取当前回合任务记忆，不使用可能属于旧回合的快照。"""

    assist = env._getup_assist

    progress = (
        env.termination_manager
        .get_term_cfg("progress")
        .func
    )

    return {
        "assist_force_cap_n": (
            assist.episode_force.detach().clone()
        ),
        "assist_withdrawn": (
            assist.withdrawn.detach().clone()
        ),
        "hold_steps": (
            progress._hold_steps.detach().clone()
        ),
        "episode_steps": (
            env.episode_length_buf.detach().clone()
        ),
    }


@torch.no_grad()
def read_contact_targets(env):
    """读取原任务实际使用的地面接触历史统计。

    ground_force_peak_n 的最后一维顺序由
    GROUND_SENSOR_NAMES 固定。
    """

    peaks = torch.stack(
        [
            ground_contact_force(
                env,
                SceneEntityCfg(name),
            )
            for name in GROUND_SENSOR_NAMES
        ],
        dim=-1,
    )

    return {
        # 保留历史力峰值，而非仅当前接触概率。
        "ground_force_peak_n": (
            peaks.detach().clone()
        ),

        # 保留原自碰撞奖励实际使用的控制步统计。
        "self_contact_cost": (
            env.scene.sensors["self_contact"]
            .snapshot()["penalty_raw"]
            .detach()
            .clone()
        ),
    }


def transition_extra_fields(state_dim):
    """回放附加字段规格。

    字典键为嵌套字段路径。
    shape 不包含环境维和时间维。
    """

    fields = {}

    task_fields = {
        "assist_force_cap_n": ((), torch.float32),
        "assist_withdrawn": ((), torch.bool),
        "hold_steps": ((), torch.long),
        "episode_steps": ((), torch.long),
    }

    # 把同一套任务记忆字段规格，分别登记到“当前任务记忆”和“下一步任务记忆”两个分组中。
    for group in (
        "task_memory",
        "next_task_memory",
    ):
        for name, spec in task_fields.items():
            fields[(group, name)] = spec

    fields.update({
        # 字段路径: (单个环境的数据形状, 数据类型)
        (
            "contact_targets",
            "ground_force_peak_n",
        ): ((6,), torch.float32),

        (
            "contact_targets",
            "self_contact_cost",
        ): ((), torch.float32),

        ("assist_wrench_w",): (
            (6,), torch.float32
        ),
        ("interval_delta_velocity_w",): (
            (6,), torch.float32
        ),
        ("interval_push_mask",): (
            (), torch.bool
        ),
        # 保存的是本步动作执行后、interval 随机推扰发生前的机器人完整状态，也就是本步奖励计算对应的状态。
        ("reward_state",): (
            (state_dim,), torch.float32
        ),
        ("amp_reward_gate",): (
            (), torch.float32
        ),
        ("amp_policy_sample_mask",): (
            (), torch.bool
        ),
    })

    return fields
