"""SuperMimic 起身任务的真实转移附加字段。"""

import torch


@torch.no_grad()
def read_task_memory(env):
    progress = (
        env.termination_manager
        .get_term_cfg("progress")
        .func
    )

    return {
        "hold_steps": (
            progress._hold_steps.detach().clone()
        ),
        "episode_steps": (
            env.episode_length_buf.detach().clone()
        ),
    }


def transition_extra_fields(state_dim):
    fields = {}

    for group in (
        "task_memory",
        "next_task_memory",
    ):
        fields[(group, "hold_steps")] = (
            (),
            torch.long,
        )
        fields[(group, "episode_steps")] = (
            (),
            torch.long,
        )

    fields[("amp_reward_gate",)] = (
        (),
        torch.float32,
    )
    fields[("amp_policy_sample_mask",)] = (
        (),
        torch.bool,
    )

    return fields