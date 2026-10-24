# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-motion transition fine-tuning: fresh optimizers and configurable exploration reset."""

import argparse
import contextlib
import importlib.metadata as metadata
import logging
import math
import os
import platform
import sys
import time
from datetime import datetime

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

logger = logging.getLogger(__name__)

import x2_style_mimic.tasks  # noqa: F401
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

RSL_RL_VERSION = "5.0.1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def reset_actor_std(algorithm, value):
    """Reset the installed RSL-RL Gaussian distribution after loading weights."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError("resume std must be finite and positive")
    distribution = algorithm.actor.distribution
    with torch.no_grad():
        if distribution.std_type == "scalar":
            before = distribution.std_param.detach().mean().item()
            distribution.std_param.fill_(value)
        elif distribution.std_type == "log":
            before = distribution.log_std_param.detach().exp().mean().item()
            distribution.log_std_param.fill_(math.log(value))
        else:
            raise ValueError(f"Unsupported Gaussian std type: {distribution.std_type}")
    print(f"[转换续训] 动作标准差均值 {before:.4f} -> {value:.4f}", flush=True)


# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="X2-Prone-MultiMotion-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
parser.add_argument("--resume-std", type=float, default=0.30,
                    help="Reset action standard deviation after loading the checkpoint.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
if not args_cli.resume:
    parser.error("This fine-tuning entry requires --resume")
if not math.isfinite(args_cli.resume_std) or args_cli.resume_std <= 0:
    parser.error("--resume-std must be finite and positive")

if args_cli.video:
    args_cli.enable_cameras = True


# Call an external callback if requested. This gives opportunity to external code to register the environments
# The function is expected to return a list of arguments that were not consumed by the callback.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
# The remaining arguments are the arguments that were not consumed by both this scripts
# argparser and (optionally) the external callback function. Both sides of this
# intersection share the same token vocabulary (the callback reads the user's
# original sys.argv), so preset tokens like ``physics=NAME`` compare correctly.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# -- check RSL-RL version ----------------------------------------------------
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        # For distributed training, launch_simulation() already resolved the
        # correct per-rank device; only apply a CLI --device override for
        # non-distributed runs (the default "cuda:0" would clobber the
        # per-rank device otherwise).
        if not args_cli.distributed:
            env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        # check for invalid combination of CPU device with distributed training
        if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError(
                "Distributed training is not supported when using CPU device. "
                "Please use GPU device (e.g., --device cuda) for distributed training."
            )

        # multi-gpu training configuration
        if args_cli.distributed:
            global_rank = int(os.getenv("RANK", "0"))
            # env_cfg.sim.device is resolved by launch_simulation() which
            # accounts for CUDA_VISIBLE_DEVICES restrictions.
            agent_cfg.device = env_cfg.sim.device

            # use global rank for seed diversity across all nodes
            seed = agent_cfg.seed + global_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        # specify directory for logging runs: {time-stamp}_{run_name}
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
        # change it (see PR #2346, comment-2819298849)
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        # set the IO descriptors export flag if requested
        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        else:
            logger.warning(
                "IO descriptors are only supported for manager based RL environments."
                " No IO descriptors will be exported."
            )

        # set the log directory for the environment (works for all environment types)
        env_cfg.log_dir = log_dir

        # Birth sampling belongs to the environment configuration.
        reset_params = env_cfg.events.reset_from_reference.params
        print(
            f"[训练出生] cfg 首帧概率={reset_params.get('initial_frame_probability', 0.5):.0%}，"
            f"转换区间={reset_params.get('transition_frame_ranges')}；区间为空时使用 start/end_phase。",
            flush=True,
        )

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # save resume path before creating a new log_dir
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # create runner from rsl-rl
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # configure_seed must be called after runner construction so that PyTorch deterministic settings
        # do not interfere with the runner's internal initialization.
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)
        # write git state to logs
        runner.add_git_repo_to_log(__file__)
        # Preserve policy/value normalization and AMP weights, but rebuild both optimizers.
        if agent_cfg.algorithm.class_name != "x2_style_mimic.algorithms.amp.ppo_window:AMPWindowPPO":
            if not hasattr(runner.alg, "amp_discriminator"):
                raise ValueError("This entry requires the multi-motion AMPWindowPPO algorithm")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path, load_cfg={
            "actor": True, "critic": True, "optimizer": False,
            "iteration": True, "rnd": False, "amp": True,
            "assist_curriculum": False,
        })
        reset_actor_std(runner.alg, args_cli.resume_std)
        print("[转换续训] 保留 actor/critic 及归一化、AMP 判别器权重；重建 PPO/AMP 优化器。", flush=True)
        print("[转换续训] 辅助课程采用当前环境配置；回放缓冲与历史重新积累。", flush=True)

        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        # run training
        try:
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        except KeyboardInterrupt:
            pass
        finally:
            env.close()


if __name__ == "__main__":
    main()
