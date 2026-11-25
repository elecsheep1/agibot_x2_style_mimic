"""仅在脚本层组装已有的起身环境和训练组件。"""
import contextlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from getup_config import PROJECT
from getup_data import OfflineWorldReplay, read_world_checkpoint, shared_vector, action_bounds_from_calibration
from contact_assets import build_contact_geometry, contact_geometry_contract, validate_geometry_checkpoint
from getup_checkpointing import collect_optimizers


@dataclass
class GetupRuntime:
    env: Any
    trainer: Any
    runner: Any
    replay: Any
    offline: OfflineWorldReplay | None
    warmup_complete: bool
    modules: dict
    optimizers: dict
    context: torch.Tensor
    metadata: dict
    reset_reference: Any
    reference: Any
    control_ids: torch.Tensor
    all_reference_paths: set


def tensor(value):
    return value.torch if hasattr(value, 'torch') else value


def extend_task_memory_fields(fields, memory, num_envs):
    """按真实任务记忆注册当前／下一步字段，避免遗漏新增状态。"""
    result = dict(fields)
    for name, value in memory.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != num_envs:
            raise ValueError(f'任务记忆必须包含环境维 [N,...]：{name}')
        spec = (tuple(value.shape[1:]), value.dtype)
        for group in ('task_memory', 'next_task_memory'):
            path = (group, name)
            if path in result and result[path] != spec:
                raise ValueError(f'任务记忆注册与实际张量不一致：{path}')
            result[path] = spec
    return result
def build_runtime(env, args, resume):
    from x2_style_mimic.assets.x2_ultra_cfg import X2_ULTRA_USD
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.motion_dataset import MotionDataset
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.supermimic.transition_data import transition_extra_fields, read_task_memory
    from x2_style_mimic.algorithms.amp.discriminator import AMPWindowDiscriminator
    from x2_style_mimic.algorithms.supermimic.actor_critic import GoalConditionedActor, GoalConditionedCritic
    from x2_style_mimic.algorithms.supermimic.amp_features import DifferentiableAMPFeatures
    from x2_style_mimic.algorithms.supermimic.buffer import SequenceReplayBuffer
    from x2_style_mimic.algorithms.supermimic.context import read_sim_context
    from x2_style_mimic.algorithms.supermimic.heads import GetupTask, GetupTaskCfg
    from x2_style_mimic.algorithms.supermimic.observations import Proprioception
    from x2_style_mimic.algorithms.supermimic.runner import SuperMimicRunner
    from x2_style_mimic.algorithms.supermimic.style import DifferentiableStyleReward
    from x2_style_mimic.algorithms.supermimic.trainer import SuperMimicTrainer, TrainerCfg
    from x2_style_mimic.algorithms.supermimic.world_model import TargetLimiter, WorldModel

    dynamics_mode = getattr(WorldModel, 'DYNAMICS_MODE', 'independent_delta_v1')
    normalization_mode = getattr(WorldModel, 'NORMALIZATION_MODE', 'fixed_input_v1')
    optimizer_mode = getattr(SuperMimicTrainer, 'OPTIMIZER_MODE', 'adam_single_v1')
    sampling_mode = getattr(SuperMimicRunner, 'SAMPLING_MODE', 'shared_batch_transitions_v1')
    if resume is not None and resume['metadata'].get('dynamics_mode', 'independent_delta_v1') != dynamics_mode:
        raise ValueError('动力学结构已改变，旧检查点不能直接续训；请从头训练新结构')
    if resume is not None and resume['metadata'].get('normalization_mode', 'fixed_input_v1') != normalization_mode:
        raise ValueError('世界模型归一化结构已改变，旧检查点不能直接续训')
    if resume is not None and resume['metadata'].get('optimizer_mode', 'adam_single_v1') != optimizer_mode:
        raise ValueError('优化器分组已改变，旧检查点不能直接续训')
    if resume is not None and resume['metadata'].get('sampling_mode', 'shared_batch_transitions_v1') != sampling_mode:
        raise ValueError('采样窗口或批量约定已改变，旧检查点不能直接续训')
    env.reset()
    device, spec, robot = env.device, env.world_state_spec, env.scene['robot']
    context, context_spec = read_sim_context(env)
    term = env.action_manager.get_term('joint_pos')
    ids = torch.arange(spec.num_joints, device=device)[term._joint_ids]
    names = [robot.joint_names[i] for i in ids.tolist()]
    limits = tensor(robot.data.joint_pos_limits)
    calibration = dict(offset=term._offset, scale=term._scale, lower=limits[:, ids, 0], upper=limits[:, ids, 1],
                       max_delta=tensor(robot.data.joint_vel_limits)[:, ids] * term.cfg.target_velocity_fraction * env.step_dt)
    calibration = {k: shared_vector(v, spec.num_actions, k, device) for k, v in calibration.items()}
    expected = dict(state_spec=asdict(spec), context_spec=asdict(context_spec), joint_names=list(robot.joint_names),
                    control_joint_names=names, step_dt=env.step_dt, physics_dt=env.cfg.sim.dt, decimation=env.cfg.decimation)
    if resume is None and args.world_model_scratch:
        initial = None
        hidden_dims = tuple(args.world_hidden_dims)
    elif resume is None:
        initial = read_world_checkpoint(args.world_model_init, expected, calibration, context.shape[-1], device)
        hidden_dims = tuple(initial['hidden_dims'])
    else:
        if resume.get('kind') != 'supermimic_getup_training':
            raise ValueError('resume 必须指向本训练入口保存的 checkpoint')
        if (resume['metadata'].get('transition_schema_version') != 4
                or resume['metadata'].get('contact_mode') != 'geometry_proxy'):
            raise ValueError('旧检查点使用辅助力或接触预测结构，请使用新配置重新训练')
        if resume['metadata'].get('actor_action_mode') != 'bounded_joint_range_v1':
            raise ValueError('旧检查点使用无界 actor，请按修正后的动作与奖励配置从头训练')
        for key, value in expected.items():
            if resume['metadata'].get(key) != value:
                raise ValueError(f'恢复训练的元信息不兼容：{key}')
        torch.testing.assert_close(resume['physics_context'].to(device), context, atol=1e-6, rtol=1e-6)
        hidden_dims = tuple(resume['metadata']['world_hidden_dims'])
    proprio = Proprioception(spec, ids, tensor(robot.data.default_joint_pos)[0, ids],
                            tensor(robot.data.default_joint_vel)[0, ids], history_length=5).to(device)
    features = DifferentiableAMPFeatures.from_urdf(spec, str(args.urdf.resolve()), list(robot.joint_names), usd_path=X2_ULTRA_USD).to(device)
    reference = MotionDataset(str(args.motion_manifest.resolve()), list(robot.joint_names), names, device, env.step_dt)
    discriminator_cfg = dict(
        learning_rate=args.discriminator_lr,
        gradient_penalty=args.discriminator_gradient_penalty,
        trunk_weight_decay=args.discriminator_trunk_weight_decay,
        head_weight_decay=args.discriminator_head_weight_decay,
        normalization_clip=args.discriminator_normalization_clip,
    )
    discriminator = AMPWindowDiscriminator(
        torch.cat(reference.amp_states), num_steps=5, **discriminator_cfg,
    ).to(device)
    style = DifferentiableStyleReward(discriminator).to(device)
    actor_hidden_dims = tuple(args.actor_hidden_dims)
    critic_hidden_dims = tuple(args.critic_hidden_dims)
    if resume is not None:
        for key, widths in (('actor_hidden_dims', actor_hidden_dims),
                            ('critic_hidden_dims', critic_hidden_dims)):
            if tuple(resume['metadata'].get(key, ())) != widths:
                raise ValueError(f'恢复训练的网络结构不兼容：{key}；请使用相同结构或从头训练')
        if resume['metadata'].get('policy_network_mode') != 'layernorm_gelu_v1':
            raise ValueError('恢复训练的策略网络结构不兼容；请使用 LayerNorm/GELU 版本检查点')
    action_lower, action_upper = action_bounds_from_calibration(calibration)
    actor = GoalConditionedActor(proprio.output_dim, 0, spec.num_actions,
                                 action_lower=action_lower, action_upper=action_upper,
                                 hidden_dims=actor_hidden_dims).to(device)
    critic = GoalConditionedCritic(
        spec.dim, context.shape[-1], 0, hidden_dims=critic_hidden_dims,
    ).to(device)
    world_timing = (dict(dt=env.step_dt, decimation=env.cfg.decimation)
                    if dynamics_mode == 'gencoord_integrated_v1' else {})
    world = WorldModel(spec, context.shape[-1], TargetLimiter(**calibration).to(device),
                       hidden_dims, **world_timing).to(device)
    if resume is None and initial is not None:
        if initial.get('normalization_mode', initial['metadata'].get('normalization_mode', 'fixed_input_v1')) != normalization_mode:
            raise ValueError('预训练模型归一化结构不匹配；请使用对应结构的检查点，或选择 --world-model-scratch')
        if initial['model']['input_mean'].shape != world.input_mean.shape:
            raise ValueError('预训练世界模型输入维度不匹配；请使用新输入结构的检查点，或选择 --world-model-scratch')
        world.load_state_dict(initial['model'], strict=True)
    task_cfg = GetupTaskCfg(**env.cfg.getup_task)
    contact_geometry = build_contact_geometry(spec, args.urdf, X2_ULTRA_USD, list(robot.joint_names)).to(device)
    geometry_contract = contact_geometry_contract(contact_geometry)
    task = GetupTask(spec, limits[0], task_cfg, features.kinematics, ids,
                     contact_geometry=contact_geometry).to(device)
    loss_scales = {name: getattr(args, name) for name in (
        'nominal_error_root_pose_pos', 'nominal_error_root_pose_ori',
        'nominal_error_joint_pos', 'nominal_error_joint_vel',
    ) if name in TrainerCfg.__dataclass_fields__}
    trainer_cfg = TrainerCfg(horizon=args.imagination_horizon, world_learning_rate=args.world_lr,
                             actor_learning_rate=args.actor_lr, critic_learning_rate=args.critic_lr,
                             gamma=args.gamma, lam=args.lam, gradient_norm=args.gradient_norm,
                             actor_gradient_value=args.actor_gradient_value,
                             critic_epochs=args.critic_epochs, critic_minibatches=args.critic_minibatches,
                             **loss_scales)
    modules = dict(actor=actor, critic=critic, world_model=world, proprioception=proprio,
                   amp_features=features, style=style, task=task)
    if resume is not None:
        validate_geometry_checkpoint(resume, geometry_contract)
        saved_discriminator_cfg = resume['metadata'].get('discriminator_cfg')
        if saved_discriminator_cfg is not None and saved_discriminator_cfg != discriminator_cfg:
            raise ValueError('恢复训练的判别器配置不一致')
        for key, value in [('trainer_cfg', asdict(trainer_cfg)), ('task_cfg', asdict(task_cfg))]:
            if resume['metadata'][key] != value:
                raise ValueError(f'恢复训练配置不一致：{key}')
        for name, module in modules.items():
            if name not in resume['modules']:
                raise ValueError(f'检查点缺少模块 {name}，不能直接恢复这一版本的训练')
            module.load_state_dict(resume['modules'][name], strict=True)
    trainer = SuperMimicTrainer(
        actor, critic, world, proprio, features, style, task, trainer_cfg,
    )
    optimizers = collect_optimizers(trainer, discriminator)
    if resume is not None:
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(resume['optimizers'][name])
    warmup_complete = bool(resume.get('warmup_complete', True)) if resume is not None else False
    statistics_required = (resume is not None and warmup_complete) or (resume is None and not args.world_model_scratch)
    if statistics_required and not bool(world.stats_ready):
        raise ValueError('已预热或预训练的世界模型缺少输入统计量')
    memory = read_task_memory(env)
    extra_fields = extend_task_memory_fields(transition_extra_fields(spec.dim), memory, env.num_envs)
    replay = SequenceReplayBuffer(env.num_envs, args.replay_steps, spec.dim, spec.num_actions,
                                  context, device=device, command_dim=0,
                                  extra_fields=extra_fields)
    runner = SuperMimicRunner(env, trainer, replay, None, action_noise_std=args.action_noise)
    if resume is not None:
        runner.total_env_steps = int(resume['total_env_steps'])
    offline = (OfflineWorldReplay(args.offline_world_data, args.offline_world_split, expected, calibration, device)
               if args.offline_world_fraction > 0 else None)
    reset_manifest = Path(env.cfg.events.reset_from_reference.params['manifest_file'])
    reset_reference = MotionDataset(str(reset_manifest.resolve()), list(robot.joint_names), names, device, env.step_dt)
    metadata = dict(**expected, world_hidden_dims=hidden_dims, trainer_cfg=asdict(trainer_cfg), task_cfg=asdict(task_cfg),
                     task_name='getup', goal_dim=0, num_envs=env.num_envs, usd=str(X2_ULTRA_USD),
                     action_noise=args.action_noise, world_horizons=list(args.world_horizons),
                     offline_world_fraction=args.offline_world_fraction, reset_manifest=str(reset_manifest.resolve()),
                     birth_probabilities=list(env.cfg.events.reset_from_reference.params['birth_probabilities']))
    metadata['initialization_mode'] = (resume['metadata'].get('initialization_mode', 'pretrained') if resume is not None
                                       else 'scratch' if args.world_model_scratch else 'pretrained')
    metadata['discriminator_cfg'] = discriminator_cfg
    metadata['transition_schema_version'] = getattr(env, 'transition_schema_version', 2)
    metadata['contact_mode'] = 'geometry_proxy'
    metadata['actor_hidden_dims'] = list(actor_hidden_dims)
    metadata['critic_hidden_dims'] = list(critic_hidden_dims)
    metadata['policy_network_mode'] = 'layernorm_gelu_v1'
    metadata['actor_action_mode'] = 'bounded_joint_range_v1'
    metadata['dynamics_mode'] = dynamics_mode
    metadata['normalization_mode'] = normalization_mode
    metadata['optimizer_mode'] = optimizer_mode
    metadata['sampling_mode'] = sampling_mode
    metadata['world_window_unit'] = 'state_frames'
    metadata['contact_geometry_contract'] = geometry_contract
    metadata['task_memory_spec'] = {name: dict(shape=list(value.shape[1:]), dtype=str(value.dtype))
                                  for name, value in memory.items()}
    return GetupRuntime(env=env, trainer=trainer, runner=runner, replay=replay, offline=offline,
                            warmup_complete=warmup_complete,
                            modules=modules, optimizers=optimizers, context=context, metadata=metadata,
                            reset_reference=reset_reference, reference=reference, control_ids=ids,
                            all_reference_paths=set(reference.paths + reset_reference.paths))


def training_env_cfg(cfg):
    """按实验配置创建 Isaac 环境配置；不启动仿真。"""
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_getup_env_cfg import (
        X2SuperMimicGetupEnvCfg,
    )

    env_cfg = X2SuperMimicGetupEnvCfg()
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    env_cfg.sim.device = cfg.device
    return env_cfg


@contextlib.contextmanager
def training_environment(cfg, env_cfg):
    """在已启动的仿真中创建环境并加载检查点；退出时关闭环境。"""
    import random

    import numpy as np
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_supermimic_env import (
        X2SuperMimicEnv,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    checkpoint = (
        torch.load(cfg.resume, map_location=cfg.device, weights_only=False)
        if cfg.resume is not None
        else None
    )
    env = X2SuperMimicEnv(env_cfg)
    try:
        yield env, checkpoint
    finally:
        env.close()


def run_training():
    """解析配置、启动仿真、构建运行时并执行起身实验。"""
    import sys

    from isaaclab_tasks.utils import (
        add_launcher_args,
        launch_simulation,
        setup_preset_cli,
    )

    from getup_config import parse_config
    from getup_training import GetupTrainingRun, file_hashes, semantic_file_hashes

    cfg, launcher = parse_config(
        launcher_hooks=(add_launcher_args, setup_preset_cli)
    )
    cfg.validate()
    cfg.prepare_output()

    # 参数已解析完毕；清空 argv，避免仿真启动阶段重复解析。
    sys.argv = [sys.argv[0]]

    source_before = file_hashes((PROJECT / 'source').rglob('*.py'))
    source_semantic_before = semantic_file_hashes((PROJECT / 'source').rglob('*.py'))
    env_cfg = training_env_cfg(cfg)

    with launch_simulation(env_cfg, launcher):
        with training_environment(cfg, env_cfg) as (env, checkpoint):
            runtime = build_runtime(env, cfg, checkpoint)
            GetupTrainingRun(
                runtime, cfg, checkpoint, source_before, source_semantic_before,
            ).run()
