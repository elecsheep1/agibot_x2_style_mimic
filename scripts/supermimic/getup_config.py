"""起身实验配方与命令行；不导入 Isaac Sim，不包含训练循环。"""
import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import json
import math
from pathlib import Path
import re
import runpy
import sys
import tomllib
from typing import get_args, get_origin, get_type_hints
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT / 'source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic/agents/x2_supermimic_cfg.py'


@dataclass
class GetupTrainConfig:
    # 实验和环境。
    experiment_name: str = 'x2_supermimic_getup'
    run_name: str = 'getup_scratch_wm8_actor32_obsnorm_target_v2'
    num_envs: int = 16
    seed: int = 43
    device: str = 'cuda:0'
    viz: str = 'none'
    output: Path | None = None

    # 初始化方式；默认配方在 TOML 中指定。
    world_model_scratch: bool = False
    world_model_init: Path | None = None
    resume: Path | None = None

    # 数据。离线比例为零时不读取离线文件。
    motion_manifest: Path = PROJECT / 'data/motions/prone_getup2/motions.json'
    urdf: Path = Path('/home/panda/robotics_ws/agibot_model/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra.urdf')
    offline_world_data: Path = PROJECT / 'logs/supermimic/policy_noise005_v1/transitions.pt'
    offline_world_split: Path = PROJECT / 'logs/supermimic/fit_noise005_capacity512_v1/report.json'
    offline_world_fraction: float = 0.5

    # 真实采样、回放与联合训练。
    iterations: int = 200
    collection_steps: int = 32
    replay_steps: int = 2048
    batch_size: int = 128
    action_noise: float = 0.1

    # 世界模型与策略优化。
    world_hidden_dims: tuple[int, ...] = (512, 512, 512)
    world_horizons: tuple[int, ...] = (8,)
    imagination_horizon: int = 32
    world_updates: int = 10
    discriminator_updates: int = 4
    world_lr: float = 2e-4
    actor_lr: float = 2e-4
    critic_lr: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    gradient_norm: float = 1.0
    actor_gradient_value: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    discriminator_lr: float = 1e-4
    discriminator_gradient_penalty: float = 10.0
    discriminator_trunk_weight_decay: float = 1e-3
    discriminator_head_weight_decay: float = 1e-1
    discriminator_normalization_clip: float = 10.0

    # 只更新世界模型的预热阶段。
    warmup_steps: int = 128
    warmup_updates: int = 50
    warmup_collect_interval: int = 50

    # 评估、保存和日志。
    eval_interval: int = 50
    eval_seconds: float = 20.0
    eval_seed: int = 4242
    checkpoint_interval: int = 50
    log_interval: int = 10

    @classmethod
    def from_mapping(cls, values):
        hints = get_type_hints(cls)
        unknown = set(values) - set(hints)
        if unknown:
            raise ValueError(f'配置中有未知字段：{", ".join(sorted(unknown))}')
        converted = {}
        for name, value in values.items():
            annotation = hints[name]
            if value is None and type(None) in get_args(annotation):
                converted[name] = None
            elif annotation is Path or Path in get_args(annotation):
                if not isinstance(value, (str, Path)):
                    raise ValueError(f'{name} 必须是路径字符串')
                converted[name] = Path(value)
            elif get_origin(annotation) is tuple:
                if not isinstance(value, (list, tuple)) or not all(type(v) is int for v in value):
                    raise ValueError(f'{name} 必须为整数列表')
                converted[name] = tuple(value)
            elif annotation is float and type(value) in (int, float):
                converted[name] = float(value)
            elif type(value) is annotation:
                converted[name] = value
            else:
                raise ValueError(f'{name} 类型不正确，要求 {annotation.__name__}')
        return cls(**converted)

    def validate(self):
        if sum((self.world_model_scratch, self.world_model_init is not None, self.resume is not None)) != 1:
            raise ValueError('请选择一种初始化方式：随机初始化、预训练世界模型或恢复训练')
        positive = ('num_envs', 'iterations', 'collection_steps', 'batch_size', 'world_updates',
                    'discriminator_updates', 'replay_steps', 'warmup_steps', 'warmup_collect_interval',
                    'eval_interval', 'checkpoint_interval', 'log_interval', 'imagination_horizon')
        if any(getattr(self, key) < 1 for key in positive) or self.warmup_updates < 0:
            raise ValueError('采样大小、轮数和间隔必须为正，预热更新数可以为零')
        if self.world_model_scratch and self.warmup_updates < 1:
            raise ValueError('随机初始化世界模型必须先执行监督预热')
        if not self.world_hidden_dims or min(self.world_hidden_dims) < 1:
            raise ValueError('世界模型隐藏层宽度必须为正')
        if not self.world_horizons or any(not 1 <= h <= 32 for h in self.world_horizons) or self.imagination_horizon > 32:
            raise ValueError('世界模型和想象预测长度必须在 [1,32] 内')
        required = max(max(self.world_horizons), 5) + 1
        if self.warmup_steps < required or self.replay_steps < max(required, self.warmup_steps):
            raise ValueError('预热步数必须覆盖最长窗口，replay_steps 必须不少于预热步数')
        for name in ('actor_lr', 'critic_lr', 'world_lr', 'eval_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} 必须为正有限数')
        if not 0 <= self.gamma < 1 or not 0 <= self.lam <= 1:
            raise ValueError('gamma 必须在 [0,1)，lam 必须在 [0,1]')
        for name in ('gradient_norm', 'actor_gradient_value', 'discriminator_lr', 'discriminator_normalization_clip'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} 必须为正有限数')
        for name in ('discriminator_gradient_penalty', 'discriminator_trunk_weight_decay', 'discriminator_head_weight_decay'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} 必须为非负有限数')
        if min(self.critic_epochs, self.critic_minibatches) < 1:
            raise ValueError('critic 更新轮数和小批次数必须为正整数')
        if not math.isfinite(self.action_noise) or self.action_noise < 0:
            raise ValueError('动作噪声必须为非负有限数')
        if not math.isfinite(self.offline_world_fraction) or not 0 <= self.offline_world_fraction <= 1:
            raise ValueError('离线数据比例必须在 [0,1] 内')
        names = ['motion_manifest', 'urdf', 'world_model_init', 'resume']
        if self.offline_world_fraction > 0:
            names += ['offline_world_data', 'offline_world_split']
        for name in names:
            value = getattr(self, name)
            if value is not None and not value.is_file():
                raise ValueError(f'{name} 文件不存在：{value}')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', self.run_name):
            raise ValueError('run-name 只允许字母、数字、下划线、点和连字符')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', self.experiment_name) or self.experiment_name in ('.', '..'):
            raise ValueError('experiment-name 必须为合法的单级目录名称')

    def prepare_output(self):
        if self.output is None:
            stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S')
            self.output = PROJECT / 'logs/supermimic' / self.experiment_name / f'{stamp}_{self.run_name}'
        if self.output.exists():
            raise ValueError('输出目录已存在，请使用新目录；恢复训练也写入新目录')

    def to_dict(self):
        # 统一使用 JSON 可保存的列表和路径表示，亦兼容旧 arguments.json。
        return json.loads(json.dumps(asdict(self), default=str))


def runner_defaults():
    # 按文件加载，避免触发任务包注册及仿真导入。
    runner = runpy.run_path(str(DEFAULT_CONFIG))['X2SuperMimicGetupRunnerCfg']()
    mode = runner.world_model_initialization
    if mode not in ('scratch', 'pretrained'):
        raise ValueError('世界模型初始化方式必须为 scratch 或 pretrained')
    values = dict(
        experiment_name=runner.experiment_name,
        run_name=runner.run_name, seed=runner.seed, iterations=runner.max_iterations,
        collection_steps=runner.collection_steps, replay_steps=runner.replay_steps,
        batch_size=runner.batch_size, action_noise=runner.action_noise,
        world_model_scratch=mode == 'scratch',
        world_model_init=runner.world_model_checkpoint if mode == 'pretrained' else None,
        offline_world_fraction=runner.offline_world_fraction,
    )
    for name in ('offline_world_data', 'offline_world_split'):
        if getattr(runner, name) is not None:
            values[name] = getattr(runner, name)
    mapping = {
        'world_model': {'hidden_dims': 'world_hidden_dims', 'horizons': 'world_horizons',
                        'updates_per_iteration': 'world_updates', 'learning_rate': 'world_lr'},
        'policy': {'imagination_horizon': 'imagination_horizon', 'actor_learning_rate': 'actor_lr',
                   'critic_learning_rate': 'critic_lr', 'gamma': 'gamma', 'lam': 'lam',
                   'max_grad_norm': 'gradient_norm', 'actor_gradient_value': 'actor_gradient_value',
                   'critic_epochs': 'critic_epochs', 'critic_minibatches': 'critic_minibatches'},
        'amp': {'updates_per_iteration': 'discriminator_updates', 'learning_rate': 'discriminator_lr',
                'gradient_penalty': 'discriminator_gradient_penalty',
                'trunk_weight_decay': 'discriminator_trunk_weight_decay',
                'head_weight_decay': 'discriminator_head_weight_decay',
                'normalization_clip': 'discriminator_normalization_clip'},
        'warmup': {'collection_steps': 'warmup_steps', 'updates': 'warmup_updates',
                   'collect_interval': 'warmup_collect_interval'},
        'evaluation': {'interval': 'eval_interval', 'duration_seconds': 'eval_seconds', 'seed': 'eval_seed'},
    }
    for group, fields_map in mapping.items():
        for source, destination in fields_map.items():
            values[destination] = getattr(getattr(runner, group), source)
    values.update(checkpoint_interval=runner.checkpoint_interval, log_interval=runner.log_interval)
    return GetupTrainConfig.from_mapping(values)


def read_config(path):
    cfg = runner_defaults()
    if Path(path).resolve() == DEFAULT_CONFIG.resolve():
        return cfg
    # 旧 TOML 仅在显式传入时作为覆盖，不再作为默认配方。
    with Path(path).open('rb') as stream:
        values = cfg.to_dict()
        values.update(tomllib.load(stream))
    return GetupTrainConfig.from_mapping(values)


def parse_config(argv=None, launcher_hooks=None):
    """配置文件 < 检查点旁的原始配置（恢复时）< 显式命令行覆盖。"""
    argv = sys.argv[1:] if argv is None else list(argv)
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config', type=Path)
    preliminary.add_argument('--resume', type=Path)
    selected, _ = preliminary.parse_known_args(argv)
    cfg = read_config(selected.config or DEFAULT_CONFIG)
    if selected.resume is not None and selected.config is None:
        # 恢复自动沿用上次实际参数，用户只需覆盖目标总轮数。
        arguments = selected.resume.parent / 'arguments.json'
        if arguments.is_file():
            values = json.loads(arguments.read_text())
            allowed = {f.name for f in fields(GetupTrainConfig)}
            for name in ('device', 'viz'):
                if values.get(name) is None:
                    values.pop(name, None)
            cfg = GetupTrainConfig.from_mapping({k: v for k, v in values.items() if k in allowed})
            cfg.output = None

    parser = argparse.ArgumentParser(description='SuperMimic + AMP 起身训练：实验配方、预热与定期真实评估。')
    parser.add_argument('--config', type=Path, help='可选旧 TOML 覆盖；默认读取 agents/x2_supermimic_cfg.py')
    parser.add_argument('--print-config', action='store_true', help='打印最终生效配置，不启动仿真')
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument('--world-model-scratch', action='store_true', default=argparse.SUPPRESS)
    initialization.add_argument('--world-model-init', type=Path, default=argparse.SUPPRESS)
    initialization.add_argument('--resume', type=Path, default=argparse.SUPPRESS)
    hints = get_type_hints(GetupTrainConfig)
    overrides = parser.add_argument_group('实验参数覆盖（未指定时沿用配置文件）')
    skip = {'world_model_scratch', 'world_model_init', 'resume', 'device', 'viz'}
    for item in fields(GetupTrainConfig):
        if item.name in skip:
            continue
        annotation = hints[item.name]
        options = dict(default=argparse.SUPPRESS)
        if annotation is Path or Path in get_args(annotation):
            options['type'] = Path
        elif get_origin(annotation) is tuple:
            options.update(type=int, nargs='+')
        else:
            options['type'] = annotation
        overrides.add_argument('--' + item.name.replace('_', '-'), **options)
    if launcher_hooks is None:
        parser.add_argument('--device', default=cfg.device)
        parser.add_argument('--viz', default=cfg.viz)
        arguments = parser.parse_args(argv)
    else:
        add_launcher_args, setup_preset_cli = launcher_hooks
        add_launcher_args(parser)
        parser.set_defaults(device=cfg.device, viz=cfg.viz)
        previous = sys.argv
        try:
            sys.argv = [previous[0]] + argv
            arguments, rest = setup_preset_cli(parser)
        finally:
            sys.argv = previous
        if rest:
            parser.error(f'未识别参数：{rest}')
    values = cfg.to_dict()
    explicit = vars(arguments)
    allowed = set(hints)
    values.update({k: v for k, v in explicit.items() if k in allowed})
    for mode in ('world_model_scratch', 'world_model_init', 'resume'):
        if mode in explicit:
            values.update(world_model_scratch=False, world_model_init=None, resume=None)
            values[mode] = explicit[mode]
            break
    try:
        cfg = GetupTrainConfig.from_mapping(values)
    except ValueError as error:
        parser.error(str(error))
    return cfg, arguments
