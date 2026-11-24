"""SuperMimic + AMP 的网络、采样和训练配置。"""

from isaaclab.utils.configclass import configclass


@configclass
class SuperMimicWorldModelCfg:
    """世界模型结构与监督训练。"""

    hidden_dims: tuple[int, ...] = (
        2048,
        2048,
        2048,
    )

    # 每次模型更新采样的窗口数。
    batch_size: int = 2048
    # 连续状态帧数；8 帧对应 7 条监督转移。
    # 多个长度时，按联合训练轮次循环选择。
    horizons: tuple[int, ...] = (8,)

    updates_per_iteration: int = 10
    learning_rate: float = 2e-4

    # 名义误差决定损失权重：weight = 1 / error²。
    # 它们不是任务成功或站稳判据。
    nominal_error_root_pose_pos: float = 0.01
    nominal_error_root_pose_ori: float = 0.02
    nominal_error_joint_pos: float = 0.1
    nominal_error_joint_vel: float = 3.14




@configclass
class SuperMimicPolicyCfg:
    """想象轨迹中的 actor 和 critic 更新。"""

    # 对齐当前核对的官方默认网络宽度。
    actor_hidden_dims: tuple[int, ...] = (
        2048,
        1024,
        512,
        256,
    )

    critic_hidden_dims: tuple[int, ...] = (
        256,
        128,
    )

    # 每次 actor 更新的想象轨迹数量。
    batch_size: int = 4096
    # 用世界模型展开的想象轨迹长度。
    imagination_horizon: int = 32

    actor_learning_rate: float = 2e-4
    critic_learning_rate: float = 2e-4

    gamma: float = 0.99
    lam: float = 0.95

    # 当前 trainer 对世界模型、actor、critic
    # 共用这个梯度范数裁剪阈值。
    max_grad_norm: float = 1.0

    # actor 完成范数裁剪后，再裁剪梯度元素。
    actor_gradient_value: float = 0.1

    critic_epochs: int = 5
    critic_minibatches: int = 4


@configclass
class SuperMimicAMPCfg:
    """AMP 判别器的独立优化配置。"""

    # AMP 判别器单独使用的批量。
    batch_size: int = 128
    updates_per_iteration: int = 4

    # 与当前 AMPWindowDiscriminator 默认值一致。
    learning_rate: float = 1e-4
    gradient_penalty: float = 10.0

    trunk_weight_decay: float = 1e-3
    head_weight_decay: float = 1e-1

    normalization_clip: float = 10.0


@configclass
class SuperMimicWarmupCfg:
    """联合训练前，只更新世界模型的预热阶段。"""

    # 每个环境先采集的真实控制步数。
    collection_steps: int = 512

    # 世界模型监督更新次数，不计入联合训练轮数。
    updates: int = 500

    # 每隔这些更新，补采一批真实数据。
    # 补采步数使用 runner.collection_steps。
    collect_interval: int = 50


@configclass
class SuperMimicEvaluationCfg:
    """固定出生状态的真实物理评估。"""

    interval: int = 50

    # 仿真时间，不是现实时间。
    duration_seconds: float = 20.0

    seed: int = 4242


@configclass
class SuperMimicRunnerCfg:
    """采样、回放以及世界模型和策略的联合训练。"""

    experiment_name: str = "supermimic"
    run_name: str = "default"

    seed: int = 43
    max_iterations: int = 200

    # 每轮联合训练，每个环境采集的真实控制步数。
    collection_steps: int = 32

    # 每个环境最多保存的转移步数。
    replay_steps: int = 2048

    # 只用于真实采样的高斯探索噪声。
    action_noise: float = 0.1

    # scratch：随机初始化世界模型。
    # pretrained：加载监督训练的世界模型。
    world_model_initialization: str = "scratch"

    # pretrained 模式下填写检查点路径。
    world_model_checkpoint: str | None = None

    # 世界模型监督批次中的离线数据比例。
    # 为零时，不读取下面两个离线文件。
    offline_world_fraction: float = 0.0
    offline_world_data: str | None = None
    offline_world_split: str | None = None

    world_model: SuperMimicWorldModelCfg = (
        SuperMimicWorldModelCfg()
    )

    policy: SuperMimicPolicyCfg = (
        SuperMimicPolicyCfg()
    )

    amp: SuperMimicAMPCfg = (
        SuperMimicAMPCfg()
    )

    warmup: SuperMimicWarmupCfg = (
        SuperMimicWarmupCfg()
    )

    evaluation: SuperMimicEvaluationCfg = (
        SuperMimicEvaluationCfg()
    )

    checkpoint_interval: int = 50
    log_interval: int = 10


@configclass
class X2SuperMimicGetupRunnerCfg(
    SuperMimicRunnerCfg
):
    """X2 多动作倒地起身的具体训练配方。"""

    experiment_name: str = "x2_supermimic_getup"
    run_name: str = (
        "scratch_wm8_actor32_obsnorm_target_v2"
    )