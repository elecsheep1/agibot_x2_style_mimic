# SuperMimic + AMP 起身训练

在项目内层目录、IsaacLab Python 环境中运行：

```bash
python scripts/supermimic/train_getup.py
```

默认算法参数读取 `tasks/manager_based/x2_style_mimic/agents/x2_supermimic_cfg.py` 中的 `X2SuperMimicGetupRunnerCfg`。`configs/getup.toml` 不再自动加载。
16 个环境、世界模型 3×512、8 步监督、32 步 BPTT、512 步初始采样、500 次世界模型预热、
200 轮联合训练、每 50 轮评估并保存，每次评估最多 20 秒仿真时间。不读取旧离线转移数据，AMP 仍使用参考动作。

先查看实际生效参数，不启动仿真：

```bash
python scripts/supermimic/train_getup.py --print-config
```

命令行只覆盖需要改变的项目；原有参数名称继续支持：

```bash
python scripts/supermimic/train_getup.py --run-name getup_trial --iterations 400
```

旧 TOML 仍可显式指定为覆盖文件；日常修改算法参数请使用上述 Python cfg：

```bash
python scripts/supermimic/train_getup.py --config scripts/supermimic/configs/my_getup.toml
```

## 文件职责

| 文件 | 负责什么 |
| --- | --- |
| `train_getup.py` | 启动环境，创建运行对象并调用 `run()` |
| `getup_config.py`、`configs/getup.toml` | Python runner cfg 适配、显式旧 TOML 覆盖、命令行与校验 |
| `getup_runtime.py` | 组装现有环境、网络、优化器与回放 |
| `getup_training.py` | 调度预热、联合训练、评估和保存 |
| `getup_logging.py` | JSON、TensorBoard 和实验配置记录 |
| `getup_data.py` | 训练所需的数据采样与模型校验，不依赖诊断脚本 |
| `getup_warmup.py` | 只更新世界模型的预热阶段 |
| `getup_evaluation.py` | 固定出生状态的真实物理评估 |
| `getup_checkpointing.py` | 检查点、随机状态与曲线图 |

算法更新仍调用 `source/` 中已有的 runner/trainer，本次整理不改正式算法实现。
组织方式参考官方 [SuperMimic 训练入口](https://github.com/HybridRobotics/supermimic/blob/main/src/supermimic/scripts/train.py)
的入口与 runner 分离，以及 [InstantAMP](https://github.com/RoboticsResearchHubVN/InstantAMP/blob/main/instantamp/train.py)
集中声明实验参数的方式；没有引入它们的其他算法或超参数。

## 预热与联合训练

预热只更新世界模型，每 50 次更新补采真实数据，actor、critic、判别器保持不变。
观测统计量随真实采样更新。完成后保存 `checkpoint_warmup.pt`，然后开始 200 轮联合训练。
每轮采集 32 个控制步，世界模型更新 10 次、判别器更新 4 次，actor/critic 使用现有 trainer 更新。
预热更新不计入联合训练轮数。

如需加载预训练模型，使用 `--world-model-init 文件路径`，它会覆盖配方的随机初始化选项。
世界模型输入结构变化后只能使用匹配的新模型。
如需混合离线数据，设置正数的 `--offline-world-fraction` 并提供对应数据和回合划分报告；留出回合不参与训练。

## 输出和恢复

默认输出在 `logs/supermimic/train_getup/时间_运行名/`。
保存最终生效的 `arguments.json`、TensorBoard、`metrics.jsonl`、评估 JSON、曲线图和完整检查点。
日志分别使用 `warmup/`、`train/` 和 `eval/`；联合训练终端日志额外显示动作 RMS。

```bash
tensorboard --logdir logs/supermimic/train_getup
```

恢复时使用 `--resume 检查点路径 --iterations 目标总轮数`。
入口自动读取检查点旁边上次保存的 `arguments.json`，无需重新填写所有超参数；也兼容旧入口保存的参数。
显式命令行仍有最高优先级。若显式指定 `--config`，则使用该配方而不是自动恢复原配置。
不允许覆盖旧输出目录。已预热模型只重建 replay，不重复额外预热。
检查点不恢复物理场景或 replay，因此不保证与不中断训练逐位一致。

16 个环境的评估覆盖 8 条动作首帧和固定 RSI，最多统计每环境首回合。
评估不写入训练 replay，不更新网络和观测统计量。
几何站稳没有双脚接触检查，不能替代原任务完整验收。
短训仅验证流程；本次脚本整理没有加入 actor 动作限幅，也不代表已经解决策略学习问题。
