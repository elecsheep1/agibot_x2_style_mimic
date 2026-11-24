# X2 SuperMimic 起身算法工作交接

更新日期：2026 年 11 月 24 日，Asia/Shanghai。交接对象为继续维护代码和开展实验的开发者。

当前目标是用纯 SuperMimic 加 AMP，通过世界模型上的 BPTT 学习倒地起身，不引入 PPO 策略更新。最新实现已通过 174 项 CPU 测试、GPU 短程训练及只读预测诊断。头部距离惩罚已生效，实际头部接触减少，但尚未观察到起身能力改善。当前优先任务是固定策略补训世界模型，验证留出数据上的长步预测能否改善，而不是继续叠加奖励或直接长训。

## 工作环境和协作约定

- 外层工作区：`/home/panda/robotics_ws/x2_style_mimic`。
- 实际项目与 Git 仓库：`/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic`。本文命令均在此目录运行。
- Python：`/home/panda/miniconda3/envs/env_isaaclab_6_0/bin/python`，用户环境名 `env_isaaclab_6_0`。
- IsaacLab：`/home/panda/robotics_ws/isaaclab_3`。GPU 仿真需要能访问 CUDA 和 Isaac Sim 缓存的执行环境；受限工具环境曾出现无 GPU，但授权后的 GPU 环境可正常运行。
- 用户自己修改正式 `source`；AI 提供完整代码及函数、类、上下文锚点，不依赖易变化的行号。脚本、测试、实验配置可以由 AI 直接修改和测试。
- 不擅自引入 PPO、辅助力、速度推扰、自碰撞预测网络，不擅自修改原 AMP 算法。
- 训练参数放在已有 cfg 或实验 TOML 中。命令直接给出，不用环境变量拼接路径。
- 用户要求中文注释、中文报错；优先沿用项目结构，避免新增不必要的接口和文件。
- Git 操作需用户明确要求。交接时 HEAD 为 `18b375d Update SuperMimic training alignment and save fallen-state additions`；当前工作区还有未提交修改及新增脚本，本次没有提交或推送。
- 并行存在 AMP 实验方向，已有 [AMP 交接文档](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/docs/amp_experiments_handoff_20261124.md)。不要覆盖其内容，也不要把不同方向的实验结论混用。

## 关键代码位置

| 职责 | 文件 |
|---|---|
| 完整物理状态与打包 | [state.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/algorithms/supermimic/state.py) |
| 广义坐标表示及积分 | [representations.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/algorithms/supermimic/representations.py) |
| 世界模型与监督损失 | [world_model.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/algorithms/supermimic/world_model.py) |
| Actor 和 critic | [actor_critic.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/algorithms/supermimic/actor_critic.py) |
| 想象轨迹与 BPTT 更新 | [rollout.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/algorithms/supermimic/rollout.py)、[trainer.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/algorithms/supermimic/trainer.py)、[objectives.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/algorithms/supermimic/objectives.py) |
| 真实采集与序列回放 | [runner.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/algorithms/supermimic/runner.py)、[buffer.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/algorithms/supermimic/buffer.py) |
| 可微任务奖励与几何接触 | [heads.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/algorithms/supermimic/heads.py)、[contact_geometry.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/algorithms/supermimic/contact_geometry.py) |
| 网络及训练 cfg | [x2_supermimic_cfg.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic/agents/x2_supermimic_cfg.py) |
| 真实环境与任务 cfg | [x2_supermimic_env.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic/x2_supermimic_env.py)、[x2_supermimic_env_cfg.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic/x2_supermimic_env_cfg.py)、[x2_supermimic_getup_env_cfg.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic/x2_supermimic_getup_env_cfg.py) |
| 共同父类与地面接触重置补丁 | [x2_getup_env.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic/x2_getup_env.py)、[self_contact.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic/mdp/self_contact.py) |

脚本入口为 [train_getup.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/scripts/supermimic/train_getup.py)。[getup_runtime.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/scripts/supermimic/getup_runtime.py) 负责组装和恢复组件，[getup_training.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/scripts/supermimic/getup_training.py) 负责实验调度。不要继续将算法逻辑堆入入口脚本。

## 当前算法和接口约定

### 状态与动力学

完整状态为 133 维：根位姿 7、世界系根速度 6、31 个关节角、31 个关节速度、29 个前一原始动作、29 个前一实际 PD 目标。四元数统一 XYZW，根位置相对环境原点。

广义坐标表示为 72 维：根高度、身体坐标系重力方向、根线速度、根角速度、全部关节角和速度。预测输出为 37 维速度增量。通过积分生成下一状态，而非独立预测每个姿态分量。控制周期 0.02 秒、物理 decimation 为 10，积分使用相应子步系数 0.55。

世界模型输入为 72 维表示、29 维受限实际目标、472 维物理 context，共 573 维。状态和目标归一化统计量只由真实采样显式更新；forward 不更新统计量。context 当前采用首次真实数据的静态标准化，不是完全复刻官方的随机化范围编码。

前一实际目标参与目标变化限速，前一原始动作参与 action rate 奖励。不能为减少维度直接删除这些控制器历史。

### 模型和优化

- 世界模型隐藏层为 2048、2048、2048；用户明确保留末层权重标准差 1e-3 和零偏置初始化。
- Actor 隐藏层为 2048、1024、512、256；critic 为 256、128。隐藏层使用 Linear、LayerNorm、GELU。宽度由 policy cfg 传入，actor 末层小初始化、critic 末层零初始化仍保留。
- Actor 观测为 5 帧、每帧 93 维，共 465 维。输入归一化只使用真实采样更新。
- Actor 输出经有界 tanh 映射到动作标定范围。范围可能大于 1，不能仅根据动作 RMS 超过 1 判断越界。
- Sequential 的中间 Linear 权重使用 Muon；首尾 Linear、偏置和 LayerNorm 参数使用 Adam。Actor 的 Adam betas 为 0.9、0.95。
- Actor 更新冻结世界模型、critic、判别器等模块的参数，但保留它们对输入的梯度；BPTT 的 backward 位于 `SuperMimicTrainer.update_actor()`。
- gamma 为 0.99，lambda 为 0.95；critic 每批想象轨迹拟合 5 轮、每轮 4 个小批次。同一批轨迹只打乱一次，随后复用分组。
- Actor 末端价值和 critic 的 lambda-return 目标只裁上限，不裁掉负价值。上限由 `task.max_step_reward / (1 - gamma)` 计算；当前正奖励权重 10 加 10、dt 0.02，对应价值上限 40。

监督损失按所有通道平均，四组名义误差为根位置 0.01 m、根旋转 0.02、关节角 0.1 rad、关节速度 3.14 rad/s。根旋转用旋转矩阵弦距离；这些名义误差是损失标度，不是成功判据。

### 时间和采样

训练配置 `world_horizons=(8,)` 表示 8 个状态帧，即 7 条监督转移。不要在不同层重复减一。底层 `sample_windows(horizon=...)` 的 horizon 仍表示转移数。

预测报告中的 8 步表示向前 8 条转移，即 0.16 秒；它与训练的 8 帧窗口不是同一单位。Actor 想象长度 32 表示 32 条转移，即 0.64 秒。

世界模型、策略和 AMP 使用独立批量。正式默认分别为 2048、4096、128；最新短程实验三者均缩小为 128。大批量的显存和学习效果尚未完成验收。

当前为无外部目标起身任务，goal_dim 为 0。保留 motion_id、frame_id 是为了出生、参考采样和诊断，不代表采用逐帧参考跟踪奖励。

### 官方对照范围

本地核对的官方代码位于 `/tmp/supermimic_official_alignment_20261123`，固定提交 `5fb92297ab4fe3d86629d9f0c635334791bb0d5f`。临时目录不保证长期保留。

已对齐广义坐标积分、主要监督损失标度、状态和动作运行归一化、Adam/Muon 分组、独立批量、价值上限、critic 分组复用及隐藏层形式。当前任务仍是 X2 的单任务本体观测加 AMP，未完整复刻官方多任务目标编码器和辅助重建结构。Critic 当前仍使用完整状态及 context；不能写成已完成全部官方架构对齐。也未接入 CUDA Graph。

## 真实环境和想象奖励的边界

SuperMimic 禁用辅助力课程、站立随机速度跳变、自碰撞奖励和专用自接触传感器；机器人资产中的物理自碰撞仍保留。启动时材质、质量、增益随机化仍存在。

Actor 实际优化想象轨迹的 task.reward 加 AMP。真实 RewardManager 回报用于采集、监测和评估，不直接作为 PPO 梯度更新 actor。几何接触代理不表示牛顿力，不应要求其与历史接触力阈值逐帧一致。

当前实际任务配置为 standing_weight 10、style_weight 10、action_rate_weight 0.015、joint_limit_weight 0.5、joint_acc_weight 1e-7、head_contact_weight 2。真实环境还继承动作二阶差分惩罚，但想象任务尚未实现该项。不要只向 GetupTaskCfg 增加 `action_second_difference_weight` 字段：配置曾因此导入报错，仅加字段也不会实现奖励，还需相应历史和回合有效性。

关节加速度存在已知口径差异：真实奖励使用仿真加速度，代理使用相邻控制帧速度差分。前者能反映周期内的部分变化，后者是控制周期平均量。本次没有据回报比例放大加速度权重。

原 PPO 站立评分还含力支撑、手部承重和手脚姿态因素；想象站立评分采用几何近似，并非所有因素完全等价。

## 接触历史缺陷及已完成修复

本地 PhysX 后端的 ContactSensor.reset 未清空过滤接触历史 `force_matrix_w_history`。其重置内核只处理当前过滤力及未过滤历史，而地面奖励读取过滤历史最大值，导致旧回合接触可能污染新回合开头。

缺陷源码位于 `/home/panda/robotics_ws/isaaclab_3/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py` 和同目录 `kernels.py`。未修改 IsaacLab 安装。

项目补丁是 self_contact.py 中的模块级 `reset_ground_contact_sensors()`，不属于 SelfContactSensor 类。它清理头部、双脚、双手六个地面传感器，只作用于 env_ids。顺序为正常 reset、读取 data 并显式索引清零过滤历史、再次 reset 当前缓冲及时间戳。第二次 reset 是为了处理 data 读取可能引起的懒更新。

共同父类 X2AMPGetUpEnv._reset_idx 在捕获旧 AMP 状态、执行 `super()._reset_idx(env_ids)` 后调用清理函数。SuperMimic 子类不重复调用。旧回合真实状态及统计保留，新回合不沿用过滤历史。

评估会在 env.reset 后再次设置固定 cohort，所以评估脚本还有独立清理调用，这是设置姿态后的必要操作，不应误删为重复代码。

GPU 复验中，第二次评估开头的异常历史接触样本从 4 个降到 0 个，当前接触样本仍为 211 个。共同父类接入后通过 CPU 部分环境重置、空索引及终止捕获顺序测试，并完成 SuperMimic GPU 短程验证。原 PPO 加 AMP 的完整独立验收尚未执行；它已通过继承获得补丁。

旧结果不能因此一概判为无效。污染集中在重置附近，成功仍要求其他条件及连续稳定时间。该缺陷也不同于 AMP 五帧窗口的清空问题。

## 头部几何诊断和距离惩罚

诊断脚本为 [head_diagnostics.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/scripts/supermimic/head_diagnostics.py)，接入 [getup_evaluation.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/scripts/supermimic/getup_evaluation.py)。每个控制步记录 FK 离地距离、实际刚体姿态离地距离、当前力、历史力、阈值标签及代理值。终止步的传感器可能已重置，故接触归因会排除这些样本。

FK 与真实 link 姿态使用同一组 USD 碰撞顶点计算距离：最大距离差约 0.00045 mm，位置最大差约 0.0014 mm。支持头部 FK 映射正确，不证明原始凸包顶点与 PhysX 烘焙碰撞形状完全相同。日志仍存在某些几何子路径的 contact API 警告，不能仅因已有非零力就认定所有传感器路径完全正确。

旧代理在 211 个当前接触样本上均非零，但中位数只有 0.0075；另有 595 个样本代理为正而历史接触力为零。原始几何距离不能精确替代超过 10 N 的力标签。

当前采用有界头部距离惩罚：head_contact_distance 为 0.005 m、head_contact_transition 为 0.010 m、权重仍为 2。令 level 为 `(distance_threshold - clearance) / transition` 裁剪至 0 到 1，再用 smoothstep，两个头部取最大值，不重复相加。

| 头部离地距离 | 惩罚程度 |
|---|---:|
| 大于等于 +5 mm | 0 |
| +2.5 mm | 0.15625 |
| 0 mm | 0.5 |
| -0.25 mm | 0.53747 |
| -2.5 mm | 0.84375 |
| 小于等于 -5 mm | 1 |

靠近地面但力未超阈值的几何惩罚现在属于设计含义，不自动判为传感器误报。真实力惩罚继续作为监测。头部权重没有同步增加。

## 已完成实验和证据

所有目录均位于 `/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/logs/supermimic/x2_supermimic_getup/`。

| 运行目录名 | 用途及结论 |
|---|---|
| 2026-11-24_17-03-41_getup_official_alignment_smoke | 16 环境、批量 32、预热 20 次、联合训练 2 轮，32 步 BPTT、更新及保存跑通 |
| 2026-11-24_17-05-14_getup_official_alignment_resume_smoke | 从上述检查点恢复，完成第 3 轮，网络和优化器状态可恢复 |
| 2026-11-24_19-10-35_getup_official_alignment_short50 | 旧头部代理的 50 轮短程参考结果；非当前配置的严格单变量对照 |
| 2026-11-24_19-56-08_getup_head_contact_diagnostic | 用户运行，揭示代理强度偏差和评估开头历史残留 |
| 2026-11-24_20-05-45_getup_head_geometry_reset_diagnostic | 验证 FK 与实际 link 姿态；仅调用普通传感器 reset 仍有残留 |
| 2026-11-24_20-09-16_getup_head_filteredhistory_diagnostic | 显式清理过滤历史后，开头异常历史接触消失 |
| 2026-11-24_20-26-59_getup_shared_ground_reset_check | 共同 AMP 父类补丁的 SuperMimic GPU 流程验证 |
| 2026-11-24_20-31-44_getup_head_distance_short50 | 最新 50 轮训练，当前优先检查点来源 |
| 2026-11-24_20-41-56_getup_head_distance_prediction_check | 最新检查点的新采样只读预测诊断 |

### 最新短程训练

实验配置为 [getup_alignment_short.toml](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/scripts/supermimic/configs/getup_alignment_short.toml)：32 环境、各类批量 128、每轮采集 32 步、世界模型每轮更新 5 次、判别器 4 次；世界模型预热 200 次，每 25 次补采；策略想象 32 步。共 50 轮，累计训练采集转移 66560 条。

| 指标 | 初始 | 第25轮 | 第50轮 |
|---|---:|---:|---:|
| 站立质量 | 0.017066 | 0.015753 | 0.014509 |
| 真实任务回报 | 0.294527 | 0.593810 | 0.359079 |
| 几何任务重算回报 | -1.179483 | 0.509708 | -0.071939 |
| 真实头部接触累计惩罚 | -0.366250 | -0.021250 | -0.042500 |
| 几何头部累计惩罚 | -1.927234 | -0.158089 | -0.527971 |
| 任务成功率 | 0 | 0 | 0 |

每次真实评估只有 5 秒，不是 20 秒完整回合验收。观察到头部接触减少，尚无起身能力改善。动作 RMS 从第 1 轮约 0.101 增至第 50 轮约 0.514。没有 NaN 或 Inf，运行期间正式 source 的内容未变。

新旧短程实验之间还包含动作正则和重置修复等差异，只有单种子。不能把两次结果差异全部归因为头部距离参数。

最新检查点：[latest.pt](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/logs/supermimic/x2_supermimic_getup/2026-11-24_20-31-44_getup_head_distance_short50/latest.pt)。汇总：[diagnostic_summary.json](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/logs/supermimic/x2_supermimic_getup/2026-11-24_20-31-44_getup_head_distance_short50/diagnostic_summary.json)。

### 最新世界模型只读诊断

新采集 3072 条带探索噪声的真实转移，采样 128 个连续窗口。使用实际记录的同一动作序列递推，不更新网络、优化器或归一化统计量。窗口可能重叠，属于新采样诊断，不是按完整回合划分的独立留出集。

| 预测转移数 | 时间 | 根位置分量 RMSE | 根姿态角 RMSE | 关节角 RMSE |
|---|---:|---:|---:|---:|
| 1 | 0.02 s | 0.074 cm | 0.40 度 | 0.00720 rad |
| 8 | 0.16 s | 2.22 cm | 7.79 度 | 0.07862 rad |
| 32 | 0.64 s | 9.60 cm | 28.42 度 | 0.17858 rad |

模型在根位姿和关节角等指标上优于保持初始状态不变的基线，但 32 步漂移明显。32 步位置与姿态基线误差分别约 16.61 cm、44.13 度，不能把模型描述成完全没有学到动力学。

任务代理的终点单步奖励 MAE 为 0.000589、0.004567、0.011267；32 步站立项 MAE 0.003696、头部项 MAE 0.007200。这些是加权且乘 dt 后的单步奖励误差，不是整条轨迹回报误差；此处不含 AMP 风格奖励。

报告：[prediction_report.json](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/logs/supermimic/x2_supermimic_getup/2026-11-24_20-41-56_getup_head_distance_prediction_check/prediction_report.json)。当前证据支持优先检查长步模型误差，尚不能证明它是唯一瓶颈，也不能证明补训后 actor 一定学会起身。

## 可复现命令

先进入项目目录并激活用户环境。不要在同一输出目录重复写入；脚本会创建带时间戳的新目录。

```bash
cd /home/panda/robotics_ws/x2_style_mimic/x2_style_mimic
python -m unittest discover -s scripts/tests -p 'test_supermimic*.py' -v
```

最后一次完整测试为 174 项通过。包括状态积分、运行归一化、监督损失、优化器分组、配置路由、价值上限、头部距离数值与梯度、部分环境接触清理、终止捕获顺序。部分测试使用模拟对象或 AST 隔离仿真依赖，不能代替全部真实后端验证。

复现最新短程实验：

```bash
python scripts/supermimic/train_getup.py \
  --config scripts/supermimic/configs/getup_alignment_short.toml \
  --run-name getup_head_distance_short50_repeat
```

只读预测诊断，不执行恢复训练：

```bash
python scripts/supermimic/diagnose_getup_model.py \
  --resume logs/supermimic/x2_supermimic_getup/2026-11-24_20-31-44_getup_head_distance_short50/latest.pt \
  --run-name getup_head_distance_prediction_repeat
```

该入口使用 --resume 读取原实验参数和检查点，但不调用训练调度器、不更新优化器。加载仍校验网络结构、task cfg、controller calibration 和接触几何等信息；若正式配置随后改变，不要绕过不兼容检查强行恢复。

小规模流程配置是 [getup_alignment_smoke.toml](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/scripts/supermimic/configs/getup_alignment_smoke.toml)，只跑 2 轮，不能用来评价学习效果。

## 下一步工作

1. 保持当前奖励、网络、物理随机化和 actor 不变。先适配现有 [refit_supermimic_world_fixed_policy.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/scripts/checks/refit_supermimic_world_fixed_policy.py)，不要未经复核直接启动。
2. 该旧脚本的冻结回调仍是 `lambda *args: None`，当前 runner 用关键字参数调用世界模型归一化更新，需支持 `*args, **kwargs`。其训练窗口仍按 8 条转移硬编码；如保持当前训练口径，应由 8 帧转换为 7 条转移，数据池、采样和报告同时修正。
3. 固定 actor、critic、判别器和全部归一化统计量，采集完整回合。按环境与回合 ID 划分 train 和 holdout，不能随机拆重叠窗口。保存数据、划分及固定评估窗口。
4. 在同一留出窗口上记录补训前后 1、8、16、32 步误差，区分起身早期、接触转换与低速后期。同步看代理奖励误差，不只看监督训练损失。
5. 可先做 500 次世界模型更新，在 0、100、250、500 次记录结果。所有策略及判别器参数、输入统计量保持不变，并比较它们的 state_dict。不要用 actor probe 同时更新策略，以免混入变量。
6. 若模型补训改善留出预测，再进行独立的联合训练对照。若仅训练误差下降而留出误差不改善，应继续查模型、数据覆盖和接触转换。想象长度 8、16、32 的课程可以作为后续受控实验，但尚未实现，之前更旧版本的 H8 实验也没有证明单独缩短即可解决问题。
7. 原 PPO 加 AMP 的共同重置补丁还应单独跑完整验收，重点检查自动重置的终止数据、部分环境清理及新回合接触标签。

不要因为 50 轮没有成功就宣称算法不成立，也不要因为训练损失小就认定世界模型可支撑 32 步 BPTT。当前单种子、短评估和新采样窗口的证据边界需要保留。

## 本轮脚本修改及接手时检查事项

新增 [diagnose_getup_model.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/scripts/supermimic/diagnose_getup_model.py)、head_diagnostics.py、短程 TOML 和正式接触重置回归测试。评估 JSON 现包含真实与代理奖励分项、分项差值、完整头部诊断 trace；读取真实 RewardManager 分项时按当前实现乘一次 dt，并校验分项和等于实际奖励。

[audit_supermimic_training.py](/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/scripts/checks/audit_supermimic_training.py) 已补真实采样时的世界模型与本体观测统计冻结，并加入任务代理误差。当前新增奖励诊断按无外部目标起身任务使用，不应未经适配直接复用于其他任务。

下一位接手先执行 git status、读取最新 cfg 和上述两个结果报告，确认用户没有继续修改。已有 AMP 交接文档和实验可能同时更新。模型架构、状态、归一化或奖励 cfg 改动后，旧 full resume 可能不兼容；保持严格加载检查，不靠忽略字段静默继续。
