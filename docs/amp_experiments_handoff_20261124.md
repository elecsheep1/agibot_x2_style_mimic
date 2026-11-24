# X2 起身 AMP 实验工作交接

更新于 2026 年 11 月 24 日，时区 Asia/Shanghai。

用户当前重点是解释和改善 AMP 判别器长期给策略低分的问题。已比较动作变化正则、关节加速度正则和降低 standing 权重，并准备减少判别器更新次数的实验。用户希望根据实际动作和评分共同判断，避免反复进行过细的指标比较。

## 工作环境与协作约定

- 外层工作区为 `/home/panda/robotics_ws/x2_style_mimic`，它本身没有 Git 仓库。
- 实际项目与 Git 仓库为 `/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic`，下文相对路径均基于此目录。
- Python 为 `/home/panda/miniconda3/envs/env_isaaclab_6_0/bin/python`。
- IsaacLab 源码位于 `/home/panda/robotics_ws/isaaclab_3`。
- GPU 仿真和网络操作在受限环境中可能需要工具的 `require_escalated`。本会话的 GPU 验收和远程只读访问均得到用户授权。
- 用户通常自己修改生产代码，希望得到完整代码和明确的插入上下文。脚本和测试可以由 AI 实现。按用户当前指令确定实际修改范围。
- 给修改位置时，提供可搜索的前后代码和完整替换段落。文件经过编辑后，旧行号会变化。
- 训练指令直接写完整参数，用户偏好不用环境变量拼接路径。
- 用户另外在推进 SuperMimic，当前工作区有该方向的新改动。操作前先检查 Git 状态，防止覆盖同时进行的工作。

## 当前问题与已验证证据

参考身体速度由姿态中心差分生成，策略侧读取物理引擎即时速度。双方共用 AMP 相对运动公式，包含骨盆旋转的 omega cross offset 修正。

针对 model_11200 的全部 33827 个未淡出五帧窗口，做过姿态匹配固定、双方相对姿态统一差分的离线检查。

| 每分量 RMS 误差 | 原始速度 | 双方相对姿态统一差分 |
|---|---:|---:|
| 身体相对线速度 | 0.4559 m/s | 0.4600 m/s |
| 身体相对角速度 | 1.6801 rad/s | 1.6788 rad/s |

统一差分后整体误差基本不变。策略相对线速度及角速度平均向量模长约为匹配参考的 2.1 倍，相对位置加速度 RMS 约 4 倍，jerk 约 6.4 倍，腕部尤其明显。支持策略实际运动偏快、变化偏急。已有相对位姿不足以精确复现世界位姿差分后再转换的全部离散过程。

检查结果位于 `logs/amp_feature_analysis/heightassist_model11200/velocity_source_check_20261122.json`。更早的参考局部尖峰、来源口径与修复对照位于 `logs/amp_feature_analysis/upright10_6900/`，对应报告包括 `reference_velocity_jump_audit.md`、`source_phase_diagnosis.md`、`repaired_reference_comparison.md`。

参考存在局部急停和折返，但修复这些局部问题后，固定旧策略的整体速度误差下降很小。不能将 AMP 低分全部归因于参考速度生成错误。

## 实验定义

公共设置为 seed 42、4096 环境、从头训练 20000 轮，standing 10、AMP 10、动态判别器更新开启。下表列出各实验变化。

| 实验 | action_rate | joint_acc | standing | 判别器更新上限 | 二阶动作差分 |
|---|---:|---:|---:|---:|---:|
| 基线 | -0.005 | -1e-7 | 10 | 20 | 无 |
| A | -0.010 | -1e-7 | 10 | 20 | 无 |
| B | -0.005 | -5e-7 | 10 | 20 | 无 |
| C | -0.005 | -1e-7 | 5 | 20 | 无 |
| 二阶强约束实验 | -0.015 | -1e-7 | 10 | 20 | -0.005 |
| D | -0.010 | -1e-7 | 10 | 10 | 0 |

二阶强约束实验和 D 已提供训练指令，其实际启动状态需要重新核实。

配置文件当前默认包含 action_rate=-0.015 和 action_second_difference=-0.005。因此复现 A、B、C、D 时必须显式将二阶项权重覆盖为 0，避免引入额外变量。A/B/C 原训练是在添加二阶项之前启动的。

## 运行目录

本地日志根目录为 `logs/rsl_rl/x2_prone_multimotion/`。

- 基线为 `2026-11-14_23-12-51_birth8_amp16_new20_stance_push_dr_selfcontact_heightassist100_scratch20k`。
- A 为 `2026-11-22_23-01-23_experimentA_actionrate001_amp10_stand10_scratch20k`。
- C 为 `2026-11-23_20-28-09_experimentC_stand5_amp10_scratch20k`。
- B 在远程主机，运行目录为 `/home/test/robotics_ws/x2_style_mimic/logs/rsl_rl/x2_prone_multimotion/2026-11-23_20-55-02_experimentB_jointacc5e7_amp10_stand10_scratch20k`。

最后一次读取的日志终点为 A 13964、B 11098、C 10461。它们是读取时的日志终点，后续应重新检查。

## A 与基线的已完成仿真验收

双方使用 model_9500、seed 42、零辅助。保留观测噪声、启动物理随机化和站立推扰。训练保存配置只差 action_rate 权重及运行名称。

| 指标 | 基线 | A |
|---|---:|---:|
| 固定首帧成功 | 64/64 | 64/64 |
| 固定首帧末态稳定 | 54/64 | 58/64 |
| 首次成功平均时间 | 5.321 s | 5.332 s |
| 全部 722 个留出状态成功 | 702/722 | 698/722 |
| 留出状态末态稳定 | 600/722 | 619/722 |
| 固定首帧稳定计时中断 | 141 | 71 |
| 涉及角速度的中断 | 85 | 25 |

相同出生后前 3 秒，A 身体相对线速度 RMS 低约 8.9%，相对角速度低约 9.5%，相对位置加速度低约 12.9%，jerk 低约 17.8%。首次成功时间基本相同。这里的加速度和 jerk 是骨盆坐标中相对位置的差分量。

结果支持 A 改善动作平滑性和末态稳定。只有单种子、单检查点，留出成功率略降。双方各自判别器交叉评分排序有分歧，不能将自己的 policy_score 直接当统一风格评分。

全部结果在 `logs/acceptance/experimentA_9500_comparison/`。

- `comparison.json` 为汇总。
- `baseline_initial.json`、`A_initial.json` 为首帧验收。
- `baseline_holdout.json`、`A_holdout.json` 为全部留出状态验收。
- `physical_comparison.json` 为相同时间段的物理指标。
- `cross_discriminator_scores.json` 为双方判别器交叉评分。
- 同目录包含 AMP windows、日志和分析脚本。

验收日志存在腕部接触 API 警告，接触冲击和穿模需要单独复核。相同种子与事件设置也不保证状态依赖的推扰触发完全相同。

## B 和 C 的曲线结论

B 已成功 SSH 读取。权重为 standing 10、action_rate -0.005、joint_acc -5e-7、AMP 10。远程与本地 `amp.py`、`getup.py`、`ppo_window.py`、`motions.json`、`motions_birth.json` 的 SHA256 一致。这些检查覆盖核心代码和清单，未据此认定所有资产、数据文件和运行库完全相同。

同一训练区间 10999 至 11098 的 100 轮平均值如下。

| 指标 | A | B |
|---|---:|---:|
| 零辅助成功率 | 97.56% | 96.48% |
| 末态连续稳定 | 10.60 s | 10.56 s |
| 起身原始风格奖励 | 0.1028 | 0.0853 |
| 起身零风格奖励比例 | 36.57% | 38.88% |
| 加权 joint_acc | -0.004267 | -0.016004 |

除去权重后，B 的全回合关节加速度平方统计量比 A 低约 25%。这与实际动作有无改善仍需回放对应，不能直接等同于起身阶段身体加速度下降。B 第 4000 轮附近成功率达到约 95%，第 6000 轮附近进入完全零辅助阶段。B 尚未执行本会话中的统一仿真验收。

C 配置与基线只差 standing 10 改为 5。最近 100 轮平均成功率为 4.29%，零辅助成功率 3.98%，末态连续稳定 0.109 s，长期停留辅助阶段 0。第 8000 轮附近曾达到约 7.5%，之后回落。已建议停止 C 并保留日志和检查点，尚未替用户执行停止。

C 的较低零风格奖励率不能证明风格更好，它长期没有站稳，运动阶段、门控及判别器样本分布与 A/B 差别很大。不能仅凭日志断言机器人趴地骗奖励，需要回放确认行为。

## 判别器更新的判断

判别器最后一层是线性输出，最小二乘目标为参考 +1、策略 -1。接近 -0.95 表示策略易被识别，不能由此认定 sigmoid 或 tanh 饱和。

风格奖励为 `clamp(1 - 0.25 * (score - 1)^2, 0, 1)`。score=-0.95 时奖励约 0.0494，score<=-1 时奖励为零，大量零值会降低这些窗口间的奖励区分度。PPO 使用标量奖励更新策略，解释时不要说 actor 经判别器反向传播。

动态判别器更新按 `上限 * 新入池窗口数 / 窗口机会数` 分配次数，保留小数余额。A/B 后期实际约每轮 3 次，C 因未站稳窗口占比高而约 18 次。

已建议 D 从 A 出发，只把上限 20 改为 10，保留动态更新。类似入池比例下平均实际更新约 1 至 2 次。若分数升高但实际动作没有改善，应考虑判别器跟进变慢这一解释。

## D 训练指令

在项目内层目录执行。

```bash
python scripts/rsl_rl/train.py \
  --task X2-Prone-MultiMotion-v0 \
  --num_envs 4096 \
  --max_iterations 20000 \
  --run_name experimentD_actionrate001_discupdates10_scratch20k \
  --seed 42 \
  --device cuda:0 \
  --viz none \
  --logger tensorboard \
  env.rewards.action_rate.weight=-0.010 \
  env.rewards.action_second_difference.weight=0.0 \
  env.rewards.joint_acc.weight=-1e-7 \
  env.rewards.standing.weight=10.0 \
  agent.algorithm.amp_cfg.style_reward_weight=10.0 \
  agent.algorithm.amp_cfg.updates_per_iteration=10 \
  agent.algorithm.amp_cfg.dynamic_updates_enabled=True
```

AMP 参数层级是 `agent.algorithm.amp_cfg`。此前给过错误的 `agent.amp` 路径，会产生 AttributeError，已纠正。

## 二阶动作惩罚实现

实现位于 `source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic/mdp/rewards.py`，类名 `ActionSecondDifferencePenalty`。配置位于同任务目录的 `x2_getup_env_cfg.py`。

公式为 `sum((a_t - 2*a_previous + a_older)^2)`。类继承 ManagerTermBase，自行保存原始动作历史，屏蔽回合前两个动作步，RewardManager 在重置时调用 `reset(env_ids)`。没有额外除以 dt，RewardManager 统一乘控制步长。当前奖励每个控制步调用一次。

本地 IsaacLab 通用奖励只有一阶 `action_rate_l2`。Spot 示例的 action_smoothness_penalty 也是一阶差分，ActionManager 仅保存当前和上一动作。

用户主动选择较强配置，一阶 -0.015、二阶 -0.005。此前建议的保守起点是一阶 -0.010、二阶 -0.001。实际训练采用哪个设置应查运行目录的 params。

## policy_score 同图产物

用户已要求将 ABC 的 policy_score 放到同一张图，已经完成。

- TensorBoard tag 为 `Loss/amp/policy_score`。
- 采用每 100 轮非重叠分组均值，保留各自日志终点。
- `logs/analysis/abc_policy_score_20261124/abc_policy_score.png`。
- 同目录 `abc_policy_score.svg` 和 `curves.json`。
- 末端分组均值约为 A -0.92785、B -0.94600、C -0.91721。

更新图时重新读取日志，注意 B 数据在远程。图中各实验使用独立判别器，曲线高低还受阶段分布影响。

## 远程主机访问

- 地址为 `192.168.5.182`，SSH 用户名为 `test`，默认端口 22。
- 用户已授权只读查看训练日志。认证凭据由用户在会话中提供，文档不保存密码。
- `/home/test/miniconda3/envs/env_isaaclab_3_0/bin/python` 可读取 TensorBoard。
- 之前误试 panda 用户认证失败，test 用户连接成功。
- 使用 SSH PTY 交互认证即可。不要修改远程训练、重启机器或停止训练，除非用户明确要求。

可通过 `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` 读取运行目录。使用 `size_guidance={'scalars': 0}` 保留全曲线。远程载入事件可能需要数十秒。

## SuperMimic 另一条工作线

项目已发展到正式 SuperMimic 加 AMP 起身入口，接手时以当前仓库为准。会话早期 `/tmp/supermimic_reference` 只是原型，当前正式入口为 `scripts/supermimic/train_getup.py`。先读 `scripts/supermimic/README.md`，使用 `python scripts/supermimic/train_getup.py --print-config` 查看生效参数。

默认参数来自任务 `agents/x2_supermimic_cfg.py` 的 `X2SuperMimicGetupRunnerCfg`。旧 TOML 需要通过 `--config` 显式指定，不能假定修改 getup.toml 会改变默认训练。

正式算法在 `source/x2_style_mimic/x2_style_mimic/algorithms/supermimic/`，已有 world_model、actor_critic、rollout、trainer、runner、buffer、objectives、gradients、observations、goals、heads，以及 AMP 特征、style、kinematics、contact_geometry、imagined_env、optimizers 和 representations 等模块。

用户此前要求按论文指导自己编写算法函数，AI 完成测试脚本。当前目标之一是验证世界模型内 BPTT actor 更新的可行性。既有入口划分预热、真实采集、世界模型更新、判别器更新、actor/critic 更新和真实物理评估。最新未提交改动涉及接触重置和 head diagnostics，具体动机和验证进度应查当前差异及相关任务上下文。

早期四关节残差局部试验曾在真实回放中获得收益，但它的任务、动作空间和算法版本与当前正式起身入口不同，不能作为当前完整 SuperMimic 起身已经成功的证明。

## Git 与同时进行的工作

分支为 main，origin 为 `git@github.com:elecsheep1/agibot_x2_style_mimic.git`。

2026 年 11 月 24 日已按用户要求全部提交并推送。

- `d6ce2a7` 增加二阶动作惩罚及权重配置。
- `18b375d` 提交当时其余 SuperMimic 改动和 `bank.partial.npz`。

当时 165 项 SuperMimic 测试通过，动作二阶差分、前两步屏蔽和部分/全部重置也通过独立检查。

本交接开始时已经出现新的未提交改动，涉及 `scripts/supermimic/getup_evaluation.py`、SuperMimic tests、heads.py、self_contact.py、x2_getup_env.py、x2_supermimic_getup_env_cfg.py，以及新增 head_diagnostics.py、getup_alignment_short.toml 和 ground contact reset 测试。应以实时 `git status` 为准，避免覆盖这些改动。上述旧测试通过不能作为新改动的验证结果。

## 下一步接手顺序

1. 查看最新 Git 状态、用户新指令和实验日志终点，确认 D 与二阶强约束实验是否实际启动。
2. 若继续分析判别器更新，优先观察 D 的 policy_score、getup_zero_reward_fraction、reference_score、成功率、门控比例和实际更新次数。
3. 对评分提升做行为核对，区别奖励区分度改善、判别器滞后和实际动作更接近参考。
4. 用户需要比较 B 时，取 model_9500 做与既有 A 验收相同的环境配置及状态集。添加二阶项后必须显式设为 0。
5. 固定首帧入口为 `scripts/rsl_rl/multimotion/play.py`，随机倒地库入口为 `play_fallen.py`。play.py 的 `--checkpoint` 需要文件可解析路径，单独传 model_9500.pt 可能找不到。
6. 留出库为 `data/fallen_states/x2_fallen_v2_hipfiltered/bank.npz`，holdout 共 722 个状态。该集合含 317 个近邻参考状态和 405 个仿真倒地状态。
7. 用户当前目标是改善 AMP 和起身动作，避免仅为提高日志分数而过度削弱判别器，或因 C 分数较高就认定策略更好。

