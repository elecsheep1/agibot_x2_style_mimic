# agibot_x2_style_mimic

基于智元灵犀 X2 Ultra 的风格化运动学习与地形适应实验，采用 Isaac Lab ManagerBased 工作流与 RSL-RL。

目前已完成 X2 Ultra v1.3.0 资产导入、零动作检查和 PPO 平地速度跟踪基线。基线回放已观察到交替迈步，但仍存在小碎步和脚部滑移。当前正在接入基于接触力历史的滑移惩罚；脚部接触报告修复尚待运行验证。AMP、地形任务和实机部署尚未完成。

## 环境与当前配置

当前开发环境使用 Ubuntu 24.04、Python 3.12、Isaac Sim 6.0 和本地 Isaac Lab 3.0 开发版本。Isaac Lab 开发版接口可能变化，复现时应记录所用源码版本。当前检查到本地 Isaac Lab 的 HEAD 为 `ffff603eafc6b74264a5261cc0183d6a65390d78`；该信息不表示上游原版在该提交上无需其他本地改动即可复现。

| 项目 | 当前设置 |
|---|---|
| 任务 ID | `Template-X2-Style-Mimic-v0` |
| 模型 | 官方 X2 Ultra URDF v1.3.0 转换的 USD |
| 模型可动关节 | 31 个，包含头部 Yaw/Pitch |
| 策略动作 | 身体 29 个关节的位置偏移，不含头部 |
| 动作缩放 | `0.25`，相对默认关节位置 |
| 物理仿真 | `dt=0.002 s`，500 Hz |
| 策略更新 | `decimation=10`，50 Hz |
| Actor 输入 | `policy`，96 维 |
| Critic 输入 | `policy + privileged`，99 维 |
| 速度指令 | 前进 0～0.5 m/s，侧移和转向为零 |
| Episode 时长 | 10 s |

当前版本根姿态四元数采用 `(x, y, z, w)` 顺序，单位旋转为 `(0, 0, 0, 1)`。

## 目录

```text
agibot_x2_style_mimic/
├── scripts/
│   ├── zero_agent.py
│   ├── prepare_x2_contact_reports.py
│   └── rsl_rl/                     # 训练和回放入口
└── source/x2_style_mimic/
    └── x2_style_mimic/
        ├── assets/
        │   ├── x2_ultra_cfg.py      # 初始姿态与执行器配置
        │   └── robots/x2_ultra/
        │       └── x2_ultra/        # USD 入口及 payloads 配套资源
        └── tasks/manager_based/x2_style_mimic/
            ├── x2_style_mimic_env_cfg.py
            ├── x2_walk_params.py
            ├── agents/rsl_rl_ppo_cfg.py
            └── mdp/                # 自定义奖励等函数
```

## 安装

Ubuntu 如未安装 Git LFS，先执行：

```bash
sudo apt update
sudo apt install git-lfs
```

然后初始化 Git LFS 并克隆项目：

```bash
git lfs install
git clone git@github.com:elecsheep1/agibot_x2_style_mimic.git
cd agibot_x2_style_mimic
git lfs pull
```

激活能够运行 Isaac Lab 的 Python 环境，在项目根目录执行：

```bash
python -m pip install -e source/x2_style_mimic
python scripts/list_envs.py
```

USD/USDA 等模型文件由 Git LFS 管理。必须下载为真实资源，不能只保留 LFS 指针；同时保留全部 payloads 文件及相对路径。

## 当前验证范围

| 内容 | 状态 |
|---|---|
| 资产加载、默认姿态、零动作和重置 | 已在本地检查 |
| 不带滑移项的 PPO 基线 | 已训练并回放，仍存在滑移和小碎步 |
| 历史接触滑移奖励 | 代码已接入，当前资产曾只识别到 base_link |
| 补充接触报告后的训练 | 尚待验证，下面的命令是后续实验流程 |
| AMP、台阶任务、sim2real | 尚未完成 |

训练日志和策略检查点不随仓库提供，克隆后需自行训练或另行获取兼容检查点。

## 接触报告准备

当前导入模型采用嵌套刚体结构。本地 Isaac Lab 的自动接触报告启用逻辑会在遇到父刚体后停止向下遍历，曾导致接触传感器仅识别到 `base_link`，滑移奖励无法匹配左右脚。

项目提供补充脚本。在关闭使用该资产的仿真进程后执行：

```bash
python scripts/prepare_x2_contact_reports.py --viz none
```

该命令会修改本地 `source/x2_style_mimic/x2_style_mimic/assets/robots/x2_ultra/x2_ultra/x2_ultra.usda` 入口层，为遍历到的刚体补充 PhysX 接触报告 API。脚本已加入项目，修复效果仍需通过下述零动作检查确认；如果重新转换并覆盖资产，需要重新检查报告是否齐全。

准备前可用 `git status --short` 记录资产状态；执行后用 `git diff --stat` 查看入口层是否改变，不要把准备脚本视为只读检查。

左右脚应能在接触传感器中解析为：

```text
left_ankle_roll_link
right_ankle_roll_link
```

机器人与传感器的刚体索引可能不同，奖励使用名称分别解析索引。

## 观测与动作约定

`policy` 按以下顺序拼接，导出和部署必须保持一致：

```text
velocity_command(3) → base_ang_vel(3) → projected_gravity(3)
→ joint_pos_rel(29) → joint_vel_rel(29) → last_action(29)
```

关节观测与动作采用环境文件中的 `POLICY_JOINT_NAMES` 顺序，并设置 `preserve_order=True`；HAL 消息仍需按其关节组约定重新映射。`privileged` 仅包含 3 维机身线速度，由 PPO 配置只分配给 critic。

`q_target = q_default + 0.25 * action`。动作缩放不是硬位置限幅，软关节限位奖励也不等于执行器目标裁剪。

## 零动作检查

```bash
python scripts/zero_agent.py \
    --task Template-X2-Style-Mimic-v0 \
    --num_envs 4 \
    --viz kit
```

零动作对应默认关节目标，不包含主动平衡控制。当前配置预期包含 29 维动作、96 维 policy 观测、3 维 privileged 观测、10 项奖励和 3 项终止条件。

新增滑移奖励后，应先确认环境能够初始化、持续步进和重置，再启动训练。

## 训练

先进行短训练检查：

```bash
python scripts/rsl_rl/train.py \
    --task Template-X2-Style-Mimic-v0 \
    --num_envs 1024 \
    --max_iterations 50 \
    --seed 42 \
    --run_name x2_slide_smoke \
    --viz none
```

检查通过后，从头训练带滑移惩罚的对照实验：

```bash
python scripts/rsl_rl/train.py \
    --task Template-X2-Style-Mimic-v0 \
    --num_envs 1024 \
    --max_iterations 3000 \
    --seed 42 \
    --run_name x2_dt002_slide_history_v2 \
    --viz none
```

日志和检查点保存至 `logs/rsl_rl/x2_velocity/` 下的运行目录，不纳入版本管理。两个训练命令均从头训练，不使用 `--resume`。修改奖励后请使用新的运行名，保留原基线。

滑移惩罚根据最近 3 次接触力采样判定接触，接触合力阈值为 5 N；累计相关脚部当前水平速度的平方，起始权重为 -1.0。传感器每物理步更新，因此这 3 个采样点不是 3 个策略步，也不覆盖完整 20 ms 策略周期。它是刚体速度层面的滑移代理指标，不是精确接触点切向速度。

## 回放与训练曲线

回放上面实验的最新匹配检查点：

```bash
python scripts/rsl_rl/play.py \
    --task Template-X2-Style-Mimic-v0 \
    --num_envs 1 \
    --load_run '.*_x2_dt002_slide_history_v2' \
    --viz kit \
    --real-time
```

回放原 PPO 基线时，将 `--load_run` 改为 `'.*_x2_dt002_baseline_v1'`。加载时核对日志中的检查点路径；不同配置之间回放应确认观测顺序、动作顺序、默认姿态和缩放兼容。

如果环境已安装 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl/x2_velocity
```

除奖励曲线外，还需检查速度误差、episode 时长、失败终止、脚部滑移、迈步方式和停止能力。训练统计良好不等于动作自然或可直接部署。

## 模型与部署适配状态

- 模型来源：[智元官方 agibot_x2_urdf](https://github.com/AgibotTech/agibot_x2_urdf)，使用 `X2_URDF-v1.3.0/x2_ultra.urdf`。
- 接口参考：[AimDK X2 开发手册](https://x2-aimdk.agibot.com/zh-cn/latest/index.html)。
- 当前模型包含头部 Pitch，但旧旗舰手册仅列头部 Yaw 为有效主动自由度。策略排除头部不等于固定模型关节，正式迁移前需要按实机处理。
- 资产中的 PD 为仿真起始配置，尚未完成实机辨识；不能将官方部署例程的仿真增益直接视为硬件标准参数。
- 当前关节 armature 和摩擦的实际输出为零，与官方部署例程 MuJoCo 模型存在差异，后续需要分别核对。
- 部署还需适配关节消息顺序、IMU 外参、观测缩放和归一化、策略频率、命令发布频率及异常退出逻辑。官方舞蹈模型接口不能直接替换为本项目的速度策略。

## 版本管理

`.gitignore` 排除训练输出、Python 缓存、本地环境与转换指纹 `.asset_hash`，保留 USD、网格、Python 包入口和测试源码。`.gitattributes` 负责模型的 LFS 管理，两者作用不同。

当前默认忽略 `.pt/.pth/.ckpt/.onnx/.jit` 检查点和导出模型；如需发布策略，应增加明确的例外与合适的 LFS 规则。新忽略规则不会自动取消已跟踪文件，已跟踪的 `.asset_hash` 等需另行决定是否从索引移除。

## 许可与来源

本仓库包含多种来源的材料，请按 [LICENSE](LICENSE) 中的适用范围及原文件标头使用：

| 内容 | 许可与来源 |
|---|---|
| 未另行标注的项目原创内容 | MIT，Copyright (c) 2026 elecsheep1 |
| 带 Isaac Lab 标头的模板及修改文件 | BSD-3-Clause，保留原标头；全文见 LICENSE |
| 智元 URDF/网格转换得到的机器人资产 | 上游 Mulan PSL v2；中英文全文及来源见 LICENSE |
| 独立安装的 Isaac Sim、RSL-RL 等依赖 | 按各自分发条款，不由本仓库重新授权 |

本地模型源仓库检查版本为 `575cc6b988f976c23550e0db85aa1e5475d3652d`。这是来源记录，不是对全部转换产物的逐字节验证。转换为 USD 或补充接触报告不表示将上游模型重新授权为 MIT。

当前 `source/x2_style_mimic/setup.py` 仍保留脚手架的 `license="Apache-2.0"`，包作者和仓库元数据也尚未更新；这些元数据与本仓库实际来源声明不一致，应在发布 Python 包前单独整理。本次文档整理未修改 Python 源码或包元数据。
