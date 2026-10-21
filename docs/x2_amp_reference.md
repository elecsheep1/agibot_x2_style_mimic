# X2 起身参考数据与 Isaac Lab 回放

当前阶段完成参考格式转换和跨模拟器骨架核对。未接入 AMP 判别器、风格奖励或起身训练任务；原行走 PPO 环境不变。

## 固定输入和输出

- 源快照：`../../retarget/motion_data/retargeted/x2_getup_v0_20261018/reference.pkl`，源数据保留在 retarget 工程。
- 目标数据：`data/motions/x2_getup_v0_50hz.npz`，346 帧、50 Hz、跨度 6.90 s。
- 转换说明：同目录的 `x2_getup_v0_50hz.json`。
- 检查报告：`data/motions/x2_getup_v0_50hz_validation.json`；再次回放默认写入 `logs/reference_replay/x2_getup_v0.json`。

NPZ 以 `allow_pickle=False` 读取。它是后续 AMP 数据加载器的基础格式，尚未固定判别器最终输入向量。

## 离线转换

在项目根目录，使用已有 gmr 环境运行（需要 NumPy、SciPy、MuJoCo，不启动 Isaac Sim）：

```bash
python scripts/convert_x2_reference.py \
    --input ../../retarget/motion_data/retargeted/x2_getup_v0_20261018/reference.pkl \
    --output data/motions/x2_getup_v0_50hz.npz
```

已有输出默认拒绝覆盖；需要重算时明确加 `--overwrite`。脚本自动寻找源快照中的标定 XML，也可用 `--mujoco_xml` 指定。

位置和关节角使用线性插值，根旋转使用最短路径 SLERP。速度按 50 Hz 目标时间轴计算；片段末尾不补静止帧，也不把末帧连接到首帧。源时间无法被目标步长整除时，舍去最后不足一个目标步长的尾部。

## 数据约定

| 字段 | 约定 |
| --- | --- |
| `time_s` | 从零开始，固定步长 0.02 s |
| `root_pos_w`、`root_quat_xyzw` | pelvis 刚体原点的世界系位姿，Z 向上；不是质心位姿 |
| `root_lin_vel_w`、`root_ang_vel_w` | pelvis 原点世界系线速度和角速度 |
| `root_lin_vel_b`、`root_ang_vel_b` | 上述速度旋转到根坐标系；不是从参考中减去根速度 |
| `projected_gravity_b` | 世界单位重力方向 `(0,0,-1)` 旋转到根坐标系 |
| `joint_names`、`joint_pos`、`joint_vel` | 参考中完整 31 关节，包含头部；绝对角度，单位 rad |
| `policy_joint_names`、`policy_joint_pos`、`policy_joint_vel` | 从现有 `POLICY_JOINT_NAMES` 读取的 29 关节顺序，按名称重排 |
| `body_names`、`body_pos_w`、`body_quat_xyzw` | MuJoCo 正运动学的 32 个真实刚体，用于核对 Isaac 骨架；不包含 toe/sole 辅助帧 |

全关节数据包含头部，因此不能取前 29 列当策略动作。关节角也不能直接当归一化动作；当前行走动作仍是 `q_default + 0.25 * action`，本回放不经过该动作映射。

根姿态映射为 MuJoCo `pelvis` → USD `base_link`，采用单位刚体变换；完整骨架位置和方向核对用于确认这一约定。

## Isaac Lab 姿态回放

激活 `env_isaaclab_6_0`，在项目根目录运行：

```bash
python scripts/replay_x2_reference.py \
    --motion data/motions/x2_getup_v0_50hz.npz \
    --viz kit --slowdown 2 --loop
```

无窗口、完整一遍核对：

```bash
python scripts/replay_x2_reference.py \
    --motion data/motions/x2_getup_v0_50hz.npz \
    --viz none
```

脚本只复用任务的资产、场景和仿真配置，不构造行走奖励、指令、终止或动作管理器。逐帧写入根原点位姿/速度和全部关节状态，使用 `sim.forward()` 更新骨架，**物理推进次数为零**。它用于检查姿态、顺序和坐标系，不验证接触力、PD 跟踪或起身成功率。只有独立回放场景关闭重力，项目原资产和训练配置不变。

零物理步模式下，本机 PhysX/Fabric 可能只更新骨架缓存而不更新渲染网格，表现为数值帧在推进、画面一直趴地。因此回放中关闭 Fabric 和自动 USD 回写，由 `x2_reference_visuals.py` 将 PhysX 刚体世界位姿显式写入内存 stage 的独立显示变换通道；嵌套刚体设置世界变换以避免重复叠加。不会保存或修改源 USD 资产。已用实际 RTX 首、中、末三帧检查网格随参考变化；没有制作视频。

`--loop` 仅为人工观看；训练参考采样禁止产生最后一帧到第一帧的跨片段转移。没有制作视频。

## 检查与下一步

离线检查：

```bash
python -m unittest discover -s tests -p test_x2_reference.py -v
```

覆盖恒定线/角速度、跨 180° 与四元数符号翻转、机体系速度、名称重排、共同时间点保持和错误数据拒绝。

Isaac 报告逐刚体给出最大位置/方向误差，并记录完整帧数和关节写入误差。位置阈值默认 5 mm，方向阈值默认 0.5°；任何一项不通过均报错。

本次完整检查覆盖 346 帧、32 个刚体：最大位置误差约 0.063 mm，最大方向误差约 0.010°，关节状态写入误差为零。MuJoCo 中插值后的全部碰撞几何最低高度约 +0.19 mm，514 对选定自碰撞几何的最小间距约 5.75 mm；参考速度符合当前 Isaac 资产限制。这些几何结果不表示已经验收 Isaac 碰撞形状或动态接触。

下一阶段才建立独立起身环境、参考重置、片段内转移采样和 AMP 判别器。当前行走环境的低高度/大倾角终止不能用于倒地起身；腕壳接触、站稳终态和多姿态恢复仍需通过训练与物理测试验收。
