# LAFAN1 → X2 UMR 起身参考候选

已放入项目，但未加入活动训练清单、未改现有 prone_getup2。

|文件|动作|原 BVH 帧（含端点，0-based）|50 Hz 帧数|时长|
|---|---|---|---:|---:|
|x2_lafan_subject2_prone_getup_umr_50hz.npz|俯卧撑起、收腿、站立|850–1050|340|6.78 s|
|x2_lafan_subject2_supine_roll_getup_umr_50hz.npz|仰卧侧身翻转、撑地起身|1200–1370|284|5.66 s|

## 处理与来源

原始：/home/panda/robotics_ws/retarget/motion_data/lafan1/extracted/fallAndGetUp2_subject2.bvh
工作目录：/home/panda/robotics_ws/retarget/motion_data/lafan1/umr_getup
原始数据来源：https://github.com/ubisoft/ubisoft-laforge-animation-dataset
BVH→SMPL-X 映射：https://github.com/jaraujo98/lafan_to_smplx （MIT，vendor 保留许可证）。
原始 LAFAN1 许可与转换脚本许可不同，原数据条款见官方库。

使用 UMR 自定义 X2 表面求解器，不是 GMR 关节 IK。为转换后的 SMPL-X 体型新建对应点数据并训练 500 epoch；两个片段共用这份映射，没有套用 CMU139/140 的对应点。
SMPL-X 转换使用与 UMR 一致的前 10 个 beta；转换后的对应人体关节位置平均误差约 6.7 cm，详情见工作目录 conversion.json，不是机器人端的跟踪误差。
之后进行 50 Hz 重采样、轻度轨迹平滑、原生几何地面与自碰撞投影。
俯卧段对右腕过快时段局部延时约 .12 s，保留姿态路径，不改变关节限位。
未补静止尾帧。models/x2.xml 冻结模型仍引用本机 GMR X2 meshes 绝对路径，不是独立资产分发包。

## 验证结论与限制

- 两条均无硬限位、穿地、自碰撞错误，也没有命名关节速度超限警告；最小原生几何离地间隙约 1.87 / 2.00 mm。
- 仍有 46 / 30 条 review 警告，主要是残余高频与脚底代理滑动。通过硬约束检查不等于动作完全自然或动力学可执行。
- Isaac Lab 完整回放：340 / 284 帧；关节写入与姿态映射通过，最大位置误差约 .062 mm。
- 默认 AMP 1e-3 全特征严格检查未通过：根与刚体位姿均对齐，但仿真读瞬时速度，参考由相邻帧差分，速度特征最大绝对差约 .0123 / .0099。独立 MuJoCo J(q) qdot 复现同样误差，见 quality/velocity_convention.json。
- 下述回放显式使用 --amp_tolerance .02；仅适用于此运动学可视化诊断，不更改训练观测、归一化或奖励。不能据此声称两种速度定义完全一致。
- 没有运行物理跟踪或训练，不宣称能承重、能成功起身。建议人工查看 previews/*.mp4 后再决定是否加入训练。

## 回放

在项目根目录、env_isaaclab_6_0 环境：

```bash
python scripts/replay/replay_x2_reference.py \
  --motion data/motions/lafan1_getup_umr/x2_lafan_subject2_prone_getup_umr_50hz.npz \
  --viz kit --loop --slowdown 2 --amp_tolerance .02 \
  --report logs/reference_replay/lafan_prone_umr.json

python scripts/replay/replay_x2_reference.py \
  --motion data/motions/lafan1_getup_umr/x2_lafan_subject2_supine_roll_getup_umr_50hz.npz \
  --viz kit --loop --slowdown 2 --amp_tolerance .02 \
  --report logs/reference_replay/lafan_supine_roll_umr.json
```

质量复查：

```bash
/home/panda/miniconda3/envs/umr/bin/python scripts/checks/check_x2_retarget.py \
  --manifest data/motions/lafan1_getup_umr/motions.json \
  --config logs/retarget_quality/reference_velocity_repair_20261031/joint_speed_limits.json \
  --output-dir logs/retarget_quality/lafan1_umr_recheck
```

已有质量检查器 7 项测试通过。所有中间结果与原始动作保留。

## 2026-11-05 俯卧段接触修复

右掌背撑地已改为掌面向下；前段脚尖接地、后段分阶段平脚落地。根轨迹、时长和左臂不变，仍需人工查看自然度。
详细报告：/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/logs/retarget_quality/lafan_prone_support_20261104/report.md
旧版备份：/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/logs/retarget_quality/lafan_prone_support_20261104/before.npz

## 2026-11-05 support revision

Prone: right swing foot gets clearance before landing, then stable stance; mild 8-degree palm extension bias preserves palm contact (not verified heel-only loading). Root trajectory unchanged.

Supine: preserve early roll, improve swing clearance and landing; right stance fully constrained after 3.5 s, left after 4.1 s. Root translation adjusted up to 0.02961 m per axis after 3 s, root orientation preserved. Locked sole clearance approximately 0.002 m.

Original inputs backed up under logs/retarget_quality/lafan_prone_support_v2_20261105/before.npz and lafan_supine_support_20261105/before.npz. Latest reports in quality/*20261105. Residual high-frequency and landing-transition sliding warnings remain; kinematic checks do not prove dynamic support.
