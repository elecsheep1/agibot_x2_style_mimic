# UMR 局部修复 v6

本次从 v5 候选继续修改，不覆盖原始 UMR、v5 或训练清单。

- 第 3 条 139_17：2.65～4.40 s，采用 sigma=0.10 s 的手部世界位置/旋转轨迹平滑，再做双臂几何约束 IK；保留手部移动，不将这一段锁死。单关节允许修正上限 0.20 rad。该区间之外关节与根位姿逐元素不变；腿部及根位姿全片段不变，前段 1 秒修复和后段锁脚保留。
- 第 4 条 140_02：5.40～6.90 s，同时平滑骨盆旋转和三个腰关节，sigma=0.14 s；复用 v5 足部世界锚点，仅在此区间补偿腿部/根位置。5.34 s 与区间外位姿逐元素不变，膝部伸直回避和锁脚保持。

量化：3～4 s 左手腕世界角加速度峰值约 327→5.2 rad/s²，右手腕约 82.5→4.9 rad/s²；5.7～6.4 s 躯干世界角加速度峰值约 13.1→3.0 rad/s²。详细指标见 data/motions/umr_local_v6/validation_summary.json。

两条新候选通过独立全帧原生几何检查和硬限位检查；Isaac CPU 完整回放的位置/方向映射通过，最大位置误差约 0.063 mm。原有 AMP 解析身体速度与有限差分参考速度不一致仍存在，amp_consistency_passed=false，未把它声明为通过。没有验证动力学平衡/接触力/执行器可行性。

工具扩展：repair_x2_reference_local.py 增加 hand-target=smooth 与 joints 模式；repair_x2_reference_support.py 增加固定锚点和 edit-interval，并消除固定帧冗余变量改善数值稳定性。

演示：scripts/replay/replay_x2_reference.py --motion data/motions/umr_local_v6/<对应NPZ> --viz kit --loop --slowdown 2。人工重点查看 139_17 的 3～4 s 和 140_02 的 6 s 左右。
