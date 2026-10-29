# UMR 第 3、4 条起身参考足部修复候选

来源：prone_getup 清单第 3 条 139_17、第 4 条 140_02，均为 UMR 数据。
最终候选目录：`data/motions/feet_repair_v3/`。原始 NPZ 与训练清单均未修改。

方法：四个踝关节角与脚部世界旋转的二阶时序平滑，带关节硬限位、原生碰撞几何地面净空和未排除碰撞对约束。其余关节、根旋转、根 XY 保持不变。允许微小根 Z 上移以维持几何净空。最大踝角改动分别约 0.0664/0.0620 rad；最大根抬高约 0.304/0.673 mm。首尾各两帧保持原样。

脚部世界旋转角加速度峰值（相邻旋转增量的时间差，rad/s²）：

| 动作/脚 | 原始 | 修复 |
|---|---:|---:|
| 139_17 left | 85.55 | 29.69 |
| 139_17 right | 102.23 | 32.32 |
| 140_02 left | 69.43 | 30.77 |
| 140_02 right | 41.07 | 24.09 |

验证：两条候选原生碰撞几何最低净空约 1.8 mm，足部相关合法碰撞对无超过 0.1 mm 的穿插，硬限位通过。完整 Isaac CPU 回放的姿态/FK 映射通过，最大刚体位置误差约 0.063 mm。

限制：仅修复足部时序折点，未做动力学支撑或完整接触锁定验证，不能宣称所有滑步或姿态异常均已消除。当前回放 AMP 检查的解析刚体速度与参考有限差分速度仍超过 1e-3 阈值；140_02 原版在相同检查下也失败且误差几乎相同。原版与修复版的详细结果见 logs/reference_replay/*feet_v3.json 及 x2_getup_140_02_original_cpu_comparison.json。

离线重现使用 `/home/panda/miniconda3/envs/umr/bin/python scripts/repair/repair_x2_reference_feet.py --input <原始NPZ> --output <新NPZ> --collision-report <原有几何审计JSON>`；默认 strength=30、foot-strength=60、max-change=0.10 rad。工具拒绝覆盖输出。

可视化：使用 `scripts/replay/replay_x2_reference.py --motion <候选NPZ> --viz kit --loop --slowdown 2`。这是运动学参考回放，不是策略推理或动力学跟踪。训练仍使用原文件，待人工查看候选后再决定数据版本切换。
