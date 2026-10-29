# prone_getup2 修复对照

原始 prone_getup 未修改；这是独立运动学修复数据集。

|动作|部位|时段 s|修复前角加速度 rad/s²|修复后|
|---|---|---|---|---|
|x2_getup_v0_50hz.npz|left_wrist_roll_link|[2.6, 3.1]|329.91|37.98|
|x2_getup_v0_50hz.npz|right_hip_roll_link|[2.9, 3.5]|291.08|47.31|
|x2_getup_139_16_umr_v1_50hz.npz|right_elbow_link|[0, 0.2]|210.77|26.52|
|x2_getup_139_16_umr_v1_50hz.npz|left_wrist_roll_link|[4, 4.6]|147.39|25.27|
|x2_getup_139_16_umr_v1_50hz.npz|right_wrist_roll_link|[6, 6.4]|131.29|24.79|
|x2_getup_139_17_umr_v1_50hz.npz|right_elbow_link|[0, 0.2]|223.55|19.08|
|x2_getup_139_17_umr_v1_50hz.npz|left_elbow_link|[3, 4]|12.63|12.63|
|x2_getup_139_17_umr_v1_50hz.npz|right_wrist_roll_link|[8.3, 8.9]|201.89|60.00|
|x2_getup_140_02_umr_v1_50hz.npz|left_wrist_roll_link|[1.6, 2.1]|260.57|28.90|
|x2_getup_140_02_umr_v1_50hz.npz|left_hip_roll_link|[3.8, 4.3]|116.00|62.72|
|x2_getup_140_02_umr_v1_50hz.npz|torso_link|[5, 7]|2.29|2.29|

完整脚部接触稳定性、改动幅度见 repair_comparison.json；告警和未验证项见质量报告。
支撑阶段为人工指定，仍不代表动力学或控制策略验收通过。

## 最终复查（2026-10-27）

四条动作已生成完整离线运动学回放，并检查重点时段的回放采样图。第三、四条保持 UMR 来源。文件名、帧数和 50 Hz 采样率保持不变。

|动作|剩余筛查告警|硬约束错误|
|---|---:|---:|
|x2_getup_v0_50hz.npz|55|0|
|x2_getup_139_16_umr_v1_50hz.npz|85|0|
|x2_getup_139_17_umr_v1_50hz.npz|151|0|
|x2_getup_140_02_umr_v1_50hz.npz|140|0|

地面与自碰撞检查采用原生 MuJoCo 几何，穿透容差 0.1 mm。剩余告警包括高频变化和接触转换阶段的疑似滑动，因此检查状态仍为 review，不能宣称全部伪影消除。未验证动力学可执行性、执行器力矩、平衡稳定性、Isaac USD 碰撞几何；缺少关节速度限值和原始 IK 目标残差的项目也未验证。

第四条约 3.98 s 的右踝另做右腿局部平滑，保留左腿、根节点及 4.3 s 后姿态。第三条 3–4 s 手臂与第四条 5–7 s 胸部已修复段保持。

- [最终质量报告](../../../logs/retarget_quality/prone_getup2_final/report.md)
- [回放视频与采样图目录](../../../logs/retarget_quality/repair_review/)：motion1_after.mp4 至 motion4_after.mp4
- [修复量化对照](repair_comparison.json)
- [来源与输出哈希](dataset_provenance.json)

本次未改写原 prone_getup 文件，也未切换训练配置。训练使用新数据时，应显式选择本目录 motions.json。
