# model_11200 AMP 窗口诊断（2026-11-06）

固定首帧、全程零辅助、确定性策略，64 环境各 4 个完整 20 秒回合。256/256 起身成功，末态稳定 256/256。采集 255232 个有效五帧窗口，重叠窗口不是独立样本。未更新模型或训练配置。

## 统计口径

未淡出/起身窗口：末帧 hold_seconds <= 0.10 s，与训练 getup 指标一致；不是动作语义阶段标签。淡出为 0.10–0.50 s，之后 AMP 门控为零。离线原始风格分数仍可计算。

|阶段|窗口数|原始奖励均值|零奖励比例|
|---|---:|---:|---:|
|all|255232|0.4519|2.58%|
|pre_fade_hold_le_010|33827|0.1557|19.49%|
|fade_hold_010_to_050|4955|0.4812|0.00%|
|post_fade_hold_ge_050|216450|0.4975|0.00%|

训练日志约 41–43% 与本次未淡出窗口 19.49% 不可直接等同：训练采样随机动作且含混合 RSI，回放采用确定性策略和固定首帧，检查点/统计时刻也不完全相同。不能单独将差异归因于探索或 RSI。

## 按出生动作统计未淡出窗口

|动作|窗口数|零奖励比例|占全部未淡出零奖励窗口|
|---|---:|---:|---:|
|0: x2_getup_139_16_umr_v1_50hz.npz|5737|23.93%|20.83%|
|1: x2_getup_139_17_umr_v1_50hz.npz|4484|19.69%|13.39%|
|2: x2_lafan_subject2_prone_getup_umr_50hz.npz|2841|18.34%|7.90%|
|3: x2_lafan_subject2_supine_roll_getup_umr_50hz.npz|7652|15.80%|18.34%|
|4: x2_getup_139_16_umr_v1_50hz_mirror.npz|3240|26.30%|12.92%|
|5: x2_getup_139_17_umr_v1_50hz_mirror.npz|3525|21.93%|11.72%|
|6: x2_lafan_subject2_prone_getup_umr_50hz_mirror.npz|1841|23.74%|6.63%|
|7: x2_lafan_subject2_supine_roll_getup_umr_50hz_mirror.npz|4507|12.09%|8.27%|

## 时间与高度

|出生后时间|窗口数|零奖励比例|占全部未淡出零奖励窗口|
|---|---:|---:|---:|
|0-0.5 s|5376|6.21%|5.07%|
|0.5-1 s|6656|8.68%|8.77%|
|1-2 s|11856|32.62%|58.65%|
|2-3 s|6600|17.64%|17.66%|
|3-5 s|3339|19.47%|9.86%|

骨盆高度 0.4–0.5 m 的未淡出窗口零奖励率 50.32%，贡献全部未淡出零奖励窗口的 38.16%；0.2–0.4 m 为 34.82%。这只定位运动阶段，未采集接触时序，不能据此确认脚滑或支撑冲击。

## 特征偏差

整段五帧在全参考库中寻找最近邻，按当前检查点 std 标准化，距离不裁剪；D 输入按检查点裁剪。误差为对应特征差异，不等于策略跟踪同一动作同一时刻的误差。

|特征|归一化 RMS|全部特征平方误差占比|
|---|---:|---:|
|body_angular_scaled|1.1902|38.19%|
|body_linear|1.0468|29.54%|
|body_rotation6d|0.6148|20.38%|
|body_position|0.4749|6.08%|
|root_linear|1.1850|3.44%|
|root_angular_scaled|0.9327|2.13%|
|root_gravity|0.2880|0.20%|
|root_height|0.1870|0.03%|

身体角速度与线速度贡献合计 67.74% 的平方误差。零奖励子集同样由这两项主导（合计约 68.21%）。上肢双腕、肩部是主要身体偏差来源。

原始物理单位：身体相对线速度每分量 RMS 差 0.364 m/s；身体相对角速度每分量 RMS 差 1.452 rad/s（撤销 0.25 缩放）；身体相对位置每分量 RMS 差 0.0546 m；根线速度每分量 RMS 差 0.268 m/s。rotation6d 是无量纲特征，不应直接解释成角度。

策略/最近参考平均向量模长：身体相对线速度 0.638/0.348 m/s，身体相对角速度 2.586/1.343 rad/s，根线速度 0.461/0.210 m/s。这是跨五帧和身体的聚合，不是同时间参考跟踪误差，不能把近两倍差异直接归结为速度生成 bug。

## 判别器敏感性

取 1024 个未淡出窗口，逐项替换为最近参考特征，重新计算冻结检查点分数；离线分数与采集分数一致性校验通过。

|替换特征|平均分数变化|
|---|---:|
|body_angular_scaled|+0.4408|
|body_rotation6d|+0.2509|
|body_linear|+0.2227|
|body_position|+0.1050|
|root_linear|+0.0311|
|root_gravity|+0.0143|
|root_height|-0.0003|
|root_angular_scaled|-0.0013|

全部身体相对线/角速度一同替换，平均分数变化 +0.8724。混合特征可能违反物理一致性，因此这是描述性敏感性，不是因果证明、删特征依据或重新训练结果。

## 解释与下一步

当前模型起身/站稳已通过本次完整验收。残余风格偏差主要集中起身中段的相对速度，身体角速度最突出，旋转其次影响判别器。位置和高度不是首要差异。参考身体速度由姿态中心差分生成、策略侧读取即时仿真速度的已知差异仍存在，但本次结果无法单独分离该来源与真实动作速度/节奏差异。

建议保留当前满意模型。后续若优化，先同步查看 1–2 秒腕/肩/踝的相对速度及支撑时序，区分整体节奏更快、局部抖动和参考速度口径；不要直接整体增加关节加速度惩罚或删除 AMP 速度输入。

数据位置：
- logs/amp_feature_analysis/heightassist_model11200/windows.pt
- logs/amp_feature_analysis/heightassist_model11200/windows.json
- logs/amp_feature_analysis/heightassist_model11200/analysis/report.md
- logs/amp_feature_analysis/heightassist_model11200/analysis/feature_errors.json
- logs/amp_feature_analysis/heightassist_model11200/zero_reward_detail.json
- logs/amp_feature_analysis/heightassist_model11200/velocity_magnitude_detail.json
- logs/reference_physics/heightassist_model11200_amp_acceptance.json
