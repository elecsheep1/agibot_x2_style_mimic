# X2 参考动作

| 文件 | 用途 | 状态 |
|---|---|---|
| x2_getup_v0_50hz.npz | 140_01，俯卧起身到半蹲 | 当前起身任务使用 |
| x2_stand_width034_candidate.npz | 站姿参考 | 当前 X2StandEnvCfg 仍读取，保留 |
| x2_getup_side_umr_v1_50hz.npz | 140_04，侧卧起身 | 已确认 UMR 候选 |
| x2_getup_140_02_umr_v1_50hz.npz | 140_02 | 已确认 UMR 候选 |
| x2_getup_140_08_umr_v2_50hz.npz | 140_08，采用 v2 | 已确认 UMR 候选 |

配套 JSON 是转换、验证或生成参数报告，不是另一份动作。UMR 候选已通过几何和跨模拟器 FK/AMP 一致性检查，尚未据此认定动力学支撑通过，也未自动接入训练任务。

已删除：140_08 v1，以及旧的 stand extension、early、cut54、reposition 实验及配套报告。历史生成元数据可能仍记录这些来源路径；现用五份 NPZ 都是自包含参考，保留配套模型及冻结包的原路径。
清理明细和校验值：../maintenance/motions_cleanup_20261021.json。
冻结模型和原始结果索引：/home/panda/robotics_ws/retarget/results/。

## 新增完整起身候选（2026-10-21）

| 片段 | 应使用的 50 Hz 参考 | 帧数 |
|---|---|---:|
| 139_16 | x2_getup_139_16_umr_v1_50hz.npz | 544 |
| 139_17 | x2_getup_139_17_umr_v1_50hz.npz | 599 |
| 139_18 | x2_getup_139_18_umr_v1_50hz.npz | 711 |
| 140_09 | x2_getup_140_09_umr_v1_50hz.npz | 339 |

四段均保留完整片段，已通过几何、FK 和 AMP 特征一致性检查；实际支撑尚待训练验证，训练配置未改。139 原帧率修正版仍保留在 UMR output。
