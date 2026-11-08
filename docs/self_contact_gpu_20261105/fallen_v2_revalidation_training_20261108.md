# 新物理模型下 v2 状态库重验与重训入口（2026-11-08）

原库 data/fallen_states/x2_fallen_v2/bank.npz 保留不变。新库 data/fallen_states/x2_fallen_v2_hipfiltered/bank.npz，3574 状态：near/train1210，near/holdout317，fall/train1642，fall/holdout405。训练2852、留出722。原3584状态均逐一在新碰撞过滤资产下回放1秒，每0.002秒检查有限值、速度、地面冲击、接触穿入与硬限位；正常PD保持原关节角，无辅助、无策略，不按起身成功筛选。

拒绝10状态：9个地面接触分离深度低于-0.015m，1个硬限位违反超过0.003rad。原行号与指标见 bank.revalidation.json。保留状态的根姿态、根速度、关节角、关节速度、来源、family及train/holdout标记逐元素不变；quality的回放峰值两列更新。资产指纹仅在验证成功后写入新库；校验逻辑没有放宽。原asset_hashes、源库SHA256、原始行号映射均存于元数据revalidation。

旧合并库到新库的重点行号映射：{'1348': 1342, '2651': 2642}。重排后的编号不要直接和旧报告混用。

默认环境配置已改用新库，出生30%原始首帧+10%近邻+10%仿真摔倒+50%RSI。从零辅助课程stage0/100N，zero_probability0.3，curriculum_enabled=True；self_contact=-0.1，10N、4物理采样历史；AMP权重10、standing权重10。

实际启动训练短测：256环境、2轮、seed42，run=2026-11-08_02-02-32_smoke_hipfiltered_bank_selfcontact10n，resume=False，训练时间7.59秒。奖励表包含self_contact=-0.1；stage0/force_cap100N实际记录。仅验证训练链路，不代表策略学习效果。先前过滤后旧策略原始首帧验收16/16成功、末态稳定16/16仍有效；尚无新策略完整验收结果。

正式命令：

```bash
python scripts/rsl_rl/train.py \
  --task X2-Prone-MultiMotion-v0 \
  --num_envs 4096 \
  --max_iterations 20000 \
  --run_name fallen_v2_hipfilter_selfcontact10n_amp10_stand10_heightassist100_scratch20k \
  --seed 42 \
  --device cuda:0 \
  --viz none \
  --logger tensorboard
```

没有启动正式长训练，没有提交Git。训练不要添加--resume、--checkpoint或--resume-std。
