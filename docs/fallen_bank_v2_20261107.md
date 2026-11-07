# 独立扩展状态库 v2

最终合并库：data/fallen_states/x2_fallen_v2/bank.npz，共3584个状态。新增库独立保存为data/fallen_states/x2_fallen_v2_additions/bank.npz，共1536个。旧v1文件未覆盖，合并库前2048行所有状态数组和split均与旧库逐项一致。

|库/来源|train|holdout|总数|
|---|---:|---:|---:|
|新增近邻|411|101|512|
|新增倒地|823|201|1024|
|合并近邻|1218|318|1536|
|合并倒地|1643|405|2048|
|合并总计|2861|723|3584|

新增近邻8动作各64个，关节扰动候选幅度0.06rad。新增倒地354个来自站立受推、670个来自独立侧向摆放后关闭PD被动落定；后者不是实际站立推倒或参考动作。没有使用holdout失败姿态作生成种子，没有按策略恢复成功筛数据。

新增倒地中的158个为骨盆重力主轴Y，合并后侧向221个（train187、holdout34），旧库为63（train52、holdout11）。新增主轴X/Y/Z分别865/158/1，此分类只是骨盆朝向代理，不是全身姿态或严格OOD定义。

正式采集seed20261109、64环境、near512/fall1024、drop_fraction0.75、侧向最低128。首次试采侧向不足后修改种子支撑姿态并重新采集，正式库为改进版完整输出。原子partial文件仅为恢复检查，不可训练。

保留原质量筛选，关节限位容差统一0.003rad。新增状态XY在重放之前居中，保留高度/朝向/实际根与关节速度。落定0.5秒、重放1秒；544个候选未落定或不合格，10个重放拒绝。实际数见新增bank.report.json。此过程验证物理状态，不证明策略能恢复。

合并脚本保留split，新增family_id偏移2241，保留输入SHA256及逐输入行范围，拒绝量化重复。新增与旧库做忽略XY/yaw的量化去重；不保证连续姿态空间完全独立。旧holdout仍为holdout，新holdout302个，不移入train。

实际训练加载器冒烟检查通过：8次批量reset/512出生，其中94个库状态，验证train-only、源类型、XY=0、高度/朝向/关节位置/根速度/关节速度不变；64环境运行1100步，64完整回合，全程零辅助。报告logs/reference_physics/model15199_fallen_v2_training_reset_smoke.json。

未修改生产mdp、训练配置、奖励和AMP，未启动训练。未来续训脚本显式传 --fallen-bank data/fallen_states/x2_fallen_v2/bank.npz；固定库验收继续使用--center-xy。下一步先用model15199验收新增holdout302条，再决定续训；当前未计算新库策略恢复率。

重现正式采集：
```bash
python scripts/checks/build_fallen_state_bank.py --output NEW_PATH/bank.npz --num-envs 64 --near-count 512 --fall-count 1024 --max-batches 128 --seed 20261109 --joint-noise .06 --drop-fraction .75 --drop-joint-noise .5 --min-side-count 128 --exclude-bank data/fallen_states/x2_fallen_v1/bank.npz --device cuda:0 --viz none
```
合并：
```bash
python scripts/checks/merge_fallen_state_banks.py --inputs data/fallen_states/x2_fallen_v1/bank.npz data/fallen_states/x2_fallen_v2_additions/bank.npz --output NEW_PATH/bank.npz
```
