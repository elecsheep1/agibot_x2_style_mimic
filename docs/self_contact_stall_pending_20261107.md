# 待解决：倒地恢复中的持续自接触停滞

状态：待解决。最终稳态站姿微调已推迟；当前回看既有GPU自碰撞惩罚方案，尚未授权启用奖励或修改控制。

原始方法存档：[GPU 自碰撞过滤实现备忘](self_contact_gpu_20261105/README.md)。其中10N阈值、-0.1权重仅为早期候选，未经训练验证，不应直接视为本轮最终参数。

固定64环境、原状态顺序、seed42、XY居中、零辅助：新增302状态原验收+两次重复均298/302，失败512/603/1074/1479。原库421状态420/421，唯一失败1348。4环境结果不可替代原基线。

603（新增库，合并库2651）：右踝/左膝持续自接触约150–160N，左髋目标与实际误差约1.01rad，根高约0.138m。1348（旧库及合并库同号）：左膝/左肩附近持续自接触约330–355N，腰yaw误差约1.21rad，根高约0.13m。隐式PD力矩为估算；1348腰roll估算需求约58.6Nm、配置48Nm，不能等同实测饱和。

候选方案：独立困难姿态与解缠覆盖；专用机器人—机器人过滤下的弱、有界、持续非相邻接触惩罚，A-B/B-A去重，不混入地面支撑；从同一checkpoint做原配置对照。未证明唯一因果，不全局提高站立奖励/PD/电机上限，不把holdout603/1348并入训练。

恢复此项时先读 logs/reference_physics/fixed64_repeat_and_stall_diagnosis.md、fixed64_stall_detailed_analysis.json、v2_fixed64_repeat_comparison.json。GPU接触视图实现在scripts/checks/build_fallen_state_bank.py的Geometry/Contacts，诊断入口scripts/rsl_rl/multimotion/play_fallen.py --detailed-trace --trace-rows。
