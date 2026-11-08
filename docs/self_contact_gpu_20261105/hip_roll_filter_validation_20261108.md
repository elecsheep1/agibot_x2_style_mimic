# 骨盆—左右髋 roll 精确物理过滤

已在实际训练使用的 x2_ultra.usda 的 base_link 上应用 PhysicsFilteredPairsAPI，仅新增同一机器人内左右 hip_roll 两条完整路径。原五对显式过滤保留；未修改关节限位、执行器、自碰撞奖励或其他碰撞对。

USD 解析检查通过：只增加这两对，目标刚体均有效。GPU PhysX 单关节扫描404个状态通过：目标两对 >1N 接触由过滤前各1次降为0；仍有74个其他远端接触事件（极限人工姿态，不代表正常运行接触）。这验证过滤选择性，不说明所有接触均已验证。

旧 model_15199 在当前模型下零辅助原始首帧验收：16环境、8动作各2回合、每环境首回合、seed42、20秒。成功16/16，回合末稳定16/16；首次成功平均5.6125秒，末态连续稳定平均17.3875秒，超时16、其他终止0，与过滤前同条件结果一致。未运行PPO更新。

结果：logs/reference_physics/x2_neighbor_collision_audit/physx_probe_hip_filtered.json；logs/reference_physics/model15199_history10n_hipfiltered_reference_initial.json。

原资产本地备份：docs/self_contact_gpu_20261105/implementation/x2_ultra.before_hip_roll_filter.usda（已移出资产哈希目录）。可审阅补丁：docs/self_contact_gpu_20261105/implementation/hip_roll_assembly_filter.patch。资产目录可能受Git忽略规则影响，补丁用于保留改动记录；本次未提交Git。

尚未复验全部状态库；更改物理过滤后的状态库落定与泛化需独立验收，不能沿用所有旧结论。
