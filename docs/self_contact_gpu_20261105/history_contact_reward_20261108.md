# 10 N 历史自身接触奖励（2026-11-08）

按用户要求，将持续接触门控惩罚替换为类似 AMP_mjlab 的历史计数。奖励公式位于 mdp/rewards.py 的 self_contact_history_count / self_contact_penalty。

传感器每物理子步保存同环境机器人所有唯一配对力模长的最大值，最近 4 次组成 [环境数,4] 缓存。每采样最大值严格超过 10 N 计 1，求和范围 0～4；同一采样多对接触不重复计数。负权重 -0.1，由 RewardManager 统一乘 dt。当前 dt=0.02 时，每控制步负奖励范围为 0～-0.008。

缓存跨控制步保留，在对应环境重置时清空；不包含地面和其他环境，不主动推进传感器。持续接触时间、persistent_pair_count 仍为诊断，阈值同步为 10 N，不参与奖励。旧 force_scale、duration_full、penalty_cap 及持续门控求和已移除。SelfContact/penalty_raw 复用奖励公式，语义已变为历史计数，旧版量级审计不再适用。

当前 physics_dt=0.002，因此 4 采样覆盖最近 8 ms，并非整个 20 ms 控制周期；控制周期较早发生且已经离开历史窗口的碰撞不计本步奖励。这是该历史公式的明确语义。若以后需要覆盖全部 10 子步，可另调 history_length，权重也需重审。

本次没有新增或移除物理碰撞过滤；骨盆/左右髋 roll 仍应作为独立资产改动处理。未启动 PPO 训练。

验证：Python 语法通过；10 N 严格边界、四样本计数上限、高力不放大计数、奖励读取无副作用检查通过。GPU 16 环境完整回合短测见 logs/reference_physics/self_contact_history10n_smoke16.json（以 complete 字段和运行报告为准）。
