# 正式自接触监测验证

生产代码由用户修改，本次仅修改/新增scripts检查入口，不启用非零奖励。

16/64环境正式入口各1100控制步，完整回合及自动reset通过。每控制步10次物理采样，manual/auto reset清零、终止步快照通过。

4096环境：120控制步，20步预热，100步计时23.818秒，约17197环境步/秒；setup约54.807秒。Torch峰值allocated999.06MiB、reserved1072MiB，结束时驱动总已用7240.94MiB（包含其他进程和非Torch分配，不是传感器独占或驱动峰值）。此为推理环境吞吐，不是PPO训练吞吐，也未做监测开关的严格性能对照。没有缓冲溢出，每周期采样min/max均10。短程2.4秒仿真，不构成4096环境长期完整回合验证。已检测到自接触峰值7981N，原始候选惩罚最大2、均值0.00618；未定位此峰值配对，不能据此断言传感器错误或机器人损坏。

4环境专项隔离：neutral_air自接触与地面均0；neutral_ground地面峰值约4069N而自接触0；已知空中腕/躯干碰撞，环境0/2自接触约125.815N而1/3为0，交换后读数随环境交换，地面始终0。遮罩reset只清env0、未清env1，通过。496无序配对映射/每环境。此专项只验证中立落地和已知腕躯干配对，不证明所有配对或所有撑地姿态已覆盖。

实现目前可进入候选惩罚尺度校准/小权重实验；仍不应把raw force直接无界求和，也不应把这次纯推理SPS当作训练SPS。使用新脚本scripts/checks/probe_self_contact_monitor.py；完整报告logs/reference_physics/self_contact_formal_isolation_probe.json、self_contact_formal4096_performance.json。
