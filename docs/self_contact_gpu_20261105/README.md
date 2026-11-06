# GPU 自碰撞过滤实现备忘（2026-11-05）

状态：四环境 GPU 隔离探针通过，尚未接入训练奖励。此目录保存方法、可复跑探针、碰撞姿态和实测结果；不依赖被忽略的 logs 目录中的历史输入。

## 方法与关键接口

使用底层 PhysX Tensor `create_rigid_contact_view`，给出源刚体的完整路径列表，以及与每个源一一对应的二维过滤路径列表。不要把源路径或过滤路径换成跨环境通配符。每一行只包含同一环境的其他刚体，排除自身、地面和其他环境。

```python
from isaaclab_physx.physics import PhysxManager
import warp as wp

# 在 sim.reset() 后创建；paths[e][b] 为实际刚体 prim 的完整路径。
sources = [paths[e][b] for e in range(num_envs) for b in range(num_bodies)]
filters = [
    [paths[e][j] for j in range(num_bodies) if j != b]
    for e in range(num_envs)
    for b in range(num_bodies)
]
view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
    sources, filter_patterns=filters, max_contact_data_count=capacity,
)
forces = wp.to_torch(view.get_contact_force_matrix(dt=physics_dt))
forces = forces.reshape(num_envs, num_bodies, num_bodies - 1, 3)
```

本地版本为 Isaac Lab 3 / Isaac Sim 环境 `env_isaaclab_6_0`。探针按 XML 关节树构建当前 USD 的嵌套刚体路径；正式实现应核对实际 stage / articulation 的刚体路径和返回顺序，不能只凭刚体名字拼接并假设顺序正确。当前机器人启用了 self collisions 和 contact sensors；保留资产已有的碰撞排除关系。

探针在 sim.reset() 前后将 `/physics/disableContactProcessing` 设为 False。尚未单独验证 Tensor 方案是否需要此开关，不应把它写成已确认的必要条件。GPU 事件回调即使在 reset 后开启处理，本次仍全部为空（包括地面）；因此采用 Tensor 读数，不使用该回调。

## 验证结果

CUDA:0，4 环境，环境间距 3 m，物理步长 0.002 s，32 刚体/环境。全身视图为 128 个源、每源 31 个过滤刚体，创建没有匹配失败。

| 场景 | 各环境自碰撞峰值 N（0 / 1 / 2 / 3） |
| --- | --- |
| 正常悬空 | 0 / 0 / 0 / 0 |
| 正常落地 | 0 / 0 / 0 / 0 |
| 环境 0、2 的手臂碰躯干 | 125.815 / 0 / 125.815 / 0 |
| 环境 1、3 的手臂碰躯干 | 0 / 125.815 / 0 / 125.815 |

单独右腕 roll—躯干通道约 13.27 N；全身峰值来自更大的配对集合，二者不是同一个统计量。正常落地自碰撞通道为零，环境选择交换后读数也随之交换。结果只覆盖一种主动制造的碰撞姿态，并不证明所有身体配对均已验证。

## 后续接入位置和约束

1. 增加独立 SelfContactSensor / 管理类，负责视图生命周期、路径映射、GPU 张量和 reset 缓存；不要改现有地面 contact_forces 的用途。
2. 在场景配置注册该对象，在 mdp/rewards.py 中增加自碰撞惩罚，再由 x2_getup_env_cfg.py 配置权重。
3. 在每个物理子步读取，并在一个控制周期内保存峰值；奖励读取后清理周期缓存。仅控制步末采样可能漏掉短时冲击。
4. 给源与过滤列建立显式身体配对表，同一对只保留一个方向。不要对整个矩阵直接计数而把 A-B、B-A 重复处罚；是否双向均有数据也需实测。
5. 可先试按配对力范数超过 10 N 的数量计罚，权重 -0.1 仅是起始候选，尚未经训练验证；也可用超过阈值的连续惩罚，但需控制冲击峰值对奖励尺度的影响。
6. 不要用“总接触力减地面力”的范数代替配对力；向量抵消与接触来源会使计数含义改变。Tensor 配对力也不等于所有摩擦力的完整合力，需按接口语义使用。
7. 上大规模训练前验证 16/64/目标环境数的启动时间、显存、吞吐、缓冲溢出和环境隔离。全身配对的路径及矩阵规模为 O(E B²)，4096 的探针容量只用于本次四环境测试，不能原样当作训练配置。
8. 补测腿腿、手腿等碰撞，以及趴地、手撑地、脚落地不误罚。保留原有相邻关节/装配碰撞过滤。

## 复跑

在项目根目录执行（会启动独立四环境仿真，不启动训练）：

```bash
/home/panda/miniconda3/envs/env_isaaclab_6_0/bin/python docs/self_contact_gpu_20261105/gpu_explicit_full_probe.py --device cuda:0 --viz none
```

复跑输出写到 `logs/reference_physics/self_contact_archive_replay_gpu.json`；stage 清单写到同目录 `self_contact_archive_replay_stage.json`。归档原始结果保留在本目录 `gpu_explicit_full.json`，不会被复跑覆盖。

## 历史失败路径

- 现有 ContactSensor 加全身跨环境刚体过滤通配符：出现 expected 4 / found 8 等匹配错误。
- 改为碰撞 shape 路径过滤：出现 expected 4 / found 0。
- CPU 接触事件回调可识别自身与地面；同一事件路径在 GPU 本次无数据。
- 上述失败不能推导为 GPU 不支持自碰撞检测；显式路径 Tensor 方案已通过本次四环境测试。

详细历史日志仍在 `logs/reference_physics/self_contact_audit_20261105/`（logs 被 Git 忽略）。其中早期 summary.json 记录的是显式 Tensor 方案通过之前的状态；后续结论以本存档和 gpu_explicit_result.json 为准。
