# Isaac Lab 3.0 迁移注意

本文记录本项目从 Isaac Lab 2 风格脚本迁移到 Isaac Lab 3.0（Isaac Sim / PhysX 后端）的固定规则。迁移辅助入口只放在 `scripts/isaaclab3_migration/`；原有实验、训练与评估入口保持不改。

## 1. 启动生命周期

新脚本应使用 Isaac Lab 3 的公共 API：

```python
from isaaclab.app import add_launcher_args, launch_simulation

add_launcher_args(parser)
# 解析 CLI、构造 env_cfg 后：
with launch_simulation(env_cfg, args):
    # 在这里创建环境、trainer、执行训练
    ...
```

不要继续从 `isaaclab_tasks.utils` 导入 `add_launcher_args`、`setup_preset_cli` 或 `launch_simulation`。这些是旧入口依赖的兼容路径。也不要在新迁移入口中手工用 `AppLauncher` 模拟 `launch_simulation`：Lab 3 的 `launch_simulation` 会依据环境配置选择并管理正确的 PhysX/Kit 生命周期。

创建环境前，先完成 CLI 解析，然后设置 `sys.argv = [sys.argv[0]]`，避免 Kit 二次解析训练参数。

## 2. 导入顺序

只有纯 Python 配置可以在启动仿真前导入。凡是会间接导入 USD、PXR、PhysX、Warp 或项目的环境/算法模块的代码，都应放进 `with launch_simulation(...)` 之后。推荐顺序：

1. 解析 CLI/TOML；
2. 构造 `env_cfg`；
3. `with launch_simulation(env_cfg, args)`；
4. 导入项目环境、训练器和 PhysX 相关模块；
5. 创建环境并开始 workflow。

## 3. PhysX 后端和求解器

训练启动横幅或日志必须确认使用 `isaacsim_physx` / PhysX。项目中若每个控制步都会写外力（例如 `permanent_wrench_composer`），不能使用 PGS：Isaac Sim 会报告 PGS 不支持 `enable external forces every iteration`。这类迁移入口应在构造环境前设为 TGS：

```python
env_cfg.sim.physics.solver_type = 1  # TGS；0 是 PGS
```

仅在确实存在逐步外力的任务中设置；Tracking 任务不应借用起身任务的辅助力逻辑。

## 4. 初始 reset 与环境编号

Isaac Lab 3 初始 reset 可能传入 `slice(None)`，旧代码若把 `env_ids` 直接喂给 `torch.as_tensor` 会失败。兼容写法是先规范化：

```python
if isinstance(env_ids, slice):
    env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
```

不要在 `build_runtime()` 已经完成 `env.reset()` 后无条件再调用一次 `env.reset()`。对于只读诊断，直接从当前环境状态建立 runner history，避免触发旧环境的重复 reset 路径。

## 5. 空 Manager 配置

旧任务有时会用 `events = None` 或 `curriculum = None` 表示“没有 term”。Isaac Lab 3 仍会构造相应 Manager 并解析配置，因此 `None` 会在首次 `SimulationContext.reset()` 时触发 `AttributeError: 'NoneType' object has no attribute '__dict__'`。迁移层必须替换为没有字段的 `@configclass` 空配置对象；不要修改原任务配置以免影响旧环境复现。

## 6. 自定义环境的迁移检查

每个任务先做一次小规模冒烟检查（如 16 或 32 环境），按顺序确认：

1. 环境创建完成；
2. 第一次 `env.reset()` 成功；
3. 用零动作执行一次 `env.step()`；
4. 检查 state、reward、terminated、truncated 均为有限值且形状正确；
5. 再启动训练器。

不应先用 4096 环境排查 API 迁移错误。确认单步通过后，再按照显存和 batch 配置扩容。

## 7. 当前项目入口

| 工作流 | 使用入口 |
| --- | --- |
| RSL-RL X2 多动作训练 | `scripts/isaaclab3_migration/train_x2_rsl_rl.py` |
| SuperMimic 起身训练 | `scripts/isaaclab3_migration/train_supermimic_getup.py` |
| SuperMimic 起身想象审计 | `scripts/isaaclab3_migration/audit_getup_imagination.py` |
| SuperMimic Tracking 训练 | `scripts/isaaclab3_migration/train_supermimic_tracking.py` |

原始 `scripts/rsl_rl/`、`scripts/supermimic/` 与 `scripts/supermimic_tracking/` 文件不作为新机器的 Isaac Lab 3 启动入口。

## 8. 常见非根因日志

- `env_cfg.viewer is deprecated`：配置迁移提示，不会导致训练退出。
- 远程资产从 `/tmp` cache 加载：首次会校验远端，通常不影响正确性。
- `Unexpected reference count ... UsdStage`：常出现在异常或提前关闭后的清理阶段；应向前查找首个错误或生命周期不匹配，不要把它当作根因。
- Protobuf `File already exists in database`：当前环境中的扩展加载警告；若环境可稳定完成 reset/step，通常不是任务脚本故障来源。

## 参考

- [Isaac Lab 3.0 migration guide](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/migrating.html)
- [Isaac Lab app lifecycle API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.app.html)
