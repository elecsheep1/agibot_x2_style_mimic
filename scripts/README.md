# 脚本入口

从项目根目录、现有 Isaac Lab Python 环境中执行。

## 保留的 Isaac Lab 原始脚本

`rsl_rl/train.py`、`rsl_rl/play.py`、`rsl_rl/train_rsl_rl.py`、
`rsl_rl/play_rsl_rl.py`、`rsl_rl/cli_args.py` 以及 `legacy/play.py`、
`legacy/play_rsl_rl.py`、`legacy/cli_args.py` 保留本次清理前的内容。
原始脚本的兼容性不属于本次修复范围。

## 统一起身任务

从头训练仍使用原始训练入口，显式指定保留的任务：

```bash
python scripts/rsl_rl/train.py --task X2-Prone-MultiMotion-v0 --num_envs 1024 --viz none
```

起身推理使用自定义多参考验收入口，全程零辅助力，从各动作第 0 帧出生：

```bash
python scripts/rsl_rl/multimotion/play.py --checkpoint /absolute/path/model.pt --num_envs 16 --viz kit
```

指定出生帧、均衡动作覆盖和细粒度时序诊断：

```bash
python scripts/rsl_rl/multimotion/play_transition.py --checkpoint /absolute/path/model.pt --num_envs 16 --birth-frames 215 340 450 270 --viz kit
```

`multimotion/train_transition.py` 仅用于相同观测与 AMP 特征结构的续训，要求 `--resume`；
它不能将旧 80 维 AMP 或旧策略检查点迁移为当前结构。

## 诊断与参考数据

- `check_x2_amp_transitions.py`：五帧 AMP、233 维状态、真实终止状态、跨回合隔离与判别器更新检查；不更新策略。
- `check_x2_reference_physics.py`：无策略的参考关节 PD 跟踪；读取当前任务站立阈值。
- `replay_x2_reference.py`：无动力学步进的 FK 映射和完整 AMP 特征比较。
  参考刚体速度来自有限差分，速度误差与位姿误差分别列出；`--amp_tolerance` 作用于缩放后的全部特征。
- `multimotion/analyze_references.py`：离线参考时序分析。
- `multimotion/check_reference_endpoints.py`、`check_endpoint_support.py`：参考末帧几何与支撑诊断。
- `convert_x2_reference.py`、`extend_x2_getup_reference.py`、`append_x2_foot_reposition.py`：参考数据转换与编辑。
- `x2_motion_utils.py`、`x2_reference_visuals.py`：参考工具共用模块。
- `prepare_x2_contact_reports.py`：显式修改资产接触报告的独立工具，普通训练和验收不需要运行。

本次删除旧自定义辅助力/平衡推理副本和已废弃站立动作生成器。
