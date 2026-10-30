# 脚本目录

从项目根目录执行 `python scripts/<分类>/<脚本>.py`。脚本已按用途移动，旧路径不再保留入口。

|目录|用途|主要脚本|
|---|---|---|
|`retarget/`|将重定向结果转换为训练参考格式|`convert_x2_reference.py`|
|`checks/`|离线质量、几何、物理跟踪和 AMP 检查|见下方检查清单|
|`repair/`|动作平滑、接触约束修复、片段编辑与数据集构建|见下方修复清单|
|`replay/`|参考动作可视化回放、离线视频与采样图|`replay_x2_reference.py`、`render_x2_motion_review.py`|
|`common/`|其他脚本导入的公共模块|`x2_motion_utils.py`、`x2_reference_visuals.py`|
|`assets/`|资产配置维护|`prepare_x2_contact_reports.py`|
|`examples/`|环境列表、零动作和随机动作示例|`list_envs.py`、`zero_agent.py`、`random_agent.py`|
|`rsl_rl/`|策略训练、续训和策略推理验收|原训练与推理入口|
|`tests/`|离线质量检查器单元测试|`test_check_x2_retarget.py`|

`retarget/convert_x2_reference.py` 负责格式、坐标与 FK 转换。UMR/GMR 的重定向求解程序位于工作区另一个 `retarget` 工程，本目录不包含它们。

## 检查与分析

- `checks/check_x2_retarget.py`：伪影、高频抖动、疑似漂浮/滑动、关节限位和原生 MuJoCo 碰撞几何检查。
- `checks/audit_x2_reference_geometry.py`：独立几何审计。
- `checks/check_x2_reference_physics.py`：无策略的参考关节 PD 跟踪。
- `checks/check_x2_amp_transitions.py`：五帧 AMP、233 维状态、终止状态与跨回合隔离检查。
- `checks/check_reference_endpoints.py`：参考末帧与 Isaac FK 映射对照。
- `checks/check_endpoint_support.py`：参考末帧自由根节点 PD 支撑检查。
- `checks/analyze_references.py`：离线参考时序分析。
- `checks/compare_prone_getup2.py`：原版与 prone_getup2 的修复量化对照。

离线检查和修复使用安装了 NumPy、SciPy、MuJoCo、Clarabel 的环境；Isaac 回放、物理跟踪和 AMP 检查使用 Isaac Lab 环境。检查通过不等于动力学可执行。

```bash
/home/panda/miniconda3/envs/umr/bin/python scripts/checks/check_x2_retarget.py \
  --manifest data/motions/prone_getup2/motions.json \
  --output-dir logs/retarget_quality/prone_getup2_check
```

## 修复与编辑

- `repair/repair_x2_reference_feet.py`：足部几何与局部姿态修复。
- `repair/repair_x2_reference_local.py`：局部手部稳定和骨盆姿态平滑。
- `repair/repair_x2_reference_support.py`：支撑足锁定与腿部协同优化。
- `repair/repair_x2_reference_temporal.py`：带几何约束的整段手臂、腰部或腿部时序平滑。
- `repair/smooth_x2_root_preserve_feet.py`：保持世界足部位姿的根轨迹平滑。
- `repair/extend_x2_getup_reference.py`：参考动作延长。
- `repair/append_x2_foot_reposition.py`：追加足部重定位片段。
- `repair/build_prone_getup2.py`：按本次实验配方构建 prone_getup2，包含缓存与来源核验，不是通用自动修复入口。

## 参考回放

```bash
python scripts/replay/replay_x2_reference.py \
  --motion data/motions/prone_getup2/x2_getup_139_17_umr_v1_50hz.npz \
  --slowdown 2 --loop \
  --report logs/reference_replay/prone_getup2_motion3.json
```

这是不进行动力学步进的参考回放。`--amp_tolerance` 检查缩放后的全部 AMP 特征，速度与位姿误差分别报告。

## 策略训练与验收

```bash
python scripts/rsl_rl/train.py --task X2-Prone-MultiMotion-v0 --num_envs 1024 --viz none
python scripts/rsl_rl/multimotion/play.py --checkpoint /absolute/path/model.pt --num_envs 16 --viz kit
python scripts/rsl_rl/multimotion/play_transition.py --checkpoint /absolute/path/model.pt --num_envs 16 --birth-frames 215 340 450 270 --viz kit
```

`rsl_rl/multimotion/train_transition.py` 用于相同观测与 AMP 结构的续训，要求 `--resume`。`train_discriminator_reset.py` 用于判别器重置实验。

`rsl_rl/multimotion/` 中的 `diagnostics.py`、`transition_diagnostics.py`、`amp_window_audit.py`、`discriminator_reset.py` 和 `hold_targets.py` 是策略入口内部依赖，保留在策略目录。

## 维护与验证

`assets/prepare_x2_contact_reports.py` 会显式修改资产接触报告配置，普通训练和验收无需运行。

```bash
/home/panda/miniconda3/envs/umr/bin/python -m unittest discover -s scripts/tests -p 'test_*.py'
/home/panda/miniconda3/envs/umr/bin/python -m unittest discover -s tests -p test_x2_reference.py
```

## 旧模型纯策略回放

参考数据或 AMP 配置变化后，仅查看旧模型动作可使用 `--policy-only` 跳过判别器加载，并通过 `--motion-manifest data/motions/prone_getup/motions.json` 选择出生动作清单。这不恢复旧版本动作文件，不用于判别器评分或续训。未指定 `--policy-only` 时保留 AMP 严格一致性校验。

## 回放站稳条件诊断

`rsl_rl/multimotion/play.py` 会自动输出七项站稳条件的全回合达标率、高度超过成功阈值时的达标率，以及连续稳定计时中断涉及的条件。统计仅包含已完成且在验收配额内的回合，JSON 中保留逐回合、逐动作和总体结果及实际阈值。一次中断可以涉及多项条件，不能把各原因次数相加当作中断总数；回合结束本身不计为中断。采集只在回放进程内挂接终止判定阶段，不改训练共用判定文件。

回放的骨盆时序同时记录 `standing_raw`、`root_up1_raw` 和 `root_up1_gain_raw`：复用当前站立奖励函数，只把 root_up 输入置为 1，其余输入保持实际值。末段表给出最后 5 秒均值及按 standing 权重和控制步长缩放的增益；这是奖励公式对照，不代表实际回正后的动力学结果。

## AMP 特征误差分析

`checks/analyze_amp_feature_errors.py` 对回放采集的完整 AMP233 窗口做分阶段、分块、分关节和分刚体误差分析，以及检查点 D 的特征替换敏感性。采集使用 `rsl_rl/multimotion/play.py --amp-window-report <windows.pt>`，不能同时启用 `--policy-only`。详见 [6900 基线分析](../docs/amp_feature_analysis_6900_20261030.md)。
