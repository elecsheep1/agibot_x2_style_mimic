# AMP 输入与开源项目对照（2026-10-30）

用户指定仓库均已按代码核对。仅用于特征设计参考，未修改本项目训练特征。

## AMP_mjlab

仓库 https://github.com/ccrpRepo/AMP_mjlab ，实际读取版本 `6c7a2947fccc973e4af8e6d90e550400f1b6fcfc`，代码副本 `/tmp/AMP_mjlab_feature_review`。

- `src/tasks/amp_loco/config/g1/env_cfgs.py` 和 `rl_cfg.py`：13 个刚体（骨盆、双髋、双膝、双踝、双肩、双肘、双腕），anchor 为 torso_link。
- `src/tasks/amp_loco/amp_env_cfg.py`：AMP 为 body_pos_b、body_ori_b、body_lin_vel_b、body_ang_vel_b，没有直接拼接关节位置/速度项。
- `rsl_rl/utils/motion_loader.py`：每帧 13×(3+6+3+3)=195 维；runner 使用两帧输入。
- `mdp/observations.py`：位置/旋转相对 torso_link；速度为世界绝对速度在各刚体自身坐标系中的表达，不是相对根速度。不能直接替换本项目相对线速度（包含根旋转修正）与相对角速度的实现。

不是“四末端位置”方案。

## Gitee legged_lab

仓库 https://gitee.com/chaomingsanhua/legged_lab ，主分支实际读取版本 `4f80081fc47fbdc643d1d78ebe4733eaba751f42`，dev/get_up 为 `f5f54fe506d7c2e558e07163fd6624ae3d44bbed`，代码副本 `/tmp/legged_lab_feature_review`。不能把 GitHub zitongbai 主分支当成此 Gitee 分支的逐行等价版本。

- `source/legged_lab/legged_lab/tasks/locomotion/amp/amp_env_cfg.py` 的 DiscriminatorCfg：去偏航的根旋转 6D、根角速度 3D、关节位置、关节速度、关键刚体相对位置。
- G1 的 KEY_BODY_NAMES：双踝、双腕、双肩，共 6 个。末端/肩位置用完整根坐标系表达。
- `config/g1/g1_amp_get_up_env_cfg.py` 在起身分支中也选上述 6 个部位；起身窗口为 10 帧。主分支普通 AMP 配置为 4 帧。
- 对 29 关节 G1：每帧 6+3+29+29+6×3=85 维。
- 没有逐关键刚体的相对旋转、相对线速度、相对角速度；没有在上述 D 输入中直接加入根高度和根线速度。

因此它接近“关节状态+少量关键点位置”的思路，但并不是只看四个末端，也不等于只用 12 维输入。

## 本项目可考虑的独立实验

保持当前根状态 10 维、29 关节位置+速度 58 维，身体层面仅保留关键点位置：

- 双腕、双踝位置：12 维，合计 80 维/帧。
- 再保留双肩位置：18 维，合计 86 维/帧，更接近上述 Gitee 部位选择。

这两个都是候选方案，尚未实现或训练验证。位置选择、姿态/速度类别删减、根姿态表达和窗口长度不宜同时全改后声称某一项生效。第一组可沿用本项目根表达和五帧窗口，明确这是“大幅简化身体特征”的组合对照；小步消融仍可先用 233→200，仅移除身体相对角速度。

关节状态仍约束手臂与腿部，因此不保留肩/膝刚体的重复旋转/速度，不等于不再约束这些部位。根高度、倾斜和根速度对本起身任务仍有信息价值，不建议仅为照搬另一个项目而一并删除。

## 本项目部位差异

[按物理量的逐部位报告](../logs/amp_feature_analysis/upright10_6900/body_report.md)。双腕相对位置 RMS 约17–18 cm，旋转误差均值约29–30°；双肩位置约9 cm，旋转约22–26°；双踝位置约12.5–12.7 cm，旋转约17–19°。

上述是全库最近整窗口对应的运动差异，不是同一个目标时刻的控制跟踪误差。手/脚等末端对撑地行为重要，误差较大不支持直接认定它们不重要。敏感性试验还表明姿态块对 D 有明显影响；降维需用任务成功率、回放及统一误差口径重新验收。
