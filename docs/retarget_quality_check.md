# 一键检查 X2 retarget 参考动作

在项目根目录执行（仅离线读取动作，不启动 Isaac、不训练、不修改 NPZ）：

```bash
/home/panda/miniconda3/envs/umr/bin/python scripts/checks/check_x2_retarget.py
```

默认检查 `data/motions/prone_getup/motions.json` 全部动作，输出
`logs/retarget_quality/report.md`、`summary.json` 和每条动作的独立 JSON。
报告包含异常起止秒数、峰值时刻、部位、实测值、阈值、输入 SHA256 和碰撞过滤列表。
默认阈值是筛查起点，warning 需要回放确认，不能当作自动修复依据。

## 新 retarget 的用法

UMR 原始输出可直接检查；不读取 pickle。原始 qpos 的自由关节四元数按 MuJoCo wxyz 解释。
训练 NPZ 按其 schema 中的 xyzw 约定解释。模型路径从元数据读取，可用 `--model` 覆盖。

```bash
/home/panda/miniconda3/envs/umr/bin/python scripts/checks/check_x2_retarget.py \
  --motion /绝对路径/reference.npz \
  --model /绝对路径/robot.xml \
  --output-dir logs/retarget_quality/new_motion
```

多文件可重复 `--motion`，也可使用 `--manifest /路径/motions.json`。
不同目录同名文件会生成不同报告，不相互覆盖。批量检查遇到坏文件会记录错误并继续。
原始 UMR 的关节映射采用该输出配套 XML 的 qpos 顺序，务必选择匹配的 XML。

## 检查范围

- 数据：帧数、时间步长、有限数值、单位四元数、schema；训练 NPZ 另核对存储刚体姿态与 FK。
- 硬约束：关节限位、机器人与地面穿透、自碰撞。默认穿透容差 1 mm。
- 抖动线索：关节加速度、jerk、高频偏差，以及刚体平移/旋转加速度。旋转使用 SO(3) 相邻差，四元数符号翻转不会误报。
- 浮起线索：骨盆高于 0.45 m、躯干倾角小于 60°且所有机器人碰撞表面离地超过 1.5 cm。正常摆腿、手撑地起身不因双脚离地自动判错。
- 滑动线索：脚部碰撞表面近地、脚局部 Z 轴近竖直、持续至少 0.1 s 时，足部几何支撑点均值的水平速度超过 0.08 m/s。仍可能包含滚动，属于 warning。
- 速度：需提供匹配的 `--urdf robot.urdf` 或配置中 `joint_velocity_limits_rad_s`；未提供时明确记为未验证，不猜测硬件速度。

自碰撞排除同体、父子体、XML exclude，以及当前 X2 五组 authored filters；报告列出实际排除项。
不同 X2 碰撞版本可通过配置的 `excluded_body_pairs` 替换默认五组。碰撞使用原生 MuJoCo 几何，未核对 Isaac USD 碰撞凸包。

**“不可达”只能部分判断**：限位违规和存储刚体目标与 FK 不一致可检出；合法 qpos 本身就是运动学可达姿态。
若要判断原始人体 IK 目标是否不可达，必须另保存目标位置/姿态与求解残差。
本工具不验证力矩、动力学平衡、真实接触力或策略是否能执行，`pass` 不代表动力学可执行。

## 精确接触阶段与阈值

对于复杂动作，建议自行标注支撑阶段，避免几何推测漏掉“本应落地但浮着”的单脚：

```json
{"left": [[4.7, 7.8]], "right": [[4.9, 7.8]]}
```

通过 `--contacts contacts.json` 使用，仅允许检查单文件；单位秒，必须包含 left/right 两侧。
这会用标注取代推测支撑阶段；支撑阶段脚离地也会报警。仍只对近水平脚评估滑动。

配置示例 `quality.json`，通过 `--config quality.json` 使用：

```json
{
  "penetration_m": 0.0001,
  "joint_acceleration_rad_s2": 100.0,
  "body_angular_acceleration_rad_s2": 100.0,
  "sliding_m_s": 0.05,
  "joint_velocity_limits_rad_s": {"left_knee_joint": 10.0}
}
```

速度值只是配置格式示例，请以配套机器人模型为准。所有实际阈值写入报告。
退出码：0 无硬错误（可能有 warning）；1 硬错误，或指定 `--strict` 后存在 warning；2 输入/检查失败。

回归测试：

```bash
/home/panda/miniconda3/envs/umr/bin/python -m unittest discover -s scripts/tests -p test_check_x2_retarget.py
```
