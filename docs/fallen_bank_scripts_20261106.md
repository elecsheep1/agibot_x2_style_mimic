# 倒地状态库脚本与用户加载器接口（2026-11-06）

本次仅新增 scripts 与此文档，不修改用户负责的 mdp/getup_births.py，不修改训练默认出生比例，不启动训练。

## 脚本

- scripts/checks/build_fallen_state_bank.py：在独立 Isaac/PhysX 场景生成两类倒地状态。近邻模式从各参考第0帧的小幅关节扰动开始；受推模式从默认站姿经正常 PD 稳定，再施加水平推力，撤去推力后显式将刚度和阻尼置零，等待落定。不是用零 action 冒充电机关闭。
- scripts/common/fallen_bank.py：脚本侧格式定义、纯数据校验、关节重排、保存、来源/划分统计，以及独立验收 reset。生产 mdp 加载器不应依赖 scripts 的 sys.path。
- scripts/rsl_rl/multimotion/play_fallen.py：独立固定状态库验收，不依赖用户尚未完成的加载器。加载相同 actor/critic 与归一化；不恢复优化器、AMP 判别器和辅助课程；全程核验零辅助。每个选中状态只计一个完整回合，覆盖全部选中行后退出。
- scripts/tests/test_fallen_bank.py：数据格式与关节重排、划分泄漏、无效四元数、错误摔倒来源及覆盖保护测试。

## 生成命令

在项目根目录、env_isaaclab_6_0 环境执行：

```bash
python scripts/checks/build_fallen_state_bank.py \
  --output data/fallen_states/x2_fallen_v1/bank.npz \
  --num-envs 16 --near-count 1024 --fall-count 1024 \
  --max-batches 512 --seed 20261106 --device cuda:0 --viz none
```

这是正式候选生成命令，本次没有执行正式 2048 状态采集。默认阈值是起始实验参数，不是已证明适用全部倒地姿态的标准。近邻目标数按启用参考动作分配配额。报告保存在输出旁 bank.report.json。

输出或报告已存在会拒绝覆盖。没有有效状态时只输出拒绝统计并报错。未达到配额会保存部分库和 complete=false 报告，并以错误退出；不能将部分库冒充完成。training_ready 要求配额完成、两来源的 train/holdout 均有样本，以及各启用动作的近邻 train/holdout 都有样本。小规模 smoke 库不用于训练。

## 验证口径和限制

- 沿用训练机器人资产、PhysX 配置及已有碰撞过滤；环境间距 4 m。
- 从碰撞网格顶点（非踝刚体原点）计算初始地面间隙。凸化碰撞形状可能与渲染网格不同，最终筛选还检查 PhysX 接触 separation。
- 身体自接触和地面接触使用显式路径 Tensor 视图；达到接触数据缓冲容量直接报错，不静默忽略。仍可能出现本地传感器 contact-report 路径警告，判定使用独立 Tensor 视图。
- 落定要求根高 0–0.35 m、root_up<0.7、根速度<0.10 m/s、角速度<0.30 rad/s、关节速度 RMS<0.30 rad/s，持续0.5秒，并有地面支撑、接触穿透不超过0.015m等。具体全部参数见 --help。
- 保存实际落定时刻的状态和小残余速度；没有擅自清零速度后仍沿用旧验证。不是动态翻滚库。
- 保存前用正常 PD 对着同一捕获关节姿态重放1秒，检查穿透、速度、关节越界和地面力。不是训练策略起身验收，不会因为当前策略起不来就删样本。
- 真实受推模式记录的是“站立受推，随后被动倒地”，不代表行走时受推、带主动防摔控制或所有 OOD 状态。
- family_id 按独立采集试验分配，哈希划分 train/holdout；近似去重忽略世界XY和yaw，量化关节/重力/高度。不同但很相似的状态仍可能保留；不能宣称严格OOD隔离或穷尽姿态多样性。
- 近邻经过被动落定后可能偏离原首帧，需可视检查，不能假定每个样本仍只是±0.03 rad。

## 文件格式：x2_fallen_bank_v1

np.load(path, allow_pickle=False)。N为样本数，J为完整关节数。所有根位置相对环境原点；四元数为XYZW；根速度为根刚体原点的世界轴速度，不是质心速度。

|字段|形状|含义|
|---|---|---|
|root_pose|N×7|XYZ+XYZW|
|root_velocity|N×6|世界轴线/角速度，保留捕获值|
|joint_pos / joint_vel|N×J|按 joint_names 排列|
|joint_names|J|完整关节名称|
|source_type|N|0=near_reference；1=simulated_fall|
|motion_id|N|近邻来源动作编号；实际摔倒为-1|
|family_id|N|独立生成试验编号|
|split|N|0=train；1=holdout|
|root_gravity_b|N×3|根坐标下单位重力，用于姿态/去重描述|
|quality|N×9|列名见 metadata_json.quality_columns|
|metadata_json|标量字符串|JSON，包括schema、资产文件指纹、参考文件SHA256、生成阈值、报告|

additional_height_offset_m=0.0：这些已经是经过落定和重放验证的局部根位姿。加载时只加 env_origins，不再加原参考重置的 height_offset，也不要再次清零捕获速度。

## 用户实现 mdp/getup_births.py

建议暴露以下接口：

```python
class FallenStateBank:
    def __init__(self, path, robot_joint_names, device, split="train"):
        ...

    def sample(self, motion_ids, source_types, generator=None):
        # motion_ids: [M]，来自原首帧分支的参考动作。
        # source_types: [M]，0选近邻，1选实际摔倒。
        # 返回 root_pose/root_velocity/joint_pos/joint_vel、
        # source_type/motion_id/bank_row。状态张量都在device。
        ...
```

### __init__ 应完成

1. np.load(...,allow_pickle=False)，解析 metadata_json。严格核验 schema、坐标系、有限值、形状、单位四元数、source_type/split/motion_id，检查 family_id 不跨 train/holdout。可参考 scripts/common/fallen_bank.py 的 load_bank 实现，但生产模块不要通过 sys.path 导入 scripts。
2. 正式训练核验 metadata.collection_report.complete 与 training_ready；验收工具允许小库，不代表训练应允许。
3. 读取当前 X2_ULTRA_USD.parent 中USD层与网格的SHA256，与 metadata.asset_hashes 比较。散列方法参考脚本 asset_hashes。
4. 按名称重排 joint_pos 和 joint_vel 到 robot_joint_names；完整关节集合必须相同。不能依赖列号碰巧一致。
5. 核对近邻来源参考的哈希/动作顺序，避免 manifest 重排后 motion_id 指向错误。当前脚本保留全部 manifest 路径和SHA256，零权重动作可能没有近邻样本。
6. 仅取 split=train 的行建立索引：near_rows[motion_id]、fall_rows；将需要的状态数组转成设备张量，按路径/资产版本缓存，避免每次 reset 读取磁盘。缺少被请求的分组立即报错，不能退回原首帧并标为扰动。

### sample 核心抽样逻辑

以下为方法主体示意，假设 __init__ 已建好 self.near_rows、self.fall_rows、self.tensors；索引也在同一 device：

```python
# 每个请求只选一个 bank_row。
index = torch.empty_like(motion_ids, dtype=torch.long)
for source in (0, 1):
    selected = source_types == source
    if not bool(selected.any()):
        continue
    if source == 0:
        for motion in motion_ids[selected].unique().tolist():
            rows = selected & (motion_ids == motion)
            pool = self.near_rows[motion]  # 不存在或为空应报错
            draw = torch.randint(len(pool), (int(rows.sum()),),
                                 device=motion_ids.device, generator=generator)
            index[rows] = pool[draw]
    else:
        pool = self.fall_rows
        draw = torch.randint(len(pool), (int(selected.sum()),),
                             device=motion_ids.device, generator=generator)
        index[selected] = pool[draw]

result = {name: values[index].clone()
          for name, values in self.tensors.items()}
result["bank_row"] = index
return result
```

sample入口须验证 source_types只含0/1、形状相同、设备一致；不要让非法source留下未初始化index。

## 接入现有 getup.py 的顺序

1. 先按现有方式抽参考状态和 initial_branch。
2. 首帧分支50%内部，以0.4概率进入状态库分支，总计20%。库分支再以0.5/0.5选近邻/实际摔倒，总配比30%原始首帧+10%近邻+10%实际摔倒+50%RSI。
3. 原参考速度清零逻辑先执行，然后再用bank返回值覆盖库分支的四组状态，避免被原 zero_initial_frame_velocity 把捕获速度覆盖。
4. 根位置仍在局部坐标下替换，再加 env_origins。height_offset仅加到非bank行，bank行不额外抬高。
5. 当前硬限位检查仍应保留；已验证bank状态若明显越界应报错，不能静默大幅裁剪后沿用验证标签。
6. birth_types：0原始首帧，1近邻，2参考RSI，3实际摔倒。实际摔倒motion_id=-1，不要伪造来源动作；新增simulated_fall分类日志，按参考动作统计自然排除-1。
7. 建议同时保存 _getup_bank_row_ids，非库出生为-1，方便定位失败状态。传入extras时在step前保存，避免自动重置后串回合。
8. 全程零辅助：本实验覆盖 assist_initial_stage=4、assist_zero_probability=1、assist_curriculum_enabled=False。不修改AMP参考库、actor/critic结构和奖励。

## 独立固定状态验收

无需先完成用户加载器即可执行；先用 model_11200 测试固定holdout，再用相同命令更换续训模型：

```bash
python scripts/rsl_rl/multimotion/play_fallen.py \
  --checkpoint logs/rsl_rl/x2_prone_multimotion/2026-11-05_21-50-40_lafan8_amp10_stand10_heightassist100_rsi50_d20b4096_scratch/model_11200.pt \
  --bank data/fallen_states/x2_fallen_v1/bank.npz \
  --split holdout --source all --num-envs 16 \
  --report logs/reference_physics/model11200_fallen_v1_holdout.json \
  --device cuda:0 --viz none
```

--source near/fall 可以分别验收；--viz kit --real-time 可看画面。报告拒绝覆盖。循环逐条覆盖所选行，每行只计一个完整回合；并行环境尾部多出的行不重复计数。没有混合训练随机抽样，也没有AMP评分。

## 本次验证与后续

纯数据4项单元测试和脚本语法检查已通过。四环境最终采集器生成近邻4个、实际摔倒4个；另一轮小库的train分区7个状态完成逐项策略验收，6/7成功且末态稳定。仅用于脚本联调，不构成泛化结论。logs/fallen_bank/smoke*不用于训练，样本太少，不能保证所有动作/来源在train和holdout都有覆盖。正式库仍需批量采集、检查coverage/training_ready和可视抽检，然后才能接入训练。
