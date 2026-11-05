# X2 retarget 离线质量检查

error 为硬约束/数据错误；warning 为待复查线索。通过不代表动力学可执行。

## stance_final.npz: review

输入：`/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/logs/retarget_quality/lafan_supine_support_20261105/stance_final.npz`

|级别|检查|部位|时段 s|峰值时刻 s|峰值 / 阈值|
|---|---|---|---|---|---|
|warning|joint_high_frequency_rad|right_elbow_joint|0.240–0.320|0.280|0.027194 / 0.025|
|warning|joint_high_frequency_rad|left_elbow_joint|1.380–1.520|1.460|0.034554 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_pitch_joint|1.380–1.520|1.460|0.04167 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_roll_joint|1.400–1.500|1.460|0.031629 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_yaw_joint|1.460–1.560|1.500|0.03611 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_roll_joint|1.600–1.760|1.680|0.06145 / 0.025|
|warning|joint_high_frequency_rad|left_elbow_joint|1.620–1.760|1.700|0.05366 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_pitch_joint|1.620–1.700|1.660|0.038534 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_yaw_joint|1.640–1.760|1.700|0.05277 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_pitch_joint|1.720–1.760|1.740|0.030272 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_pitch_joint|1.780–1.860|1.820|0.043505 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_roll_joint|1.840–2.040|1.940|0.06981 / 0.025|
|warning|joint_high_frequency_rad|left_elbow_joint|1.880–1.940|1.900|0.027092 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_yaw_joint|1.880–1.940|1.900|0.03034 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_pitch_joint|2.000–2.040|2.020|0.027558 / 0.025|
|warning|sole_proxy_sliding_m_s|right|3.140–3.240|3.140|0.32826 / 0.08|
|warning|joint_high_frequency_rad|left_hip_pitch_joint|3.200–3.320|3.260|0.033 / 0.025|
|warning|joint_high_frequency_rad|left_hip_roll_joint|3.360–3.560|3.460|0.040977 / 0.025|
|warning|joint_high_frequency_rad|left_hip_yaw_joint|3.400–3.500|3.460|0.034119 / 0.025|
|warning|joint_high_frequency_rad|left_hip_pitch_joint|3.440–3.600|3.500|0.051396 / 0.025|
|warning|joint_high_frequency_rad|right_hip_roll_joint|3.440–3.540|3.480|0.035678 / 0.025|
|warning|joint_high_frequency_rad|right_knee_joint|3.440–3.560|3.500|0.044366 / 0.025|
|warning|joint_high_frequency_rad|left_hip_yaw_joint|3.620–3.660|3.640|0.027668 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_roll_joint|3.620–3.680|3.660|0.030765 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_yaw_joint|3.700–3.700|3.700|0.025349 / 0.025|
|warning|joint_high_frequency_rad|right_knee_joint|3.720–3.820|3.760|0.028603 / 0.025|
|warning|sole_proxy_sliding_m_s|left|3.720–3.780|3.720|0.19753 / 0.08|
|warning|joint_high_frequency_rad|right_hip_roll_joint|3.760–3.880|3.800|0.026986 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_roll_joint|3.800–3.880|3.840|0.035281 / 0.025|
|warning|joint_high_frequency_rad|left_knee_joint|4.560–4.700|4.640|0.058984 / 0.025|
|warning|joint_high_frequency_rad|left_hip_roll_joint|4.580–4.660|4.620|0.035023 / 0.025|
|warning|joint_high_frequency_rad|left_hip_pitch_joint|4.620–4.660|4.640|0.02773 / 0.025|
|warning|joint_high_frequency_rad|left_knee_joint|4.780–4.920|4.840|0.040579 / 0.025|
|warning|joint_high_frequency_rad|left_elbow_joint|5.620–5.620|5.620|0.025238 / 0.025|

未验证：Dynamic feasibility, actuator torque and balance stability；Human/IK target reachability without original target residuals；Isaac USD collision hulls; native MuJoCo collision geometry only
