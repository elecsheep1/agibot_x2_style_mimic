# X2 retarget 离线质量检查

error 为硬约束/数据错误；warning 为待复查线索。通过不代表动力学可执行。

## final_candidate.npz: review

输入：`/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/logs/retarget_quality/lafan_prone_support_20261104/final_candidate.npz`

|级别|检查|部位|时段 s|峰值时刻 s|峰值 / 阈值|
|---|---|---|---|---|---|
|warning|joint_high_frequency_rad|left_wrist_pitch_joint|0.040–0.040|0.040|0.030625 / 0.025|
|warning|joint_high_frequency_rad|right_elbow_joint|0.040–0.060|0.040|0.033027 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_yaw_joint|0.040–0.040|0.040|0.025331 / 0.025|
|warning|joint_high_frequency_rad|right_elbow_joint|0.200–0.260|0.220|0.028237 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_pitch_joint|0.900–0.960|0.920|0.028558 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_pitch_joint|1.100–1.160|1.120|0.030755 / 0.025|
|warning|joint_high_frequency_rad|left_knee_joint|2.060–2.080|2.060|0.025307 / 0.025|
|warning|joint_high_frequency_rad|left_knee_joint|2.360–2.460|2.400|0.028913 / 0.025|
|warning|joint_high_frequency_rad|left_hip_pitch_joint|2.400–2.500|2.440|0.028176 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_yaw_joint|3.600–3.760|3.680|0.099408 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_yaw_joint|3.600–3.740|3.680|0.069353 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_pitch_joint|3.640–3.700|3.680|0.034594 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_roll_joint|3.680–3.740|3.700|0.03338 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_pitch_joint|3.760–3.820|3.800|0.034127 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_yaw_joint|3.800–4.000|3.880|0.086439 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_pitch_joint|3.800–3.920|3.860|0.042305 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_yaw_joint|3.800–3.940|3.860|0.074726 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_roll_joint|3.820–3.860|3.840|0.02761 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_roll_joint|3.860–3.860|3.860|0.025551 / 0.025|
|warning|joint_high_frequency_rad|right_hip_roll_joint|3.880–3.940|3.920|0.02852 / 0.025|
|warning|joint_high_frequency_rad|right_knee_joint|3.880–3.940|3.920|0.026956 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_pitch_joint|3.940–4.020|3.980|0.030058 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_roll_joint|4.040–4.100|4.080|0.027739 / 0.025|
|warning|sole_proxy_sliding_m_s|right|4.100–4.280|4.100|0.87633 / 0.08|
|warning|joint_high_frequency_rad|right_elbow_joint|4.140–4.360|4.220|0.034513 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_pitch_joint|4.420–4.540|4.480|0.038435 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_roll_joint|4.440–4.600|4.520|0.046446 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_yaw_joint|4.580–4.600|4.580|0.026304 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_pitch_joint|4.680–4.720|4.700|0.028056 / 0.025|
|warning|joint_high_frequency_rad|left_elbow_joint|4.700–4.760|4.720|0.028676 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_roll_joint|4.740–4.800|4.760|0.028545 / 0.025|
|warning|sole_proxy_sliding_m_s|right|4.860–5.000|4.940|0.1705 / 0.08|
|warning|joint_high_frequency_rad|left_elbow_joint|4.960–5.000|4.980|0.026667 / 0.025|
|warning|joint_high_frequency_rad|left_ankle_pitch_joint|4.980–5.060|5.040|0.029204 / 0.025|
|warning|sole_proxy_sliding_m_s|right|5.080–5.320|5.160|0.15214 / 0.08|
|warning|joint_high_frequency_rad|left_knee_joint|5.140–5.300|5.220|0.041048 / 0.025|
|warning|joint_high_frequency_rad|left_ankle_pitch_joint|5.240–5.380|5.300|0.038095 / 0.025|
|warning|joint_high_frequency_rad|left_knee_joint|5.460–5.600|5.540|0.042333 / 0.025|
|warning|joint_high_frequency_rad|left_ankle_pitch_joint|5.540–5.640|5.580|0.039003 / 0.025|
|warning|joint_high_frequency_rad|right_knee_joint|5.560–5.660|5.620|0.0332 / 0.025|
|warning|joint_high_frequency_rad|left_elbow_joint|5.600–5.720|5.660|0.030687 / 0.025|
|warning|sole_proxy_sliding_m_s|right|5.600–5.660|5.620|0.089098 / 0.08|
|warning|joint_high_frequency_rad|left_knee_joint|5.720–5.780|5.740|0.030201 / 0.025|
|warning|sole_proxy_sliding_m_s|left|5.720–5.840|5.720|0.60442 / 0.08|
|warning|joint_high_frequency_rad|right_elbow_joint|5.820–5.860|5.840|0.025989 / 0.025|
|warning|sole_proxy_sliding_m_s|right|5.980–6.420|6.240|0.32796 / 0.08|

未验证：Dynamic feasibility, actuator torque and balance stability；Human/IK target reachability without original target residuals；Isaac USD collision hulls; native MuJoCo collision geometry only
