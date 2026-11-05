# X2 retarget 离线质量检查

error 为硬约束/数据错误；warning 为待复查线索。通过不代表动力学可执行。

## x2_lafan_subject2_prone_getup_umr_50hz.npz: review

输入：`/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/data/motions/lafan1_getup_umr/x2_lafan_subject2_prone_getup_umr_50hz.npz`

|级别|检查|部位|时段 s|峰值时刻 s|峰值 / 阈值|
|---|---|---|---|---|---|
|warning|joint_high_frequency_rad|left_wrist_pitch_joint|0.040–0.040|0.040|0.030625 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_roll_joint|0.040–0.040|0.040|0.036368 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_roll_joint|0.120–0.240|0.180|0.035897 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_pitch_joint|0.900–0.960|0.920|0.028558 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_pitch_joint|1.100–1.160|1.120|0.030755 / 0.025|
|warning|joint_high_frequency_rad|left_knee_joint|2.020–2.160|2.100|0.032711 / 0.025|
|warning|joint_high_frequency_rad|left_hip_pitch_joint|2.120–2.160|2.140|0.025635 / 0.025|
|warning|joint_high_frequency_rad|left_knee_joint|2.340–2.460|2.380|0.032927 / 0.025|
|warning|joint_high_frequency_rad|left_hip_pitch_joint|2.400–2.500|2.460|0.02878 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_roll_joint|3.640–3.820|3.760|0.041021 / 0.025|
|warning|joint_high_frequency_rad|right_elbow_joint|3.780–3.920|3.860|0.042248 / 0.025|
|warning|joint_high_frequency_rad|right_hip_roll_joint|3.880–3.940|3.920|0.027554 / 0.025|
|warning|joint_high_frequency_rad|right_knee_joint|3.880–3.920|3.900|0.026415 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_pitch_joint|3.880–3.980|3.920|0.029225 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_roll_joint|3.900–4.000|3.940|0.030172 / 0.025|
|warning|sole_proxy_sliding_m_s|right|3.980–4.260|3.980|1.6905 / 0.08|
|warning|joint_high_frequency_rad|right_elbow_joint|4.100–4.340|4.220|0.03607 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_roll_joint|4.180–4.240|4.200|0.02834 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_pitch_joint|4.420–4.540|4.480|0.038435 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_roll_joint|4.440–4.600|4.520|0.046446 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_yaw_joint|4.580–4.600|4.580|0.026304 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_pitch_joint|4.680–4.720|4.700|0.028056 / 0.025|
|warning|joint_high_frequency_rad|left_elbow_joint|4.700–4.760|4.720|0.028676 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_roll_joint|4.740–4.800|4.760|0.028545 / 0.025|
|warning|sole_proxy_sliding_m_s|right|4.860–5.020|4.940|0.18293 / 0.08|
|warning|joint_high_frequency_rad|left_elbow_joint|4.960–5.000|4.980|0.026667 / 0.025|
|warning|joint_high_frequency_rad|left_ankle_pitch_joint|4.980–5.080|5.040|0.030308 / 0.025|
|warning|sole_proxy_sliding_m_s|right|5.080–5.340|5.140|0.15713 / 0.08|
|warning|joint_high_frequency_rad|left_knee_joint|5.140–5.300|5.220|0.042178 / 0.025|
|warning|joint_high_frequency_rad|left_ankle_pitch_joint|5.240–5.320|5.280|0.029629 / 0.025|
|warning|joint_high_frequency_rad|left_knee_joint|5.480–5.600|5.540|0.034176 / 0.025|
|warning|joint_high_frequency_rad|left_elbow_joint|5.600–5.720|5.660|0.030687 / 0.025|
|warning|sole_proxy_sliding_m_s|right|5.620–5.660|5.640|0.083517 / 0.08|
|warning|sole_proxy_sliding_m_s|left|5.640–5.840|5.640|1.1156 / 0.08|
|warning|joint_high_frequency_rad|right_elbow_joint|5.820–5.860|5.840|0.025989 / 0.025|
|warning|sole_proxy_sliding_m_s|right|6.000–6.400|6.240|0.3926 / 0.08|

未验证：Dynamic feasibility, actuator torque and balance stability；Human/IK target reachability without original target residuals；Isaac USD collision hulls; native MuJoCo collision geometry only

## x2_lafan_subject2_supine_roll_getup_umr_50hz.npz: review

输入：`/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/data/motions/lafan1_getup_umr/x2_lafan_subject2_supine_roll_getup_umr_50hz.npz`

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
|warning|joint_high_frequency_rad|right_ankle_pitch_joint|2.580–2.660|2.620|0.03153 / 0.025|
|warning|sole_proxy_sliding_m_s|right|2.940–3.200|2.940|1.9706 / 0.08|
|warning|joint_high_frequency_rad|right_hip_pitch_joint|3.000–3.020|3.020|0.025193 / 0.025|
|warning|joint_high_frequency_rad|left_hip_pitch_joint|3.200–3.320|3.260|0.036613 / 0.025|
|warning|joint_high_frequency_rad|left_hip_roll_joint|3.380–3.540|3.460|0.053312 / 0.025|
|warning|joint_high_frequency_rad|left_hip_pitch_joint|3.440–3.600|3.520|0.048598 / 0.025|
|warning|joint_high_frequency_rad|right_hip_roll_joint|3.460–3.500|3.480|0.026358 / 0.025|
|warning|sole_proxy_sliding_m_s|left|3.600–3.780|3.600|0.83775 / 0.08|
|warning|joint_high_frequency_rad|left_wrist_roll_joint|3.620–3.680|3.660|0.030765 / 0.025|
|warning|sole_proxy_sliding_m_s|right|3.640–3.720|3.680|0.10538 / 0.08|
|warning|joint_high_frequency_rad|left_shoulder_yaw_joint|3.700–3.700|3.700|0.025349 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_roll_joint|3.800–3.880|3.840|0.035281 / 0.025|
|warning|sole_proxy_sliding_m_s|right|3.960–4.060|4.000|0.084707 / 0.08|
|warning|sole_proxy_sliding_m_s|right|4.340–5.060|4.920|0.29714 / 0.08|
|warning|joint_high_frequency_rad|left_elbow_joint|5.620–5.620|5.620|0.025238 / 0.025|

未验证：Dynamic feasibility, actuator torque and balance stability；Human/IK target reachability without original target residuals；Isaac USD collision hulls; native MuJoCo collision geometry only
