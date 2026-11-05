# X2 retarget 离线质量检查

error 为硬约束/数据错误；warning 为待复查线索。通过不代表动力学可执行。

## final.npz: review

输入：`/home/panda/robotics_ws/x2_style_mimic/x2_style_mimic/logs/retarget_quality/lafan_prone_support_v2_20261105/final.npz`

|级别|检查|部位|时段 s|峰值时刻 s|峰值 / 阈值|
|---|---|---|---|---|---|
|warning|joint_high_frequency_rad|left_wrist_pitch_joint|0.040–0.040|0.040|0.030625 / 0.025|
|warning|joint_high_frequency_rad|right_elbow_joint|0.040–0.040|0.040|0.030636 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_pitch_joint|0.900–0.960|0.920|0.028558 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_pitch_joint|1.100–1.160|1.120|0.030755 / 0.025|
|warning|joint_high_frequency_rad|left_knee_joint|2.060–2.080|2.060|0.025307 / 0.025|
|warning|joint_high_frequency_rad|left_knee_joint|2.360–2.460|2.400|0.028913 / 0.025|
|warning|joint_high_frequency_rad|left_hip_pitch_joint|2.400–2.500|2.440|0.028176 / 0.025|
|warning|joint_high_frequency_rad|right_elbow_joint|3.560–3.680|3.580|0.037422 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_yaw_joint|3.600–3.760|3.680|0.10387 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_pitch_joint|3.620–3.660|3.660|0.028036 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_yaw_joint|3.620–3.740|3.700|0.07914 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_pitch_joint|3.660–3.720|3.700|0.044936 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_roll_joint|3.700–3.760|3.740|0.034698 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_roll_joint|3.720–3.740|3.740|0.030849 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_pitch_joint|3.760–3.840|3.800|0.0449 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_yaw_joint|3.800–3.940|3.860|0.062278 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_yaw_joint|3.820–4.040|3.900|0.078021 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_pitch_joint|3.820–3.920|3.860|0.040522 / 0.025|
|warning|joint_high_frequency_rad|right_hip_yaw_joint|3.840–3.920|3.880|0.034769 / 0.025|
|warning|joint_high_frequency_rad|right_knee_joint|3.840–3.980|3.920|0.034359 / 0.025|
|warning|joint_high_frequency_rad|right_hip_roll_joint|3.900–3.980|3.940|0.029492 / 0.025|
|warning|joint_high_frequency_rad|right_shoulder_pitch_joint|3.940–4.040|3.980|0.031732 / 0.025|
|warning|joint_high_frequency_rad|right_hip_yaw_joint|4.060–4.120|4.080|0.02908 / 0.025|
|warning|joint_high_frequency_rad|right_wrist_roll_joint|4.100–4.120|4.100|0.025345 / 0.025|
|warning|joint_high_frequency_rad|right_elbow_joint|4.140–4.360|4.220|0.034976 / 0.025|
|warning|joint_high_frequency_rad|right_knee_joint|4.160–4.240|4.200|0.031295 / 0.025|
|warning|joint_high_frequency_rad|right_hip_roll_joint|4.200–4.260|4.240|0.027751 / 0.025|
|warning|sole_proxy_sliding_m_s|right|4.240–4.260|4.240|0.16689 / 0.08|
|warning|joint_high_frequency_rad|left_shoulder_pitch_joint|4.420–4.540|4.480|0.038435 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_roll_joint|4.440–4.600|4.520|0.046446 / 0.025|
|warning|joint_high_frequency_rad|left_wrist_yaw_joint|4.580–4.600|4.580|0.026304 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_pitch_joint|4.680–4.720|4.700|0.028056 / 0.025|
|warning|joint_high_frequency_rad|left_elbow_joint|4.700–4.760|4.720|0.028676 / 0.025|
|warning|joint_high_frequency_rad|left_shoulder_roll_joint|4.740–4.800|4.760|0.028545 / 0.025|
|warning|joint_high_frequency_rad|left_elbow_joint|4.960–5.000|4.980|0.026667 / 0.025|
|warning|joint_high_frequency_rad|left_ankle_pitch_joint|4.980–5.060|5.040|0.029204 / 0.025|
|warning|joint_high_frequency_rad|left_knee_joint|5.140–5.300|5.220|0.041048 / 0.025|
|warning|joint_high_frequency_rad|left_ankle_pitch_joint|5.240–5.380|5.300|0.038095 / 0.025|
|warning|joint_high_frequency_rad|left_knee_joint|5.460–5.600|5.540|0.042333 / 0.025|
|warning|joint_high_frequency_rad|left_ankle_pitch_joint|5.540–5.640|5.580|0.039003 / 0.025|
|warning|joint_high_frequency_rad|right_knee_joint|5.560–5.640|5.600|0.040058 / 0.025|
|warning|joint_high_frequency_rad|left_elbow_joint|5.600–5.720|5.660|0.030687 / 0.025|
|warning|joint_high_frequency_rad|left_knee_joint|5.720–5.780|5.740|0.030201 / 0.025|
|warning|sole_proxy_sliding_m_s|left|5.720–5.840|5.720|0.60442 / 0.08|
|warning|joint_high_frequency_rad|right_elbow_joint|5.820–5.860|5.840|0.025989 / 0.025|

未验证：Dynamic feasibility, actuator torque and balance stability；Human/IK target reachability without original target residuals；Isaac USD collision hulls; native MuJoCo collision geometry only
