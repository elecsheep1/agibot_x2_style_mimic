# Contact sensor timing verification — 2026-11-09

One-second nominal-policy replay of the same64-state ordering, focus row2965,500 physics updates. Internal timestamps and raw cached buffers were read without accessing public sensor.data, so this probe did not force extra updates to ordinary contact sensors.

| Sensor | Distinct timestamp values | Update interval |
|---|---:|---|
| head_pitch_ground_contact | 50 | {0.02: 49} |
| left_foot_ground_contact | 50 | {0.02: 49} |
| left_hand_ground_contact | 50 | {0.02: 49} |
| self_contact | 500 | {0.002: 499} |

- Scene configuration has lazy_sensor_update=True. Ordinary ContactSensor inherits SensorBase.update: physics steps advance timestamps, but buffers refresh on data access. Reward/progress access currently happens on20ms control steps.
- Custom SelfContactSensor overrides update with force_recompute=True, so it really samples each2ms physics step.
- Therefore ordinary history_length=3 is three refresh samples (normally control-step samples here), NOT a6ms physical history. Earlier descriptions of it as6ms were incorrect. Self-contact history_length=4 is four actual physics samples.
- At physics t=0.810s, direct head ground Fz=17621.86N. Ordinary head sensor last refreshed at0.800s and cached history peak=9145.33N. At t=0.822s, after the next control reward read, cached history peak=12879.89N; the17.62kN substep peak was not retained.
- Recorder samples after scene.update and before control-step RewardManager reads. Thus the t=0.820s recorder entry still shows the0.800s cache; the following2ms recorder entry confirms the0.820s refresh.
- This event DOES trigger head-contact punishment: neighboring sampled forces already exceed10N. Do not describe it as an entirely unpenalized collision. The existing binary reward does not distinguish sampled9kN from actual17kN. A sufficiently short between-read impulse can be missed altogether.
- Changing history_length alone does not increase sampling frequency. First ensure actual per-physics-step acquisition, then define a whole-control-window peak/impulse statistic. Merely sampling faster does not solve binary severity-insensitive punishment.
- Globally disabling lazy updates can also change foot-support and wrist-load history semantics and performance. A dedicated impact monitor can be introduced separately; changes to success criteria should be explicit and versioned.
- Additional checks: six self-contact history tests and three standing reward diagnostic tests passed. Current fallen-bank play still defaults center_xy=False despite training centering bank XY.
- No production training code changed. Evidence: logs/reference_physics/model12400_sensor_timing.json.
