# model_12400 physical-substep diagnosis — 2026-11-09

## Scope and controls

Four independent 6-second probes, 64 environments with identical selected-row ordering and seed42. Focus rows880,1508,1990,2965 retain the same environment slots as the prior focused probe. XY centered and assistance disabled. Each physics step is 0.002s. DR retains current material/mass/gain randomization and policy observation noise. Nominal disables them. Held-target variants still evaluate the policy but replace actions on focus environments with normalized initial joint positions; PD, limits and target limiter remain active. This is pose holding, not motor-off or passive dynamics.

No training source, reward weight, collision filter or simulator setting was modified. Probe processes completed normally; this is diagnosis, not full holdout acceptance.

## Measured external ground reaction

| Row | DR policy max ground Fz (N) | DR held target (N) | Nominal policy (N) | Nominal held target (N) |
|---|---:|---:|---:|---:|
| 880 | 9681.5 | 2356.7 | 421.5 | 430.8 |
| 1508 | 2356.8 | 1819.1 | 416.9 | 421.4 |
| 1990 | 1053.2 | 4957.3 | 417.3 | 414.8 |
| 2965 | 4479.7 | 19397.4 | 438.8 | 431.2 |

- Held-target root speeds remained below0.003m/s in both parameter settings. Thus these states do not spontaneously launch under pose-holding control; strong reactions appear when the policy changes the targets.
- Unit check: nominal row880 standing ground Fz411.94N versus total mass41.9665kg *9.81=411.69N. Reported large reactions are not explained by an obvious force-vs-impulse unit mismatch.

## Located mechanisms

### Row1508: policy-induced wrist entrapment

DR held-target maximum self contact is0N. With the policy, left_wrist_roll_link is squeezed against base_link and torso_link. At1.084s, pair forces1398.69N and1188.62N; left shoulder-roll actual-to-target error3.105rad. Maximum measured self penetration2.394mm. Nominal policy shows the same trapped pair. This is strong controller/constraint opposition after policy motion, not evidence that initial spawn geometry alone explodes.

### Row1990: folded-body upper/lower-limb contact

DR held-target self peak108.71N is left knee/left shoulder yaw, with negligible root motion. Policy action produces left hip yaw/left shoulder roll1247.39N at0.208s and stays low. Nominal policy can untangle and recover. Hence initial contact exists, but policy/parameter sensitivity changes the subsequent contact pattern; do not equate the held-target contact with the policy peak pair.

### Row880: loss of balance and wrist-ground impact

DR policy total ground Fz9681.46N at2.244s: right wrist yaw5631.1N and left wrist roll4014.6N dominate. Whole-body COM vz is -1.915m/s at this sample, root vz -1.241m/s: it is a descending impact, not evidence of an upward launch at that moment. Root angular velocity later reaches15.198rad/s; maximum root speed3.932m/s. Maximum sampled ground penetration27.876mm. Self penetration is only0.328mm. Nominal policy recovers.

### Row2965: head-ground impact/reaction and rapid rotation

Nominal policy total ground Fz19397.37N at0.810s, head_pitch_link17621.9N and left wrist roll1775.5N. COM vz0.568m/s and root angular velocity14.254rad/s after this physics step. Maximum root angular velocity15.508rad/s and speed4.521m/s. Peak-force sample ground separation -0.665mm; minimum sampled ground separation in the trajectory -2.564mm. DR policy is less unstable and recovers. Held targets do not reproduce the impact. Large ground reaction can arise without domain randomization or observation noise.

## Interpretation limits

- The evidence identifies the controller/contact interaction and affected bodies. It does not independently establish numerical solver error or exclude solver contribution; no timestep/solver-iteration sensitivity ablation was performed.
- Whole-body COM upward velocity maxima across these policy probes remain below1m/s; previous centered probes also stayed below normal standing pelvis height. These tests reproduce severe impact/tumbling and rebound, not high-altitude ballistic ejection. They cannot identify an unrecorded event from another bank state or random realization.
- Ground and self force matrices are read at every physics step. Contact separation is checked every10ms and whenever high force/rotation triggers it; it is not guaranteed to capture every substep separation minimum.
- Authored collision-vertex minimum Z at birth is slightly below floor (3.6-7.0mm), but actual held-target contact separations remain small and roots remain stable. Raw mesh-vertex Z should not alone be called explosive PhysX penetration.
- Estimated implicit PD torque is not measured drive torque. Projected joint constraint forces would not isolate the motor contribution either.
- Self-contact reward counts threshold exceedance, not force severity. Head-contact reward is binary too. Their short histories are not a full20ms peak accumulator. These rewards do not directly express a maximum acceptable impact.

## Next implementation decisions

1. Align bank acceptance XY handling with training; retain original report for historical comparison.
2. Treat trapped wrist recovery and high-impact ground contact as separate problems. Do not filter valid wrist/base/torso contacts to hide the entrapment.
3. Evaluate a separate impact-severity statistic over the complete control window, and then choose targeted head/hand-ground penalties or action regularization based on measured triggers. Do not expect stronger self-contact count alone to resolve head-ground reaction.
4. If diagnosing numerical contact behavior further, perform controlled timestep/solver sensitivity tests while retaining control period, policy, targets and physical parameters.

Artifacts: `logs/reference_physics/model12400_substeps_{dr_policy,nominal_policy,dr_hold,nominal_hold}.json`; key poses `model12400_substep_key_poses.png`. Archived scripts `model12400_substep_probe.py` and `model12400_substep_recorder.py` were run from /tmp with original names x2_model12400_substeps.py and x2_substep_recorder.py; restore those module names when reproducing.

## Follow-up correction: ordinary ground sensor timing

Measured in docs/contact_sensor_timing_review_20261109.md: ordinary head/hand/foot sensors refresh every20ms under lazy updates; their3-frame histories are not physical6ms histories. Custom self-contact refreshes every2ms. The17.62kN head peak was not cached by the ordinary sensor, although neighboring>10N values still triggered the binary head penalty.
