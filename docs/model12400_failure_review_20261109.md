# model_12400 holdout failure and launch review — 2026-11-09

## Original acceptance

- Complete 722-state holdout report, zero assist verified. Success and stable-at-end: 696/722 (96.40%). Near-reference: 315/317; simulated fall: 381/405.
- Original report has no state traces, no ground-only force time series, no per-environment randomized parameter snapshot. Original self-contact audit is not a foot-ground force report.
- Evaluation preserved bank XY (`center_xy=false`); production bank training resets center XY. Original command advice to omit `--center-xy` was incorrect for training-aligned acceptance.
- Eleven failures started beyond the 2m termination radius and ended after one control step. Nine other failures terminated later; six timed out. Later termination alone does not prove launch.

## Failed bank rows

| Group | Bank rows |
|---|---|
| later_termination | 716, 1409, 1628, 1621, 1812, 2617, 3144, 2965, 3248 |
| timeout | 880, 1178, 1508, 1990, 2640, 3443 |
| initial_outside_2m | 1075, 1108, 1134, 1157, 1374, 1673, 1713, 1759, 2017, 1885, 1935 |

- Bank row 1508: original self-contact nonzero 99.8%, maximum duration 19.881s. Bank row 1990: 100%, duration 19.793s. These are sustained-contact failures; exact trapped body pairs are not identified by the original aggregate audit.
- Original self-contact >2kN occurred in 27 states, all eventually successful. Highest event: row1813 right knee/right elbow 3858.60N; row1943 right hip yaw/torso 3788.02N. Success does not certify low impact.
- Current reward counts 4 recent physics samples above 10N; it does not scale with force magnitude. Sparse impulses are not represented by the training mean force.

## Focused reruns

Temporary script archived beside reports as model12400_probe_script.py. 64 deliberately selected holdout states: all original failures plus high-force states and fillers. Both runs centered XY. DR retained current random physics and policy observation noise. Nominal disabled all three randomization events and policy corruption. These are targeted diagnostic reruns, not unbiased whole-holdout acceptance.

Changing row ordering changes environment-slot randomization assignment; removing startup randomization also changes RNG consumption. The comparisons show sensitivity of the combined setting, not an isolated causal estimate of friction, mass, gains, noise or centering.

| Bank row | DR success | Nominal success | DR max speed (m/s) | Nominal max speed (m/s) |
|---|---|---|---|---|
| 880 | False | True | 3.889 | 1.205 |
| 1508 | False | False | 0.469 | 0.445 |
| 1990 | False | True | 0.358 | 1.000 |
| 3443 | False | True | 1.077 | 1.484 |
| 2965 | True | False | 1.147 | 4.519 |

- dr: 60/64 success; failures [880, 1508, 1990, 3443]; highest sampled pelvis height 0.680m.

- nominal: 62/64 success; failures [2965, 1508]; highest sampled pelvis height 0.674m.

- DR row880: speed 3.889m/s, angular speed 15.198rad/s. At t=2.26s: pelvis height0.464m, vertical velocity -0.928m/s, joint speed28.799rad/s, maximum per-body net contact force4687.53N. This supports falling-impact/violent tumbling rather than high upward launch.
- Nominal row2965: speed4.519m/s, angular speed13.699rad/s, maximum per-body net contact force12879.89N. Large impulses can occur without randomization or sensor noise.
- Sampled per-body net contact forces include all contact sources and are not ground-only forces. Physics substep maxima can exceed control-sampled maxima. PD estimates are not actual measured actuator torques.
- DR rows1508/1990 stayed low: max pelvis heights0.123/0.141m, with target errors up to3.107/4.015rad. Nominal1508 also failed. This supports a policy/constraint deadlock to investigate, not proof of a particular collision-mesh defect.
- No high-altitude ejection was reproduced in these 64-state centered reruns. This does not negate an observed event in a different state/parameter realization. Original rapid out-of-bounds reset can also produce apparent jumps/teleports.

## Next work

1. Re-evaluate full holdout with centered XY, keep original report intact.
2. Inspect rows1508,1990 for trapped body pairs; rows880,2965 for fall-impact/unstable recovery.
3. Before changing physics or reward weights, collect per-physics-step ground-only impulse, penetration, controller targets and drive measurements around problematic impacts. Use frozen/held-target and policy-driven comparisons to separate initial-state contact problems from controller-driven ones.
4. No training source, collision filters, rewards or checkpoints were modified by this review.
