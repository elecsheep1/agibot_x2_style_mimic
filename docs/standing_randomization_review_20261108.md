# Standing posture and domain randomization review — 2026-11-08

Scope: current X2 multi-motion training configuration, standing reward and diagnostics, reset bank, AMP history and training entry points. No production code changed during this review.

## Validation

- Parsed 54 Python files under project source and RSL-RL scripts successfully.
- Unit discovery: 33 tests passed; one module failed to import because this environment lacks `clarabel` (offline reference rotation repair). This is not a complete all-tests pass.
- Actual 64-environment scratch training completed two PPO/AMP iterations. Run: `logs/rsl_rl/x2_prone_multimotion/2026-11-08_20-42-08_smoke_standing_dr_review` (see actual run directory for authoritative timestamp).
- Actual 16-environment old-policy smoke completed 1100 control steps at zero assist with current physical randomization and actor noise. Report: `logs/reference_physics/standing_dr_review_mixed_smoke.json`. Completed episodes: 16; ever-success: 16. This small mixed-training-birth smoke is not full holdout acceptance and does not establish sim2sim readiness.
- Eight reset batches exercised all four branches: counts {'0': 30, '1': 16, '2': 69, '3': 13}. All 29 bank samples belonged to train and matched stored root/joint state and velocity exactly.
- Runtime environment spreads: body mass 1.8518734 kg, inertia maximum component spread 0.08758175, joint stiffness 15.5145264, damping 0.8976111. Parameters varied and were finite.
- Four hand/foot ground sensors each resolved exactly one expected rigid body. Startup warnings referred to child mesh paths without contact report API; matching on parent wrist/ankle bodies succeeded.
- Self-contact updated exactly 10 times per control step; raw history penalty remained within [0,4].

## Design limits retained

- Limb score uses height and upright gates; the optional velocity gate discussed earlier is NOT implemented. It can influence late get-up. At fraction 0.10, maximum raw reward reduction is 0.065; low-height branch is unchanged.
- Critic concatenates noisy policy observations and clean privileged observations. AMP observations remain uncorrupted.
- Startup material assignment is per collision shape; make_consistent means dynamic friction <= static friction. It does not mean equal friction across a robot. 65 buckets is legal.
- Existing play entry points inherit random physics and policy noise. Nominal acceptance requires explicitly disabling both; existing nominal baselines are not directly comparable.
- Action delay has not been implemented. MuJoCo deployment must reproduce policy ordering, XYZW conversion, 5-frame history, action offset/scale, target hard limits and rate limiter, PD and control periods.
- Randomization changes contact response; settled banks were originally validated under nominal dynamics. This smoke validates operation but not every bank state under every randomized parameter setting.

## Training

Use transition fine-tuning from the hip-filtered self-contact model_8500, 4000 additional iterations, std reset 0.30. That entry preserves actor/critic normalization and AMP weights, resets optimizers, and forces zero assist for bank-mixed births. Keep bank and reward weights fixed for this branch.

Unrelated deleted/migrated motion data and temporary bank files are excluded from this checkpoint.
