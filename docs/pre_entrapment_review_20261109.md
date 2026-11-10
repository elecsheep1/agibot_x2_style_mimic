# Dynamic recovery collection and core logic review — 2026-11-09

## Scope and permissions

No source file was edited in this task. Source changes require the user's explicit confirmation.
Reviewed active reset/bank loading, action limits, standing/limb and self-contact reward paths,
terminal capture, assistance settings and AMP history/gating. This is a targeted core-path review,
not a claim that every algorithm or physical failure in the repository has been verified.
All 123 Python files under source/scripts parsed successfully at review time.
The initial scripts/tests discovery ran 36 tests successfully and had one module import error:
reference rotation smoothing imports an unavailable optional `clarabel` dependency.
Five new dynamic-bank tests pass separately. The 9 self-contact tests are included in the 36.

## Findings

- Current self_contact reward reads the per-control-period mean of the physics-step force-duration cost.
  It does not also charge the old history count. The reward function is read-only.
- Weak-contact diagnostic duration and strong-contact reward duration have separate thresholds.
  Both survive control boundaries and are cleared on per-environment reset.
- Contact views merge A/B and B/A once, and filter to robot bodies in the same environment.
  Existing physical collision filters are unchanged.
- Reward scale changed despite the same -0.1 weight: one saturated pair contributes -0.002 per
  20 ms control step, whereas the previous four-sample count could contribute -0.008 per step.
  Two saturated pairs contribute -0.004. This is intentional formula replacement but does not
  automatically make the penalty stronger; compare per-failure reward traces before adjusting weight.
- Strict strong-contact continuity is a remaining tuning limitation: a sample below 150 N clears that
  pair's timer. Pulsed squeezing may therefore escape this penalty; inspect traces before tuning.
- Bank states are centered in XY and their link-origin world-axis velocities are restored before
  world env translation. `_write_reference_state` compensates root-origin/COM velocity offset.
- Pair-force magnitude is calculated from the contact matrix vector, not a sum of absolute
  pointwise normal forces. Opposing contact points on one body pair can cancel in that vector;
  this monitor is not a complete measure of internal squeezing load. No sensor formula changed.
- Existing settled-bank train/holdout family isolation and asset/reference fingerprint checks remain.
- AMP terminal state capture precedes auto-reset, and AMP windows accumulate genuine post-reset
  states rather than injected reference history. Existing phase/standing gates remain.
- Reference finite-difference velocities versus instantaneous simulator velocities remain different;
  this work does not change that observation source mismatch.
- Ordinary ground/head contact sensor lazy sampling remains a separate known limitation; adding a
  self-contact duration reward does not fix impact recording or guarantee no ground rebound.
- `probe_self_contact_factory.py` is a legacy experimental shim with obsolete cfg fields and a
  standalone import incompatible with current relative imports. Use the production sensor and
  new collector, not this old diagnostic entry point.

## New script-only implementation

- scripts/checks/collect_pre_entrapment.py
- scripts/common/pre_entrapment.py
- scripts/tests/test_pre_entrapment.py

The collector selects only existing TRAIN rows, keeps parent family IDs and a SHA256 parent-bank
fingerprint, runs model_12400 with zero assist, and maintains a bounded trajectory ring.
The first same-pair force >150 N lasting >=0.10 s triggers candidate extraction at approximately
0.30 s before, 0.15 s before, and at the trigger. A 0.10 s event is an onset candidate, not proof
of persistent entrapment. Insufficient history and physically invalid candidates are skipped.
Stored root velocities and all joint velocities are real sampled values; they are not zeroed.
Candidate checks include finite arrays, hard joint position tolerance 0.003 rad, velocity limits,
penetration <=5 mm, root linear speed <=1.5 m/s and root angular speed <=6 rad/s.
The dynamic schema is deliberately different from the old settled-bank schema.

Every exported candidate is restored through the production write helper and compared against
saved pose/velocities/joints (1e-4 tolerance), with fresh actions/observations/contact timers.
It is replayed for 0.20 s with the old policy and current DR. Nonfinite values, termination,
root speed >3 m/s, angular speed >12 rad/s, current ground force >=6 kN, current self force >=3 kN,
or sampled penetration >15 mm reject it. These are screening values, not proven hardware limits.
Final replay validation observes every real 2 ms physics substep and asserts the expected update count.
No success requirement is used: this avoids discarding difficult but reachable recovery states.
A short replay cannot prove eventual recoverability or absence of every transient impulse.

An exploratory smoke output was explicitly invalidated and renamed smoke.exploratory.npz because
contact path warnings needed investigation. Do not use it. canonical.npz is the subsequent
production-source-view run: 128 train parents -> 5 candidates -> 4 replay-accepted states from
2 families. One candidate failed replay. The sample is too small to enable a 5% training branch.
A larger 1024-parent/256-env collection is run separately as train_candidates.npz.
Original bank/holdout is never rewritten. Reports mark fewer than 32 accepted families as not
ready for source integration; this is an initial diversity guard, not statistical proof.

## Proposed source integration — approval required, NOT applied

1. mdp/getup_births.py: add a separate DynamicRecoveryBank loader. Validate the dynamic schema,
   parent SHA256, current robot asset fingerprint, original reference identities, train-only parent
   lineage, joint names, finite state arrays and replay flag. Sample families uniformly first, then
   one of that family's stages. Do not relax FallenStateBank's settled-bank coverage requirements.
2. mdp/getup.py: optional `recovery_state_bank=None`, `recovery_probability=0.0`. Within the existing
   simulated-fall branch only, replace a configured fraction with dynamic states. Restore all real
   velocities without added height; enforce existing hard-limit tolerance. Add a separate recovery
   row ID and birth type 4; never claim a dynamic row is an original static-bank row.
3. x2_getup_env.py: capture recovery row IDs before auto-reset and log type 4 as `pre_entrapment`.
   Preserve existing terminal-state and assistance accounting.
4. x2_getup_env_cfg.py: keep the new branch disabled by default until a sufficiently diverse bank
   passes mixed-reset smoke checks. A dedicated zero-assist continuation entry can use conditional
   probability 0.5 within the 10% simulated-fall branch, yielding 30/10/5/5/50 overall.
5. Script-side mixed-reset/evaluation: validate dynamic states against their own bank, label and
   report them separately, preserve the original 722-state holdout for the final comparison.

Do not merge dynamic rows into the old settled bank or overwrite original holdout rows. Do not
use previously diagnosed heldout trajectories (1508/1990 etc.) as training material.

## Final measured result

1024 train parents scanned -> 45 candidates from 21 families -> 43 passed the preliminary replay.
The final replay clears action limiter state and observation history AFTER writing the saved state,
and checks every 2 ms physical substep. It accepts 37 states from 19 families and rejects 6 of the
43 preliminary states. Only train_validated.npz has the final reset-protocol marker; the default
loader rejects the earlier intermediate banks. All 37 states retain nonzero root and joint velocities.
Original parent-bank SHA256 is unchanged. No source files changed during this task.
The diversity guard remains false (<32 families); do not enable a full training branch automatically.
Source integration requires explicit user approval.

Final data summary:
```json
{
  "count": 37,
  "families": 19,
  "all_train": true,
  "lead_seconds_counts": {
    "0.0": 11,
    "0.16": 16,
    "0.3": 10
  },
  "nonzero_root_velocity_rows": 37,
  "nonzero_joint_velocity_rows": 37,
  "max_replay_ground_force_n": 4682.41748046875,
  "max_replay_self_force_n": 2947.578857421875
}
```
