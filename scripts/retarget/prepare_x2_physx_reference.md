# PhysX reference kinematics

Run from the project root in `env_isaaclab_6_0`:

```bash
python scripts/retarget/prepare_x2_physx_reference.py \
  --input-dir data/motions/prone_getup2 \
  --output-dir data/motions/prone_getup2_physx \
  --num-envs 32 --headless
```

The output directory must differ from the source directory. Existing outputs
are rejected unless `--overwrite` is supplied; original inputs are never
overwritten, even with that flag. For a single file use `--motion PATH`; for
only the active manifest entries use `--manifest PATH` instead of `--input-dir`.

The script differentiates the full root/joint trajectories, writes reference
states into a separate IsaacLab/PhysX scene, and refreshes FK with
`sim.forward()`. It never calls `sim.step()`, creates no RL environment, and
does not load AMP or policy modules. All joints are mapped by name. Root origin
velocity is explicitly converted to COM velocity for the PhysX state writer.
Body origin velocity is calculated from world-frame link Jacobian joint
columns plus root rigid motion, and checked against PhysX's body origin velocity
before any output is saved.

Added fields:

- `body_lin_vel_w`: `[T, B, 3]`, world-frame link-origin velocity [m/s].
- `body_ang_vel_w`: `[T, B, 3]`, world-frame angular velocity [rad/s].

The generated copy also updates `body_pos_w`, `body_quat_xyzw`, differentiated
root/joint velocities, root-local velocities, projected gravity, and policy-joint
velocity fields to keep them consistent. Root positions, joint positions, time
samples, name order and ground convention are preserved. Metadata records the
source hash, root USD hash, numerical method, frame counts and validation errors.
`report.json` summarizes each clip. With `--input-dir`, an existing
`motions.json` is copied with active entries and weights preserved, even when
other inactive NPZs are also converted.

`--max-frames N` is a preview only: derivatives still use the full trajectory,
the output metadata is marked partial, and no training manifest is generated.
Use a separate output directory for previews.

Training does not automatically consume these fields yet. Its reference loader
must explicitly read the generated world velocities, then apply the existing
pelvis-relative transformation. Do not recompute body velocities by position
differences in that loader, which would discard this preprocessing.

Regenerate when the reference trajectory, sampling, or robot kinematic structure
changes. This is a kinematics cache, not a contact or dynamic reachability test.

CPU-only numerical checks:

```bash
python -m unittest discover -s scripts/tests \
  -p test_physx_reference_preprocessing.py -v
```
