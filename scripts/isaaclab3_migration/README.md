# Isaac Lab 3 migration helpers

These scripts keep Isaac Lab 3 compatibility work separate from the original
project launchers, which target the earlier development environment.

Run a short RSL-RL smoke test:

```bash
python scripts/isaaclab3_migration/train_x2_rsl_rl.py \
  --task Template-X2-Style-Mimic-v0 --num_envs 64 --max_iterations 3 --viz none
```

For `X2-Prone-MultiMotion-v0`, the launcher automatically supplies the Isaac
Lab 3-compatible self-contact root path. Do not add that override manually
unless you need a different source robot path.

Inspect the resolved X2 contact-sensor bodies. With `--viz kit`, close the Kit
window to stop; `--viz none` runs 100 zero-action simulation steps and exits.

```bash
python scripts/isaaclab3_migration/zero_agent_x2.py --viz kit
```
