"""Collect transitions using an exported X2 get-up policy and optional Gaussian action noise.

Uses the same simulator checks and data format as collect_supermimic_smoke.py.
"""

from collect_supermimic_smoke import main


if __name__ == "__main__":
    main(policy_mode=True)
