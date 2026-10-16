"""X2 行走任务的共享参数。"""

from typing import Final

COMMAND_NAME: Final = "base_velocity"
FOOT_BODY_NAMES: Final = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
FOOT_CONTACT_FORCE_THRESHOLD: Final = 5.0  # N