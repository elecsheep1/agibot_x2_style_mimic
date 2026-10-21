""" 多动作重置配置。"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.configclass import configclass

from .mdp import getup as getup_mdp
from .x2_getup_env_cfg import PROJECT_DIR
from .x2_style_mimic_env_cfg import POLICY_JOINT_NAMES

MOTION_MANIFEST = (
    PROJECT_DIR
    / "data"
    / "motions"
    / "prone_getup"
    / "motions.json"
)

@configclass
class MultiMotionGetUpEventsCfg:
    reset_from_reference = EventTerm(
        func=getup_mdp.reset_from_motion_dataset,
        mode="reset",
        params={
            "manifest_file": str(MOTION_MANIFEST),
            "policy_joint_names": list(POLICY_JOINT_NAMES),
            "start_phase": 0.0,
            "end_phase": 0.0,
            "use_reference_velocity": False,
            "height_offset": 0.01
        },
    )