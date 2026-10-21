from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass

@configclass
class LimitedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type | str = (
        "x2_style_mimic.tasks.manager_based.x2_style_mimic."
        "mdp.actions:LimitedJointPositionAction"
    )

    target_velocity_fraction: float = 0.5