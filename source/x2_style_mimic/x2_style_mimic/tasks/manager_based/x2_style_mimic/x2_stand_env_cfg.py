"""Reference-centered standing task with posture checks and explicit fall resets."""
import math
import numpy as np

from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.envs import mdp

from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_getup_env_cfg import (
    PROJECT_DIR, GetUpRewardsCfg, GetUpTerminationsCfg, X2GetUpEnvCfg,
)
from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_style_mimic_env_cfg import POLICY_JOINT_NAMES


def stand_linear_speed_l2(env):
    return env.scene['robot'].data.root_lin_vel_w.torch.square().sum(dim=-1)


def stand_angular_speed_l2(env):
    return env.scene['robot'].data.root_ang_vel_w.torch.square().sum(dim=-1)


def stand_joint_speed_l2(env, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name].data.joint_vel.torch[:,asset_cfg.joint_ids].square().sum(dim=-1)


def reference_pose_error(env, reference_positions, asset_cfg: SceneEntityCfg):
    actual=env.scene[asset_cfg.name].data.joint_pos.torch[:,asset_cfg.joint_ids]
    reference=actual.new_tensor(reference_positions)
    return (actual-reference).square().sum(dim=-1)


def torso_up_cosine(env, asset_cfg: SceneEntityCfg):
    # Local API uses xyzw. World z component of the torso's local +Z axis.
    quat=env.scene[asset_cfg.name].data.body_link_pose_w.torch[:,asset_cfg.body_ids,3:7]
    return (1.-2.*quat[...,:2].square().sum(dim=-1)).mean(dim=-1)


def torso_tilt_cost(env, asset_cfg: SceneEntityCfg):
    return (1.-torso_up_cosine(env,asset_cfg)).clamp_min(0.)


def stand_failure(env, minimum_height, maximum_tilt, asset_cfg: SceneEntityCfg):
    robot=env.scene[asset_cfg.name]
    height=robot.data.root_link_pose_w.torch[:,2]-env.scene.env_origins[:,2]
    cosine=math.cos(maximum_tilt)
    return ((height<minimum_height)
            | (-robot.data.projected_gravity_b.torch[:,2]<cosine)
            | (torso_up_cosine(env,asset_cfg)<cosine))


@configclass
class StandRewardsCfg(GetUpRewardsCfg):
    success=None
    base_linear_speed=RewTerm(func=stand_linear_speed_l2,weight=-1.)
    base_angular_speed=RewTerm(func=stand_angular_speed_l2,weight=-.2)
    joint_speed=RewTerm(func=stand_joint_speed_l2,weight=-.002,
        params={'asset_cfg':SceneEntityCfg('robot',joint_names=POLICY_JOINT_NAMES,preserve_order=True)})
    action_rate=RewTerm(func=mdp.action_rate_l2,weight=-.01)
    reference_pose=RewTerm(func=reference_pose_error,weight=-.5,params={
        'reference_positions':[0.]*len(POLICY_JOINT_NAMES),
        'asset_cfg':SceneEntityCfg('robot',joint_names=POLICY_JOINT_NAMES,preserve_order=True)})
    waist_pose=RewTerm(func=reference_pose_error,weight=-2.,params={
        'reference_positions':[0.,0.,0.],
        'asset_cfg':SceneEntityCfg('robot',joint_names=['waist_yaw_joint','waist_pitch_joint','waist_roll_joint'],preserve_order=True)})
    torso_tilt=RewTerm(func=torso_tilt_cost,weight=-2.,
        params={'asset_cfg':SceneEntityCfg('robot',body_names=['torso_link'],preserve_order=True)})
    failure=RewTerm(func=mdp.is_terminated,weight=-50.)


@configclass
class StandTerminationsCfg(GetUpTerminationsCfg):
    success=None
    stand_failure=DoneTerm(func=stand_failure,params={
        'minimum_height':.52,'maximum_tilt':.65,
        'asset_cfg':SceneEntityCfg('robot',body_names=['torso_link'],preserve_order=True)})


@configclass
class X2StandEnvCfg(X2GetUpEnvCfg):
    rewards:StandRewardsCfg=StandRewardsCfg()
    terminations:StandTerminationsCfg=StandTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s=5.
        motion=PROJECT_DIR/'data/motions/x2_stand_width034_candidate.npz'
        with np.load(motion,allow_pickle=False) as f:
            reference=dict(zip(f['joint_names'].tolist(),map(float,f['joint_pos'][-1])))
            position=tuple(map(float,f['root_pos_w'][-1]))
            rotation=tuple(map(float,f['root_quat_xyzw'][-1]))
        # Align default pose, relative observations and zero-action targets.
        self.scene.robot.init_state.joint_pos=reference.copy()
        self.scene.robot.init_state.pos=position
        self.scene.robot.init_state.rot=rotation
        self.rewards.reference_pose.params['reference_positions']=[reference[n] for n in POLICY_JOINT_NAMES]
        self.rewards.waist_pose.params['reference_positions']=[reference[n] for n in ('waist_yaw_joint','waist_pitch_joint','waist_roll_joint')]
        self.events.reset_from_reference.params.update(motion_file=str(motion),
            random_frame_probability=1.,start_phase=1.,end_phase=1.)
