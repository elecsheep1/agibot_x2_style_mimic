"""独立 X2 参考跟踪环境；不继承旧起身配置。"""
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp as base_mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import (CommandTermCfg, ObservationGroupCfg, ObservationTermCfg,
                               RewardTermCfg, TerminationTermCfg)
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass
from x2_style_mimic.assets.x2_ultra_cfg import X2_ULTRA_CFG
from x2_style_mimic.algorithms.supermimic_tracking.rewards import TrackingRewardCfg
from . import mdp

CONTROL_JOINT_NAMES = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint']
PROJECT = Path(__file__).resolve().parents[6]


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path='/World/ground',spawn=sim_utils.GroundPlaneCfg(size=(100.,100.)))
    robot = X2_ULTRA_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    light = AssetBaseCfg(prim_path='/World/Light',spawn=sim_utils.DomeLightCfg(intensity=1000.))


@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type = mdp.TrackingCommand
    resampling_time_range = (1.e9,1.e9)
    debug_vis = False
    future_frames: int = 10
    bin_frames: int = 50
    sampling_mode: str = 'return'
    gamma: float = .99
    ema_alpha: float = .05
    uniform_ratio: float = .5
    hard_ratio: float = .4
    memory_ratio: float = .1
    softmax_temp: float = 25.
    memory_alpha: float = .01
    min_remaining_steps: int = 100


@configclass
class CommandsCfg:
    motion = MotionCommandCfg()


@configclass
class NoisyActionCfg(JointPositionActionCfg):
    class_type = mdp.NoisyJointPositionAction
    noise_std: float = .1


@configclass
class ActionsCfg:
    joint_pos = NoisyActionCfg(asset_name='robot',joint_names=CONTROL_JOINT_NAMES,
                               preserve_order=True,scale=.25,use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.observation_part,params={'start':0,'end':29})
        joint_vel = ObservationTermCfg(func=mdp.observation_part,params={'start':29,'end':58})
        ang_vel = ObservationTermCfg(func=mdp.observation_part,params={'start':58,'end':61})
        gravity = ObservationTermCfg(func=mdp.observation_part,params={'start':61,'end':64})
        actions = ObservationTermCfg(func=mdp.observation_part,params={'start':64,'end':93})
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False  # 噪声在同一观测函数中应用，与想象侧同幅值。
            self.history_length = 10
            self.flatten_history_dim = True
    @configclass
    class GoalCfg(ObservationGroupCfg):
        reference = ObservationTermCfg(func=mdp.goal_observation)
        def __post_init__(self):
            self.concatenate_terms = True
    policy = PolicyCfg()
    goal = GoalCfg()


@configclass
class RewardsCfg:
    motion_global_root_pos = RewardTermCfg(func=mdp.reward_term,weight=1.,params={'name':'motion_global_root_pos'})
    motion_global_root_ori = RewardTermCfg(func=mdp.reward_term,weight=1.,params={'name':'motion_global_root_ori'})
    motion_body_pos = RewardTermCfg(func=mdp.reward_term,weight=1.,params={'name':'motion_body_pos'})
    motion_body_ori = RewardTermCfg(func=mdp.reward_term,weight=1.,params={'name':'motion_body_ori'})
    motion_body_lin_vel = RewardTermCfg(func=mdp.reward_term,weight=1.,params={'name':'motion_body_lin_vel'})
    motion_body_ang_vel = RewardTermCfg(func=mdp.reward_term,weight=1.,params={'name':'motion_body_ang_vel'})
    action_rate_l2 = RewardTermCfg(func=mdp.reward_term,weight=1.,params={'name':'action_rate_l2'})
    joint_limit = RewardTermCfg(func=mdp.reward_term,weight=1.,params={'name':'joint_limit'})


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=base_mdp.time_out,time_out=True)
    clip_end = TerminationTermCfg(func=mdp.clip_end,time_out=True)
    root_height = TerminationTermCfg(func=mdp.tracking_failure,params={'component':0})
    root_tilt = TerminationTermCfg(func=mdp.tracking_failure,params={'component':1})


@configclass
class X2SuperMimicTrackingEnvCfg(ManagerBasedRLEnvCfg):
    scene = TrackingSceneCfg(num_envs=4096,env_spacing=3.)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = None
    curriculum = None
    control_joint_names = CONTROL_JOINT_NAMES
    motion_manifest: str = str(PROJECT/'data/motions/prone_getup2/motions_birth.json')
    urdf: str = '/home/panda/robotics_ws/agibot_model/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra.urdf'
    tracking_reward = TrackingRewardCfg()
    domain_randomization: bool = True
    reset_noise: bool = True
    obs_noise: bool = True
    root_height_threshold: float = .25
    root_tilt_threshold: float = .8
    def __post_init__(self):
        self.decimation = 10
        self.sim.dt = .002
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.
        self.viewer.eye = (2.5,2.5,1.5)
        self.tracking_reward.dt = self.sim.dt*self.decimation
