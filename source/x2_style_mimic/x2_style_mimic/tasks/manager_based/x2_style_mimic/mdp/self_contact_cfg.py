from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils.configclass import configclass


def _create_self_contact_sensor(cfg):
    # 场景创建发生在 SimulationApp 启动之后。
    from .self_contact import SelfContactSensor

    return SelfContactSensor(cfg)


@configclass
class SelfContactSensorCfg(SensorBaseCfg):
    class_type = _create_self_contact_sensor

    # 低阈值接触诊断，不参与奖励。
    force_threshold: float = 10.0
    duration_start: float = 0.1

    contact_capacity_min: int = 262144
    contacts_per_sensor: int = 8
    validate_contact_buffer: bool = True

    # 以下参数用于持续强接触惩罚。
    pressure_force_threshold: float = 150.0
    pressure_release_threshold: float = 100.0
    pressure_release_rate: float = 0.5
    pressure_force_scale: float = 300.0
    pressure_grace_seconds: float = 0.05
    pressure_ramp_seconds: float = 0.20

    # 短促强冲击：按控制周期内每个配对的峰值计算。
    impact_force_threshold: float = 800.0
    impact_force_scale: float = 800.0
    impact_weight: float = 1.0
