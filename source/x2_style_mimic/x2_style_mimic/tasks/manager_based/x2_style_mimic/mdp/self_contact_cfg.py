from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils.configclass import configclass


def _create_self_contact_sensor(cfg):
    # 场景创建发生在 SimulationApp 启动之后。
    from .self_contact import SelfContactSensor

    return SelfContactSensor(cfg)


@configclass
class SelfContactSensorCfg(SensorBaseCfg):
    class_type = _create_self_contact_sensor

    force_threshold: float = 10.0
    history_length: int = 4

    # 仅用于持续接触诊断，不参与奖励门控。
    duration_start: float = 0.1

    contact_capacity_min: int = 262144
    contacts_per_sensor: int = 8
    validate_contact_buffer: bool = True
