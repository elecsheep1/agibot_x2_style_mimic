"""Test deferred sensor construction without modifying user production modules."""
from pathlib import Path
import importlib.util
import runpy
import sys
import types

from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils.configclass import configclass

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic/mdp/self_contact.py'


def create_sensor(cfg):
    name='_self_contact_runtime_probe'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,SOURCE)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name].SelfContactSensor(cfg)


@configclass
class ProbeSelfContactCfg(SensorBaseCfg):
    class_type= create_sensor
    force_threshold: float=50.
    force_scale: float=150.
    duration_start: float=.1
    duration_full: float=.3
    penalty_cap: float=3.
    contact_capacity_min: int=262144
    contacts_per_sensor: int=8
    validate_contact_buffer: bool=True


if __name__=='__main__':
    # Isolate only the early configuration import. The real implementation
    # is imported unmodified by create_sensor after SimulationApp starts.
    shim=types.ModuleType('x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.self_contact')
    shim.SelfContactSensorCfg=ProbeSelfContactCfg
    sys.modules[shim.__name__]=shim
    sys.path.insert(0,str(ROOT/'scripts'))
    runpy.run_path(str(ROOT/'scripts/checks/smoke_fallen_training_reset.py'),run_name='__main__')
