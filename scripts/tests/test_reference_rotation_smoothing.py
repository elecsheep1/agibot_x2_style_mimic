"""Geometric regressions for large-angle offline reference smoothing."""
import sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repair.repair_x2_reference_temporal import smooth_rotation_trajectory

class RotationSmoothingTests(unittest.TestCase):
    def test_crossing_pi_has_no_spurious_spin(self):
        source=Rotation.from_euler('x',np.linspace(165,195,61)[:,None],degrees=True)
        result=smooth_rotation_trajectory(source,3.)
        steps=(result[1:]*result[:-1].inv()).magnitude()
        self.assertLess(float(steps.max()),np.radians(.6))
        self.assertLess(float((result*source.inv()).magnitude().max()),np.radians(1.))

    def test_equivalent_antipodal_quaternions(self):
        source=Rotation.from_euler('xyz',np.column_stack([np.linspace(0,1,51),np.linspace(.2,.5,51),np.linspace(2.8,3.5,51)]))
        q=source.as_quat();q[::2]*=-1
        a=smooth_rotation_trajectory(source,2.)
        b=smooth_rotation_trajectory(Rotation.from_quat(q),2.)
        self.assertLess(float((a*b.inv()).magnitude().max()),1e-12)

    def test_world_rotation_does_not_change_smoothing(self):
        source=Rotation.from_euler('zyx',np.column_stack([np.linspace(2.7,3.6,51),np.linspace(.2,.5,51),np.linspace(-.3,.1,51)]))
        frame=Rotation.from_euler('xyz',[.8,-.4,.7])
        expected=frame*smooth_rotation_trajectory(source,2.)
        result=smooth_rotation_trajectory(frame*source,2.)
        self.assertLess(float((expected*result.inv()).magnitude().max()),1e-12)

if __name__=='__main__':unittest.main()
