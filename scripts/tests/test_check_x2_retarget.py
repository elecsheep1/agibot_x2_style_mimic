"""Regression cases with known defects, independent of the production motions."""
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks.check_x2_retarget import DEFAULTS, audit


class RetargetChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.xml = self.directory / 'robot.xml'
        self.xml.write_text('''<mujoco><worldbody><body name="pelvis" pos="0 0 1">
          <freejoint/><geom type="sphere" size=".08" mass="1"/>
          <body name="torso_link" pos="0 0 .2"><joint name="waist" range="-60 60"/>
            <geom type="sphere" size=".08" mass="1"/></body>
          <body name="left_ankle_roll_link" pos="0 .25 -.9">
            <joint name="left_ankle" range="-60 60"/>
            <geom type="sphere" size=".1" mass="1"/></body>
          <body name="right_ankle_roll_link" pos="0 -.25 -.9">
            <joint name="right_ankle" range="-60 60"/>
            <geom type="sphere" size=".1" mass="1"/></body>
        </body></worldbody></mujoco>''')
        model = mujoco.MjModel.from_xml_path(str(self.xml))
        self.q = np.tile(model.qpos0, (40, 1))

    def tearDown(self):
        self.temp.cleanup()

    def check_motion(self, q):
        path = self.directory / 'reference.npz'
        np.savez(path, qpos=q, fps=np.array([50.]), robot_xml=str(self.xml))
        return audit(path, dict(DEFAULTS))

    def test_clean_and_quaternion_sign_invariance(self):
        self.q[::2, 3:7] *= -1
        self.assertEqual(self.check_motion(self.q)['status'], 'pass')

    def test_teleport_jitter_and_joint_unreachable(self):
        self.q[20, 7] = 1.5
        result = self.check_motion(self.q)
        self.assertEqual(result['status'], 'fail')
        checks = {e['check'] for e in result['events']}
        self.assertIn('joint_limit_rad', checks)
        self.assertIn('joint_acceleration_rad_s2', checks)

    def test_floating_and_sliding(self):
        self.q[:, 2] += .1
        self.assertIn('inferred_unsupported_upright_m', {e['check'] for e in self.check_motion(self.q)['events']})
        self.q[:, 2] -= .1
        self.q[:, 0] = np.arange(40) * .02 * .2
        self.assertIn('sole_proxy_sliding_m_s', {e['check'] for e in self.check_motion(self.q)['events']})

    def test_ground_penetration(self):
        self.q[:, 2] -= .02
        result = self.check_motion(self.q)
        self.assertEqual(result['status'], 'fail')
        self.assertIn('ground_penetration_m', {e['check'] for e in result['events']})

    def test_bad_quaternion_is_rejected(self):
        self.q[0, 3:7] = 0
        with self.assertRaisesRegex(ValueError, 'unit length'):
            self.check_motion(self.q)

    def test_nonadjacent_collision(self):
        self.xml.write_text(self.xml.read_text().replace('0 -.25 -.9', '0 .26 -.9'))
        result = self.check_motion(self.q)
        self.assertEqual(result['status'], 'fail')
        self.assertIn('self_penetration_m', {e['check'] for e in result['events']})

    def test_contact_schedule_detects_single_floating_foot(self):
        self.xml.write_text(self.xml.read_text().replace('0 .25 -.9', '0 .25 -.8'))
        path = self.directory / 'reference.npz'
        np.savez(path, qpos=self.q, fps=np.array([50.]), robot_xml=str(self.xml))
        result = audit(path, dict(DEFAULTS), contacts={'left': [[0, .78]], 'right': [[0, .78]]})
        self.assertIn('scheduled_stance_floating_m', {e['check'] for e in result['events']})


if __name__ == '__main__':
    unittest.main()
