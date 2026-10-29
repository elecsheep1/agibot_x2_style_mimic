"""参考转换的坐标系、时间轴与名称映射检查，无需启动 Isaac Lab。"""

from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from retarget.convert_x2_reference import convert_reference
from common.x2_motion_utils import joint_indices, load_motion, policy_joint_names


class ReferenceConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "reference.pkl"
        self.policy = policy_joint_names()
        self.names = ["head_pitch_joint"] + list(reversed(self.policy)) + ["head_yaw_joint"]
        times = np.arange(7) / 30
        # 经过 180°，且等价四元数符号交替；正确结果仍应是恒定正向角速度。
        quats = Rotation.from_euler("z", (170 + 150*times)[:, None], degrees=True).as_quat()
        quats[1::2] *= -1
        joints = np.tile(np.arange(31)*.001, (7, 1))
        joints[:, self.names.index("left_wrist_pitch_joint")] = times
        self.raw = {
            "fps": 30., "quat_order": "xyzw", "start_frame_30hz": 0,
            "joint_names": self.names, "dof_pos": joints,
            "root_pos": np.c_[.2*times, np.zeros((7, 2))], "root_rot": quats,
        }
        self.source.write_bytes(pickle.dumps(self.raw))

    def convert(self):
        return convert_reference(self.source, 50., self.policy, None)

    def test_known_linear_and_angular_velocity(self):
        p = self.convert()
        np.testing.assert_allclose(p["root_lin_vel_w"], np.tile([.2, 0, 0], (11, 1)), atol=1e-6)
        np.testing.assert_allclose(p["root_ang_vel_w"], np.tile([0, 0, np.radians(150)], (11, 1)), atol=1e-6)
        np.testing.assert_allclose(p["projected_gravity_b"], np.tile([0, 0, -1], (11, 1)), atol=1e-6)
        orientation = Rotation.from_quat(p["root_quat_xyzw"])
        expected_body_velocity = orientation.inv().apply(np.tile([.2, 0, 0], (11, 1)))
        np.testing.assert_allclose(p["root_lin_vel_b"], expected_body_velocity, atol=1e-6)
        np.testing.assert_allclose(p["policy_joint_vel"][:, self.policy.index("left_wrist_pitch_joint")], 1, atol=1e-6)

    def test_resampling_keeps_duration_and_shared_frames(self):
        p = self.convert()
        np.testing.assert_allclose(p["time_s"], np.arange(11)*.02)
        np.testing.assert_allclose(p["joint_pos"][[0, 5, 10]], self.raw["dof_pos"][[0, 3, 6]], atol=1e-6)
        self.assertTrue(np.all(np.sum(p["root_quat_xyzw"][:-1] * p["root_quat_xyzw"][1:], axis=1) >= 0))

    def test_policy_mapping_is_name_based_and_npz_needs_no_pickle(self):
        p = self.convert()
        for i, name in enumerate(self.policy):
            np.testing.assert_array_equal(p["policy_joint_pos"][:, i], p["joint_pos"][:, self.names.index(name)])
        path = Path(self.temp.name) / "motion.npz"
        np.savez_compressed(path, **p)
        loaded = load_motion(path)
        self.assertEqual(loaded["metadata"]["root_frame"], "pelvis")
        self.assertEqual(loaded["policy_joint_names"].tolist(), self.policy)

    def test_invalid_mapping_or_quaternion_is_rejected(self):
        with self.assertRaises(ValueError):
            joint_indices(["a", "a"], ["a"])
        with self.assertRaises(ValueError):
            joint_indices(["a"], ["b"])
        self.raw["root_rot"][0] *= 2
        self.source.write_bytes(pickle.dumps(self.raw))
        with self.assertRaises(ValueError):
            self.convert()


if __name__ == "__main__":
    unittest.main()
