"""Numerical preprocessing checks; no IsaacLab application is started."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

MODULE = Path(__file__).resolve().parents[1] / "retarget/prepare_x2_physx_reference.py"
spec = importlib.util.spec_from_file_location("physx_reference_preprocessing", MODULE)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class PreprocessingTests(unittest.TestCase):
    def trajectory(self):
        times = np.arange(11, dtype=np.float64) * .02
        angle = .7 * times
        quaternion = np.c_[np.zeros((len(times), 2)), np.sin(angle / 2), np.cos(angle / 2)]
        return dict(time_s=times, root_pos_w=np.c_[2 * times, -times, 0 * times],
                    root_quat_xyzw=quaternion, joint_pos=np.c_[3 * times, -2 * times])

    def test_analytic_translation_rotation_and_joint_speeds(self):
        motion = self.trajectory()
        # Alternate quaternion sign: the physical angular velocity must not jump.
        motion["root_quat_xyzw"][1::2] *= -1
        _, root, joint = tool.differentiated_state(motion)
        np.testing.assert_allclose(root[:, :3], np.tile([2., -1., 0.], (11, 1)), atol=1e-6)
        np.testing.assert_allclose(root[:, 3:], np.tile([0., 0., .7], (11, 1)), atol=1e-6)
        np.testing.assert_allclose(joint, np.tile([3., -2.], (11, 1)), atol=1e-6)

    def test_reject_nonuniform_time(self):
        motion = self.trajectory()
        motion["time_s"][3] += .001
        with self.assertRaisesRegex(ValueError, "uniformly"):
            tool.differentiated_state(motion)

    def test_world_local_vector_round_trip(self):
        q = torch.tensor([[0., 0., 2 ** -.5, 2 ** -.5]], dtype=torch.float64)
        v = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
        world = tool.rotate(q, v)
        torch.testing.assert_close(world, torch.tensor([[0., 1., 0.]], dtype=torch.float64))
        torch.testing.assert_close(tool.inverse_rotate(q, world), v)

    def test_never_overwrite_source_even_with_overwrite_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reference.npz"
            source.touch()
            args = SimpleNamespace(motion=source, manifest=None, input_dir=None,
                                   output_dir=Path(directory), overwrite=True)
            with self.assertRaisesRegex(ValueError, "different"):
                tool.sources(args)


if __name__ == "__main__":
    unittest.main()
