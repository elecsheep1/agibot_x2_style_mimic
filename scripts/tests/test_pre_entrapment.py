"""Dynamic states must retain velocities and cannot inherit holdout parents."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.fallen_bank import save_bank, QUALITY_COLUMNS, sha256
from common.pre_entrapment import save, load


class DynamicBankTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.parent = Path(self.tmp.name) / 'parent.npz'
        self.output = Path(self.tmp.name) / 'dynamic.npz'
        rows = [dict(root_pose=np.array([0.,0.,.1,0.,0.,0.,1.]), root_velocity=np.zeros(6), joint_pos=np.array([.1,.2]), joint_vel=np.zeros(2), source_type=np.int64(1), motion_id=np.int64(-1), family_id=np.int64(i+10), split=np.int64(i), root_gravity_b=np.array([1.,0.,0.]), quality=np.zeros(len(QUALITY_COLUMNS))) for i in range(2)]
        save_bank(self.parent, rows, ['a','b'], {})
        self.row = dict(root_pose=rows[0]['root_pose'].copy(), root_velocity=np.array([.2,0.,0.,0.,0.,.3]), joint_pos=np.array([.1,.2]), joint_vel=np.array([.4,.5]), parent_row=np.int64(0), family_id=np.int64(10), split=np.int64(0), time_s=np.float64(.5), lead_seconds=np.float64(.15), trigger_pair=np.array([0,1]), ground_min_separation_m=np.float64(-.001), self_min_separation_m=np.float64(-.001))
        self.meta = dict(parent_bank=str(self.parent), parent_sha256=sha256(self.parent), quat_order='xyzw', velocity_frame='world_axes', root_frame='link_origin', position_frame='environment_local', replay_validated=True, reset_protocol='clear_action_and_observation_history_after_state_write_v1')

    def test_roundtrip_and_joint_reorder_preserve_velocities(self):
        save(self.output, [self.row], ['a','b'], self.meta)
        data = load(self.output, self.parent, ['b','a'])
        np.testing.assert_allclose(data['root_velocity'][0], self.row['root_velocity'])
        np.testing.assert_allclose(data['joint_vel'][0], [.5,.4])
        with self.assertRaises(FileExistsError): save(self.output, [self.row], ['a','b'], self.meta)

    def test_holdout_parent_rejected_even_when_labelled_train(self):
        self.row.update(parent_row=np.int64(1), family_id=np.int64(11))
        with self.assertRaisesRegex(ValueError, 'Holdout'): save(self.output, [self.row], ['a','b'], self.meta)

    def test_lineage_and_finite_guards(self):
        self.row['family_id'] = np.int64(99)
        with self.assertRaisesRegex(ValueError, 'lineage'): save(self.output, [self.row], ['a','b'], self.meta)
        self.row['family_id'] = np.int64(10); self.row['joint_vel'][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'): save(self.output, [self.row], ['a','b'], self.meta)

    def test_old_reset_protocol_rejected(self):
        self.meta.pop('reset_protocol')
        save(self.output, [self.row], ['a','b'], self.meta)
        with self.assertRaisesRegex(ValueError, 'reset protocol'):
            load(self.output, self.parent)

    def test_wrong_parent_and_unvalidated_rejected(self):
        save(self.output, [self.row], ['a','b'], self.meta)
        other = Path(self.tmp.name) / 'other'; other.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'parent'): load(self.output, other)
        self.output.unlink(); self.meta['replay_validated'] = False
        save(self.output, [self.row], ['a','b'], self.meta)
        with self.assertRaisesRegex(ValueError, 'validation'): load(self.output, self.parent)


if __name__ == '__main__': unittest.main()
