"""Fallen-bank compatibility, split leakage and corruption guards."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.fallen_bank import load_bank, save_bank, family_split, dedup_key, QUALITY_COLUMNS


class FallenBankTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'bank.npz'
        self.rows=[]
        for i in range(2):
            self.rows.append(dict(root_pose=np.array([0,0,.1,0,0,0,1.]),root_velocity=np.zeros(6),
                joint_pos=np.array([.1,.2]),joint_vel=np.array([.01,.02]),source_type=np.int64(i),
                motion_id=np.int64(0 if i==0 else -1),family_id=np.int64(i),split=np.int64(i),
                root_gravity_b=np.array([1.,0,0]),quality=np.zeros(len(QUALITY_COLUMNS))))

    def test_joint_mapping_and_round_trip(self):
        save_bank(self.path,self.rows,['a','b'],{})
        x=load_bank(self.path,joint_names=['b','a'])
        np.testing.assert_allclose(x['joint_pos'][0],[.2,.1])
        np.testing.assert_allclose(x['joint_vel'][0],[.02,.01])
        with self.assertRaises(FileExistsError):save_bank(self.path,self.rows,['a','b'],{})

    def test_reject_family_leakage(self):
        self.rows[1]['family_id']=np.int64(0)
        with self.assertRaisesRegex(ValueError,'Family leakage'):
            save_bank(self.path,self.rows,['a','b'],{})

    def test_reject_invalid_quaternion_and_fake_fall_motion(self):
        self.rows[0]['root_pose'][3:]=0
        with self.assertRaisesRegex(ValueError,'quaternion'):
            save_bank(self.path,self.rows,['a','b'],{})
        self.path.unlink();self.rows[0]['root_pose'][6]=1
        self.rows[1]['motion_id']=np.int64(2)
        with self.assertRaisesRegex(ValueError,'Real falls'):
            save_bank(self.path,self.rows,['a','b'],{})

    def test_split_determinism_and_yaw_independent_fingerprint(self):
        self.assertEqual(family_split(42,12,.2),family_split(42,12,.2))
        self.assertEqual(dedup_key([.03,.06],[1,0,0],.1),dedup_key([.031,.061],[1,0,0],.101))
        counts=sum(family_split(42,i,.2) for i in range(1000))
        self.assertTrue(150<counts<250)


if __name__=='__main__':unittest.main()
