"""Checks the descriptive feature-error statistics, without simulator dependencies."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from checks.analyze_amp_feature_errors import BLOCKS, groups, mismatch, nearest_match, score_stats


class FeatureErrorTests(unittest.TestCase):
    def test_layout_and_candidate_indices(self):
        joints=[str(i) for i in range(29)]
        bodies=['torso_link','left_shoulder','right_shoulder','left_wrist','right_wrist',
                'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
        g=groups(joints,bodies)
        self.assertEqual(sorted(sum(g['blocks'].values(),[])),list(range(233)))
        self.assertEqual(len(g['candidate_groups']['intermediate_body_rotation6d_36']),36)
        self.assertEqual(len(g['candidate_groups']['torso_wrists_ankles_rotation6d_30']),30)
        self.assertEqual(sorted(sum(g['bodies'].values(),[])),list(range(68,233)))

    def test_matching_whole_window_and_error_shares(self):
        ref=torch.zeros(2,5,233);ref[1,:,0]=10
        w=torch.zeros(1,5,233);w[:,:,0]=9;w[:,:,1]=2
        std=torch.ones(233);std[1]=2
        match=nearest_match(w,ref,std,'cpu')
        self.assertEqual(match.item(),1)
        result=mismatch(w,ref,match,std,torch.tensor([True]),
                        {'blocks':{k:list(range(a,b)) for k,(a,b) in BLOCKS.items()}})
        self.assertAlmostEqual(result['normalized_rms'],(2/233)**.5)
        shares={r['name']:r['normalized_sse_share_percent'] for r in result['blocks']}
        self.assertAlmostEqual(shares['root_height'],50)
        self.assertAlmostEqual(shares['root_gravity'],50)

    def test_zero_reward_uses_valid_selected_windows(self):
        result=score_stats(torch.tensor([-1.1,-1.,0.,1.]),torch.tensor([True,True,True,False]))
        self.assertEqual(result['count'],3)
        self.assertAlmostEqual(result['zero_reward_fraction'],2/3,places=6)
        self.assertAlmostEqual(result['reward_mean'],.25)
        self.assertEqual(score_stats(torch.ones(1),torch.tensor([False])),{'count':0})

if __name__=='__main__':unittest.main()
