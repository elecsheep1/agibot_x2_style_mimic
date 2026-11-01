"""Checks the descriptive feature-error statistics, without simulator dependencies."""
import sys
import tempfile
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from checks.analyze_amp_feature_errors import BLOCKS, groups, mismatch, nearest_match, score_stats, sensitivity, checkpoint_normalization


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

    def test_sensitivity_reproduces_old_and_clipped_checkpoint_scores(self):
        # Force an extreme feature so missing clipping cannot silently pass.
        for running in (False, True):
            with self.subTest(running=running), tempfile.TemporaryDirectory() as directory:
                net = torch.nn.Sequential(torch.nn.Linear(1165,256),torch.nn.ELU(),
                    torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,1))
                with torch.no_grad():
                    for parameter in net.parameters(): parameter.zero_()
                    net[0].weight[0,0]=1
                    net[2].weight[0,0]=1
                    net[4].weight[0,0]=1
                state={'network.'+k:v for k,v in net.state_dict().items()}
                state.update(state_mean=torch.zeros(233),state_std=torch.ones(233))
                settings={}
                if running:
                    state.update(state_count=torch.tensor(100.),state_var=torch.ones(233))
                    settings['normalization_clip']=10.
                checkpoint=Path(directory)/'model.pt'
                torch.save({'amp_window':dict(discriminator=state,settings=settings)},checkpoint)
                windows=torch.zeros(1,5,233);windows[0,0,0]=100
                data=dict(checkpoint=str(checkpoint),windows=windows,
                    state_mean=state['state_mean'],state_std=state['state_std'],
                    score=torch.tensor([10. if running else 100.]))
                gs={k:{} for k in ['blocks','bodies','joints','body_components','candidate_groups']}
                gs['blocks']={'height':[0]}
                result=sensitivity(data,torch.zeros_like(windows),torch.tensor([0]),
                    gs,torch.tensor([True]),'cpu')
                self.assertAlmostEqual(result['blocks'][0]['score_change_mean'],
                    -10. if running else -100.)

    def test_running_checkpoint_requires_explicit_clip_setting(self):
        with self.assertRaisesRegex(ValueError, 'missing normalization_clip'):
            checkpoint_normalization(dict(discriminator={'state_count':torch.tensor(1.)},settings={}))

if __name__=='__main__':unittest.main()
