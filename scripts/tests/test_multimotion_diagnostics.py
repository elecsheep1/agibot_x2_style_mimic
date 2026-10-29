"""Replay diagnostics: terminal snapshots, interruption attribution and quotas."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rsl_rl/multimotion'))
import diagnostics as d


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        params = dict(hold_time=3., minimum_height=.5, maximum_tilt=.35,
                      maximum_linear_speed=.2, maximum_angular_speed=.5,
                      maximum_knee_angle=1.5, force_threshold=5.)
        self.env = SimpleNamespace(
            num_envs=1, step_dt=.02, _getup_reference_motion_paths=['test.npz'],
            termination_manager=SimpleNamespace(compute=lambda: None,
                get_term_cfg=lambda _: SimpleNamespace(params=params)),
            reward_manager=SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(weight=20.,params={})),
            reset_terminated=torch.tensor([False]), reset_time_outs=torch.tensor([True]),
            cfg=SimpleNamespace(events=SimpleNamespace(reset_from_reference=SimpleNamespace(
                params=dict(start_phase=0.,end_phase=0.)))))
        self.diag = d.MultiMotionDiagnostics(self.env, 1, Path(self.temp.name)/'report.json')

    def step(self, failed=(), hold=.02, done=False):
        masks = {k: torch.tensor([k not in failed]) for k in d.CONDITION_LABELS}
        with patch.object(d, 'standing_conditions', return_value=masks), patch.object(d, 'pelvis_pose', return_value=torch.tensor([[0., 25., 25., .6]])), patch.object(d, 'standing_reward_comparison', return_value=torch.tensor([[.95, .99, .04]])):
            self.env.termination_manager.compute()
        progress = dict(assist_episode_force_n=torch.tensor([0.]),
                        assist_applied_force_n=torch.tensor([0.]),
                        motion_ids=torch.tensor([0]), succeeded=torch.tensor([False]),
                        first_success_time=torch.tensor([-1.]),
                        max_hold_seconds=torch.tensor([.04]),hold_seconds=torch.tensor([hold]))
        self.diag.update(torch.tensor([done]), dict(getup_progress=progress))

    def test_terminal_failures_are_counted_and_multiple_causes_preserved(self):
        self.step(hold=.02)
        self.step(hold=.04)
        self.step(failed=('linear_speed','feet_contact'),hold=0.,done=True)
        r=self.diag.summarize()['overall']['standing_diagnostics']
        self.assertEqual(r['steps'],3)
        self.assertEqual(r['interruptions'],1)
        self.assertEqual(r['interruption_causes']['linear_speed'],1)
        self.assertEqual(r['interruption_causes']['feet_contact'],1)
        self.assertEqual(r['height_qualified_steps'],3)
        self.assertAlmostEqual(r['all_pass_percent'],200/3)
        self.diag.report()
        self.assertTrue(self.diag.output_path.exists())
        row=self.diag.rows[0]
        self.assertAlmostEqual(row['pelvis_trace'][-1]['time_s'],.06)
        self.assertEqual(row['pelvis_last_5s']['pitch_deg']['mean'],25.)
        self.assertEqual(self.diag._pose_traces[0],[])
        self.assertAlmostEqual(row['pelvis_last_5s']['root_up1_gain_raw']['mean'],.04,places=6)
        self.step(failed=('height',),hold=0.,done=True)
        self.assertEqual(self.diag.summarize()['overall']['standing_diagnostics']['steps'],3)

    def test_floor_frames_do_not_dilute_height_qualified_rates(self):
        self.step(failed=('height','pelvis_tilt'),hold=0.)
        self.step(hold=.02,done=True)
        r=self.diag.summarize()['overall']['standing_diagnostics']
        self.assertEqual(r['pass_percent']['pelvis_tilt'],50.)
        self.assertEqual(r['pass_percent_when_high']['pelvis_tilt'],100.)
        self.assertEqual(r['interruptions'],0)

    def test_timer_disagreement_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'disagree'):
            self.step(failed=('knees',),hold=.02)


if __name__=='__main__':unittest.main()
