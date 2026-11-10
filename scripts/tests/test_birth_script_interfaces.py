"""Prevent mixed-birth API changes from silently breaking script reset callbacks."""
import ast
import inspect
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from common.fallen_bank import reset_from_bank_for_eval
MDP = ROOT / 'source/x2_style_mimic/x2_style_mimic/tasks/manager_based/x2_style_mimic'


class BirthInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ast.parse((MDP / 'x2_getup_env_cfg.py').read_text())
        events = next(n for n in self.cfg.body if isinstance(n, ast.ClassDef) and n.name == 'GetUpEventsCfg')
        term = next(n.value for n in events.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'reset_from_reference' for t in n.targets))
        params = next(k.value for k in term.keywords if k.arg == 'params')
        self.keys = {ast.literal_eval(k) for k in params.keys}

    def test_config_matches_both_production_and_fixed_bank_callbacks(self):
        tree = ast.parse((MDP / 'mdp/getup.py').read_text())
        reset = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reset_from_motion_dataset')
        self.assertLessEqual(self.keys, {arg.arg for arg in reset.args.args})
        self.assertLessEqual(self.keys, set(inspect.signature(reset_from_bank_for_eval).parameters))
        self.assertNotIn('recovery_probability', self.keys)
        self.assertNotIn('perturb_initial_probability', self.keys)

    def test_play_entries_explicitly_select_reference_initial(self):
        for name in ('play.py', 'play_transition.py'):
            tree = ast.parse((ROOT / 'scripts/rsl_rl/multimotion' / name).read_text())
            values = [k.value for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'reset_params' and n.func.attr == 'update' for k in n.keywords if k.arg == 'birth_probabilities']
            self.assertEqual(len(values), 1, name)
            self.assertEqual(ast.literal_eval(values[0]), (1., 0., 0., 0., 0.))

    def test_fixed_bank_initializes_metadata_with_reference_only(self):
        tree = ast.parse(inspect.getsource(reset_from_bank_for_eval))
        values = [k.value for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords if k.arg == 'birth_probabilities']
        self.assertEqual(ast.literal_eval(values[0]), (1., 0., 0., 0., 0.))

    def test_transition_entry_does_not_override_assist_course(self):
        tree = ast.parse((ROOT / 'scripts/rsl_rl/multimotion/train_transition.py').read_text())
        forbidden = {'assist_initial_stage', 'assist_zero_probability', 'assist_curriculum_enabled'}
        assignments = [t.attr for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Attribute) and t.attr in forbidden]
        self.assertEqual(assignments, [])


if __name__ == '__main__': unittest.main()
