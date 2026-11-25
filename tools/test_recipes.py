import contextlib
import io
import json
import tomllib
import unittest
from subprocess import CompletedProcess
from pathlib import Path
from unittest.mock import patch

from recipes import CARDS, planner_args, run_planner, startup, toml

ROOT = Path(__file__).resolve().parents[1]

# config name -> (card, desktop, drafts, vision, streams)
SHIPPED = {
    '16gb': ('16', False, True, False, 1),
    '16gb-long': ('16', False, False, False, 1),
    '16gb-desktop': ('16', True, False, False, 1),
    '24gb': ('24', False, True, False, 2),
    '24gb-vision': ('24', False, True, True, 2),
    '24gb-desktop': ('24', True, True, False, 2),
    '24gb-desktop-vision': ('24', True, True, True, 2),
    '32gb': ('32', False, True, False, 4),
    '32gb-vision': ('32', False, True, True, 4),
    '32gb-desktop': ('32', True, True, False, 4),
    '32gb-desktop-vision': ('32', True, True, True, 4),
    'dgx-spark': ('spark', False, True, False, 8),
    'dgx-spark-vision': ('spark', False, True, True, 8),
    'dgx-spark-desktop': ('spark', True, True, False, 8),
    'dgx-spark-desktop-vision': ('spark', True, True, True, 8),
}


class RecipeTest(unittest.TestCase):
    def report(self, name):
        return json.loads((ROOT / 'data/planner' / f'{name}.json').read_text())

    def test_shipped_configs_match_recorded_planner_reports(self):
        for name, (card, desktop, drafts, vision, streams) in SHIPPED.items():
            r = self.report(name)
            c = CARDS[card]
            self.assertEqual(r['model'], c['branch'], name)
            self.assertEqual(r['drafts'], drafts, name)
            self.assertEqual(r['drafter'], 'DFlash2' if drafts else None, name)
            self.assertEqual(r['parallel'], streams, name)
            self.assertEqual(r['recipe']['vision'], vision, name)
            self.assertEqual(r['kv_cache'], c['kv'], name)
            expected = toml(r, card, desktop, c['reserve_gib'], c['branch'], c['kv'])
            config = (ROOT / 'configs' / f'{name}.toml').read_text()
            self.assertEqual(config, expected, name)
            self.assertNotIn('prompt-cache-gib', tomllib.loads(config), name)

    def test_recorded_reports_sit_on_the_admission_boundary(self):
        for name in SHIPPED:
            r = self.report(name)
            self.assertTrue(r['result']['admitted'], name)
            if r['model_cap_binds']:
                self.assertIsNone(r['next_context'], name)
            else:
                self.assertFalse(r['next_context']['admitted'], name)
                self.assertEqual(r['next_context']['context'], r['maximum'] + 1, name)
            self.assertLessEqual(r['rounded_down'], r['maximum'])
            self.assertEqual(r['rounded_down'] % 1024, 0)
            chosen = r['recipe']
            self.assertTrue(chosen['result']['admitted'], name)
            self.assertEqual(chosen['context'], chosen['result']['context'], name)
            self.assertLessEqual(chosen['context'], r['rounded_down'], name)
            result = chosen['result']
            pool = result['pool_tokens']
            self.assertGreaterEqual(pool, chosen['context'] + 16, name)
            self.assertEqual(pool % 256, 0, name)
            self.assertEqual(result['fixed_bytes_per_stream'], result['tracked']['per_stream']['bytes'], name)
            self.assertEqual(result['components']['target_global_kv']['bytes'],
                             pool * result['components']['global_kv_per_token']['bytes'], name)

    def test_planner_arguments(self):
        self.assertEqual(planner_args('16', True, 1, False, 'fp8', 0.5),
                         ['--device-mib', '16303', '--parallel', '1', '--kv-cache', 'fp8',
                          '--desktop-reserve-gib', '1.0', '--reserve-gib', '0.5', '--no-drafts'])
        self.assertEqual(planner_args('spark', True, 8, True, 'fp8', None),
                         ['--device-mib', '124608', '--parallel', '8', '--kv-cache', 'fp8',
                          '--unified', '--compute-capability', '121', '--free-mib', '118256'])
        self.assertEqual(planner_args('24', False, 2, True, 'fp8', None, './draft', True),
                         ['--device-mib', '24564', '--parallel', '2', '--kv-cache', 'fp8',
                          '--drafter', './draft', '--vision'])

    def test_optional_prompt_cap_in_plan_and_toml(self):
        args = planner_args('32', False, 4, True, 'fp8', None, prompt_cache_gib=0)
        self.assertEqual(args[-2:], ['--prompt-cache-gib', '0'])
        report = {'rounded_down': 32768, 'parallel': 4, 'drafts': True,
                  'recipe': {'context': 32768, 'prompt_cache_gib': 1.5}}
        self.assertEqual(tomllib.loads(toml(report, '32', False, None, '4.0bpw', 'fp8'))['prompt-cache-gib'], 1.5)

    def test_planner_requires_shared_pool_retention_limits_for_admitted_results(self):
        valid = {'pages': 64, 'bytes': 4096, 'marks': 8}
        disabled = {'pages': 0, 'bytes': 0, 'marks': 0}
        for pool, retention in (
            (True, valid), (True, disabled), (False, valid), (True, None),
            (True, {'pages': 64}), (True, {**valid, 'marks': -1}),
        ):
            with self.subTest(pool=pool, retention=retention):
                result = {'admitted': True, 'prefix_retention': retention}
                if pool:
                    result['pool_tokens'] = 16384
                cmd = ['planner']
                output = CompletedProcess(cmd, 0, json.dumps({'result': result}), '')
                with patch('recipes.subprocess.run', return_value=output):
                    if pool and retention in (valid, disabled):
                        self.assertEqual(run_planner(cmd)['result'], result)
                    else:
                        with self.assertRaises(SystemExit):
                            run_planner(cmd)

    def test_startup_uses_the_fixed_context_report_when_available(self):
        result = {'admitted': True, 'context': 32768, 'pool_tokens': 131072}
        output = io.StringIO()
        with patch('recipes.run_planner', return_value={'result': result}) as run, contextlib.redirect_stdout(output):
            startup(['32768', 'planner', '--prompt-cache-gib', '0'])
        run.assert_called_once_with(['planner', '--prompt-cache-gib', '0'], 32768)
        self.assertEqual(output.getvalue(), '32768 131072\n')

    def test_startup_falls_back_to_the_automatic_plan_on_refusal(self):
        refused = {'result': {'admitted': False, 'message': 'context exceeds capacity'}}
        automatic = {'maximum': 16368, 'result': {'admitted': True, 'context': 16368, 'pool_tokens': 16384}}
        output = io.StringIO()
        with patch('recipes.run_planner', side_effect=[refused, automatic]) as run, contextlib.redirect_stdout(output):
            startup(['32768', 'planner'])
        self.assertEqual([call.args for call in run.call_args_list], [(['planner'], 32768), (['planner'],)])
        self.assertEqual(output.getvalue(), '16368 16384\n')

    def test_startup_refuses_explicit_zero_when_the_full_window_does_not_fit(self):
        refused = {'result': {'admitted': False, 'message': 'full model window exceeds capacity'}}
        with patch('recipes.run_planner', return_value=refused) as run:
            with self.assertRaisesRegex(SystemExit, 'full model window exceeds capacity'):
                startup(['0', 'planner'])
        run.assert_called_once_with(['planner'], 0)

    def test_startup_uses_automatic_sizing_when_context_is_omitted(self):
        automatic = {'maximum': 16368, 'result': {'admitted': True, 'context': 16368, 'pool_tokens': 16384}}
        with patch('recipes.run_planner', return_value=automatic) as run, contextlib.redirect_stdout(io.StringIO()) as output:
            startup(['', 'planner'])
        run.assert_called_once_with(['planner'])
        self.assertEqual(output.getvalue(), '16368 16384\n')

    def test_card_context_cache_and_vision(self):
        for name, (card, _, _, vision, _) in SHIPPED.items():
            recipe = self.report(name)['recipe']
            self.assertGreaterEqual(recipe['context'], 8192, name)
            retention = recipe['result']['prefix_retention']
            self.assertEqual(retention['pages'] * 256, recipe['result']['pool_tokens'], name)
            self.assertEqual(retention['bytes'], recipe['result']['components']['target_global_kv']['bytes'], name)
            self.assertGreater(retention['marks'], 0, name)
            self.assertNotIn('prompt_cache_gib', recipe, name)
            if card == '16':
                self.assertFalse(recipe['vision'], name)
            if vision:
                text_plan = self.report(name.removesuffix('-vision'))
                self.assertLess(recipe['result']['pool_tokens'], text_plan['recipe']['result']['pool_tokens'], name)
                self.assertEqual(self.report(name)['parallel'], text_plan['parallel'], name)


if __name__ == '__main__':
    unittest.main()
