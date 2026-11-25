import json
import unittest
from pathlib import Path
from unittest.mock import patch

from recipes import CARDS, GIB, cache_bytes, choose_cache, paired_streams, planner_args, toml

ROOT = Path(__file__).resolve().parents[1]

# config name -> (card, desktop, drafts, vision, streams)
SHIPPED = {
    '16gb': ('16', False, True, False, 1),
    '16gb-long': ('16', False, False, False, 1),
    '16gb-desktop': ('16', True, False, False, 1),
    '24gb': ('24', False, True, False, 2),
    '24gb-vision': ('24', False, True, True, 2),
    '24gb-desktop': ('24', True, True, False, 1),
    '24gb-desktop-vision': ('24', True, True, True, 1),
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
            self.assertEqual((ROOT / 'configs' / f'{name}.toml').read_text(), expected, name)

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
            self.assertLessEqual(chosen['prompt_cache_gib'] * GIB, cache_bytes(chosen['result']), name)

    def test_planner_arguments(self):
        self.assertEqual(planner_args('16', True, 1, False, 'fp8', 0.5),
                         ['--device-mib', '16303', '--parallel', '1', '--kv-cache', 'fp8',
                          '--desktop-reserve-gib', '1.0', '--reserve-gib', '0.5', '--no-drafts'])
        self.assertEqual(planner_args('spark', True, 8, True, 'fp8', None),
                         ['--device-mib', '124608', '--parallel', '8', '--kv-cache', 'fp8',
                          '--unified', '--free-mib', '118256'])
        self.assertEqual(planner_args('24', False, 2, True, 'fp8', None, './draft', True),
                         ['--device-mib', '24564', '--parallel', '2', '--kv-cache', 'fp8',
                          '--drafter', './draft', '--vision'])

    def test_prompt_cache_preserves_largest_context_with_two_gib(self):
        def result(context):
            budget = max(0, (80 * 1024 - context) * GIB // (4 * 1024))
            return {'context': context, 'rooms': {'prompt_cache_budget': {'bytes': budget}}}

        with patch('recipes.run_planner', side_effect=lambda _, context: {'result': result(context)}):
            context, gib, selected = choose_cache([], 80 * 1024, 2, result(80 * 1024))
        self.assertEqual((context, gib), (72 * 1024, 2))
        self.assertEqual(selected, result(context))

    def test_prompt_cache_disables_at_infeasible_context_floor(self):
        measurement = {'rooms': {'prompt_cache_budget': {'bytes': GIB}}}
        with patch('recipes.run_planner', return_value={'result': measurement}):
            self.assertEqual(choose_cache([], 64 * 1024, 2, measurement), (64 * 1024, 0, measurement))
            self.assertEqual(choose_cache([], 63 * 1024, 2, measurement), (63 * 1024, 0, measurement))

    def test_spark_cache_falls_back_to_largest_whole_gib_at_floor(self):
        def result(context):
            return {'rooms': {'prompt_cache_budget': {'bytes': (70 * 1024 - context) * GIB // 1024}}}

        with patch('recipes.run_planner', side_effect=lambda _, context: {'result': result(context)}):
            context, gib, _ = choose_cache([], 70 * 1024, 8, result(70 * 1024))
        self.assertEqual((context, gib), (64 * 1024, 6))

    def test_paired_streams_reduce_only_when_vision_cannot_keep_the_cache_floor(self):
        def at_floor(streams, context):
            self.assertEqual(context, 65536)
            return {'result': {'admitted': True, 'rooms': {'prompt_cache_budget': {'bytes': (5 - streams) * GIB}}}}

        with patch('recipes.run_planner', side_effect=at_floor):
            self.assertEqual(paired_streams(int, 4, 2), 3)
            self.assertEqual(paired_streams(int, 3, 8), 3)

    def test_card_context_cache_and_vision(self):
        for name, (card, _, _, vision, _) in SHIPPED.items():
            recipe = self.report(name)['recipe']
            if card == '16':
                self.assertGreaterEqual(recipe['context'], 32768, name)
                self.assertEqual(recipe['prompt_cache_gib'], 0, name)
                self.assertFalse(recipe['vision'], name)
            else:
                self.assertGreaterEqual(recipe['context'], 65536, name)
                self.assertEqual(recipe['prompt_cache_gib'], CARDS[card]['cache_gib'], name)
                if vision:
                    text_plan = self.report(name.removesuffix('-vision'))
                    self.assertLess(recipe['context'], text_plan['recipe']['context'], name)
                    self.assertEqual(self.report(name)['parallel'], text_plan['parallel'], name)


if __name__ == '__main__':
    unittest.main()
