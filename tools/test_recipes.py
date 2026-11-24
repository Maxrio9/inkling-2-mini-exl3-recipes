import json
import unittest
from pathlib import Path

from recipes import CARDS, planner_args, toml

ROOT = Path(__file__).resolve().parents[1]

# config name -> (card, desktop, drafts)
SHIPPED = {
    '16gb': ('16', False, True),
    '16gb-long': ('16', False, False),
    '16gb-desktop': ('16', True, False),
    '24gb': ('24', False, True),
    '24gb-desktop': ('24', True, True),
    '32gb': ('32', False, True),
    '32gb-desktop': ('32', True, True),
    'dgx-spark': ('spark', False, True),
    'dgx-spark-desktop': ('spark', True, True),
}


class RecipeTest(unittest.TestCase):
    def report(self, name):
        return json.loads((ROOT / 'data/planner' / f'{name}.json').read_text())

    def test_shipped_configs_match_recorded_planner_reports(self):
        for name, (card, desktop, drafts) in SHIPPED.items():
            r = self.report(name)
            c = CARDS[card]
            self.assertEqual(r['model'], c['branch'], name)
            self.assertEqual(r['drafts'], drafts, name)
            self.assertEqual(r['parallel'], c['parallel'], name)
            expected = toml(r, card, desktop, c['reserve_gib'], c['branch'], 'fp8')
            self.assertEqual((ROOT / 'configs' / f'{name}.toml').read_text(), expected, name)

    def test_recorded_reports_sit_on_the_admission_boundary(self):
        for name in SHIPPED:
            r = self.report(name)
            self.assertTrue(r['result']['admitted'], name)
            self.assertFalse(r['next_context']['admitted'], name)
            self.assertEqual(r['next_context']['context'], r['maximum'] + 1, name)
            self.assertLessEqual(r['rounded_down'], r['maximum'])
            self.assertEqual(r['rounded_down'] % 1024, 0)

    def test_planner_arguments(self):
        self.assertEqual(planner_args('16', True, 1, False, 'fp8', 0.5),
                         ['--device-mib', '16303', '--parallel', '1', '--kv-cache', 'fp8',
                          '--desktop-reserve-gib', '1.0', '--reserve-gib', '0.5', '--no-drafts'])
        self.assertEqual(planner_args('spark', True, 8, True, 'fp8', None),
                         ['--device-mib', '124608', '--parallel', '8', '--kv-cache', 'fp8',
                          '--unified', '--free-mib', '118256'])

    def test_rows_meet_the_context_floor_or_are_documented(self):
        # 64k tokens per request without a desktop session; the 16 GB DSpark row cannot reach it at one request.
        below = {n for n, (card, desktop, _) in SHIPPED.items() if not desktop and self.report(n)['rounded_down'] < 65536}
        self.assertEqual(below, {'16gb'})


if __name__ == '__main__':
    unittest.main()
