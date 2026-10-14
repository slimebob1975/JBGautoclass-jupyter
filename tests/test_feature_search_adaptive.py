"""Regression guards for the budget-aware Revision 134 subset search."""
import unittest
from pathlib import Path
import sys
import tempfile
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from JBGFeatureSearch import Search, SearchOptions, random_subsets


class AdaptiveSearchTests(unittest.TestCase):
    def test_random_proposals_reproducible_and_unique(self):
        opts = SearchOptions(trials=12, sizes=(2, 4, 6), max_evaluations=20)
        cols = tuple(f'x{i}' for i in range(12))
        a = random_subsets(cols, opts)
        self.assertEqual(a, random_subsets(cols, opts))
        self.assertEqual(len(a), 12)
        self.assertEqual(len(set(a)), len(a))
        self.assertEqual({len(s) for s in a}, {2, 4, 6})

    def test_budget_limited_refinement_remains_bounded(self):
        rng = np.random.default_rng(42)
        x = pd.DataFrame(rng.normal(size=(90, 15)), columns=[f'x{i}' for i in range(15)])
        y = pd.Series(np.where(x['x0'] + x['x1'] > 0, 'yes', 'no'))
        opts = SearchOptions(trials=2, sizes=(3,), inner_folds=3, outer_folds=3,
                             top_sets=1, refine_rounds=2, max_evaluations=7)
        events = []
        rows = []
        search = Search(x, y, 'yes', opts, {}, 'development', rows.append,
                        on_event=events.append)
        result = search.run()
        self.assertIsNotNone(result)
        self.assertLessEqual(len(rows), 7)
        self.assertTrue(any(e['event'] == 'refinement_budget_limited' for e in events))
        self.assertTrue(any(r['stage'] == 'refine' for r in rows))


if __name__ == '__main__':
    unittest.main()
