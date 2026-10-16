"""Revision 143: pooled out-of-fold subgroup reporting."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from JBGFeatureSearch import SearchOptions, run_experiment, subgroup_probability_diagnostics

class SubgroupTests(unittest.TestCase):
    def test_insufficient_and_single_class(self):
        y = pd.Series(['yes', 'no', 'yes', 'no', 'no', 'no'])
        scores = [0.8, 0.2, 0.7, 0.1, 0.3, 0.4]
        values = pd.Series(['A', 'A', 'A', 'A', 'B', 'B'])
        out = subgroup_probability_diagnostics(y, scores, 'yes', values, minimum_positives=2)
        self.assertEqual(out[0]['status'], 'ok')
        self.assertEqual(out[0]['roc_auc'], 1.0)
        self.assertEqual(out[1]['status'], 'undefined_single_class')
        self.assertIsNone(out[1]['roc_auc'])

    def test_oof_subgroup_without_extra_fits(self):
        rng = np.random.default_rng(11)
        X = pd.DataFrame({'amount': rng.normal(size=90), 'fund': [0] * 45 + [1] * 45})
        y = pd.Series(np.where(X['amount'] > 0, 'yes', 'no'))
        options = SearchOptions(model_type='Linear', inner_folds=3, outer_folds=3,
                                trials=2, max_evaluations=3, sizes=(1,))
        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiment(X, y, 'yes', Path(tmp) / 'study',
                        options=options, text_options={'text_columns': ()},
                        investigation_mode='diagnostics_only', subgroup_column='fund', progress=None)
            report = result['diagnostics']['full_baseline']
            self.assertEqual(result['attempted_fits'], 3)
            self.assertEqual(report['subgroup_column'], 'fund')
            self.assertEqual(sum(r['rows'] for r in report['subgroups']), 90)
            self.assertEqual(len(report['subgroups']), 2)
            self.assertNotIn('recommendation', result)

    def test_bad_column_rejected(self):
        X = pd.DataFrame({'amount': range(30)})
        y = pd.Series(['yes', 'no'] * 15)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Unknown subgroup column'):
                run_experiment(X, y, 'yes', Path(tmp)/'run',
                               investigation_mode='diagnostics_only', subgroup_column='KassaNr')
