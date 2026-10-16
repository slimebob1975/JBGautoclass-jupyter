"""Revision 142: out-of-fold probability diagnostics."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from JBGFeatureSearch import SearchOptions, run_experiment, probability_diagnostics


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = pd.DataFrame({'amount': rng.normal(size=75),
                               'category': np.where(rng.random(75) > .5, 'A', 'B')})
        self.y = pd.Series(np.where(self.X.amount > 0, 'yes', 'no'))
        self.options = SearchOptions(model_type='CatBoost', catboost_iterations=5,
                                     inner_folds=3, outer_folds=3, trials=2,
                                     max_evaluations=3, sizes=(1,),
                                     top_sets=1, refine_rounds=0)
        self.text_options = {'text_columns': ('category',)}

    def test_diagnostics_only_does_not_produce_recommendation(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_experiment(self.X, self.y, 'yes', Path(temp) / 'run',
                                    options=self.options, text_options=self.text_options,
                                    investigation_mode='diagnostics_only', progress=None)
            self.assertEqual(result['status'], 'completed')
            self.assertNotIn('recommendation', result)
            self.assertEqual(result['attempted_fits'], 3)
            report = result['diagnostics']['full_baseline']
            self.assertGreater(report['roc_auc'], .9)
            self.assertEqual(len(report['threshold_analysis']), 9)
            self.assertFalse((Path(temp) / 'run' / 'selected_features.json').exists())

    def test_diagnostics_only_progress_events_are_gui_compatible(self):
        events = []
        with tempfile.TemporaryDirectory() as temp:
            result = run_experiment(self.X, self.y, 'yes', Path(temp) / 'run',
                                    options=self.options, text_options=self.text_options,
                                    investigation_mode='diagnostics_only', progress=None,
                                    on_event=events.append)
            self.assertEqual(result['status'], 'completed')
            self.assertNotIn('progress_reporting_error', result)
            starts = [e for e in events if e['event'] == 'fit_started']
            self.assertEqual(len(starts), self.options.outer_folds)
            self.assertTrue(all(e['model_type'] == 'CatBoost' for e in starts))
            self.assertEqual([e['attempted_fits'] for e in starts], [1, 2, 3])

    def test_combined_mode_produces_both_oof_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_experiment(self.X, self.y, 'yes', Path(temp) / 'run',
                                    options=self.options, text_options=self.text_options,
                                    investigation_mode='selection_plus_diagnostics', progress=None)
            self.assertEqual(set(result['diagnostics']), {'selected', 'full_baseline'})
            self.assertIn('recommendation', result)

    def test_thresholds_are_descriptive_and_scores_finite(self):
        report = probability_diagnostics(pd.Series(['yes', 'no', 'yes', 'no']),
                                         [.9, .1, .8, .2], 'yes')
        self.assertEqual(report['roc_auc'], 1.0)
        self.assertEqual(report['average_precision'], 1.0)
        self.assertEqual(report['threshold_analysis'][0]['tp'], 2)
        with self.assertRaises(ValueError):
            probability_diagnostics(pd.Series(['yes', 'no']), [float('nan'), .2], 'yes')


if __name__ == '__main__':
    unittest.main()
