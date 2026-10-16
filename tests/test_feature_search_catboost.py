"""Revision 141: native categoricals, fold-local CV and option guards."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from JBGFeatureSearch import SearchOptions, make_pipeline, run_experiment


class CatBoostInvestigationTests(unittest.TestCase):
    def test_clone_native_categories_and_missing_values(self):
        x = pd.DataFrame({'category': ['a', 'b', None, 'c'] * 12,
                          'amount': [1., np.nan, 2., 3.] * 12})
        y = pd.Series(['no', 'yes'] * 24)
        for weighting in ('None', 'Balanced'):
            options = SearchOptions(model_type='CatBoost', catboost_iterations=8,
                                    catboost_class_weight=weighting)
            model = clone(make_pipeline(('category', 'amount'), options,
                                        {'text_columns': ['category']}))
            model.fit(x, y)
            self.assertEqual(model.predict(x).shape, (48,))
            self.assertEqual(model.predict_proba(x).shape, (48, 2))
            self.assertEqual(model.named_steps['columns'].categorical_columns_, ('category',))

    def test_reject_incompatible_processing(self):
        for kwargs in ({'oversampling': 'SME'}, {'reduction': 'PCA'},
                       {'catboost_class_weight': 'unknown'}, {'catboost_iterations': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(SearchOptions(model_type='CatBoost'), **kwargs).validate()

    def test_small_nested_cv(self):
        rng = np.random.default_rng(42)
        n = 90
        x = pd.DataFrame({'category': np.where(np.arange(n) % 3 == 0, 'a', 'b'),
                          'amount': rng.normal(size=n)})
        y = pd.Series(np.where(np.arange(n) % 4 == 0, 'yes', 'no'))
        options = SearchOptions(model_type='CatBoost', catboost_iterations=8,
                                trials=1, sizes=(1,), inner_folds=2, outer_folds=2,
                                refine_rounds=0, max_evaluations=2)
        with tempfile.TemporaryDirectory() as folder:
            result = run_experiment(x, y, 'yes', Path(folder) / 'out', options,
                                    {'text_columns': ['category']}, progress=None)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['options']['model_type'], 'CatBoost')
