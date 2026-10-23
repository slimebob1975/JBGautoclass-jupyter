"""Revision 138: positive-control Combination/PCA and assisted approval safeguards."""
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from JBGFeatureSearch import SearchOptions, make_estimator, make_pipeline

ROOT = Path(__file__).parents[1]


class CombinationTests(unittest.TestCase):
    def setUp(self):
        self.base = SearchOptions(trials=1, sizes=(2,), inner_folds=2,
            outer_folds=2, max_evaluations=2)

    def test_existing_futv_membership_and_weights(self):
        name, estimator = make_estimator(replace(self.base, model_type='Combination'))
        self.assertEqual(name, 'futv')
        self.assertIsInstance(estimator, VotingClassifier)
        self.assertEqual(estimator.voting, 'soft')
        self.assertEqual(estimator.weights, (1, 2, 1))
        self.assertEqual([name for name, _ in estimator.estimators], ['mlpc', 'rfcl', 'abc'])
        self.assertEqual(estimator.estimators[0][1].max_iter, 20000)
        self.assertEqual(estimator.estimators[0][1].random_state, 42)
        clone(estimator)

    def test_pca_after_fold_only_sampling(self):
        opt = replace(self.base, model_type='Combination', reduction='PCA')
        pipeline = make_pipeline(('a', 'b', 'c', 'd'), opt, {})
        self.assertEqual([k for k, _ in pipeline.steps],
            ['columns', 'impute', 'scale', 'pca', 'futv'])
        self.assertIsInstance(pipeline.named_steps['pca'], PCA)
        self.assertEqual(pipeline.named_steps['pca'].n_components, .95)
        self.assertIsNone(getattr(pipeline.named_steps['pca'], 'components_', None))
        clone(pipeline)

    def test_combination_pca_can_fit_predict(self):
        X, y = make_classification(n_samples=75, n_features=4, n_informative=3,
                                   n_redundant=0, random_state=42)
        X = pd.DataFrame(X, columns=list('abcd'))
        opt = replace(self.base, model_type='Combination', reduction='PCA')
        model = make_pipeline(tuple(X.columns), opt, {})
        with threadpool_limits(limits=1):
            model.fit(X.iloc[:60], y[:60])
            predictions = model.predict(X.iloc[60:])
        self.assertEqual(len(predictions), 15)
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_validation_and_gui_apply_contract(self):
        replace(self.base, reduction='NOR').validate()
        replace(self.base, reduction='PCA').validate()
        with self.assertRaises(ValueError):
            replace(self.base, reduction='SVD').validate()
        source = (ROOT / 'src/GUI/Widgets.py').read_text(encoding='utf-8')
        self.assertIn('if self.feature_selection_mode.value in ("assisted", "reuse")', source)
        self.assertIn('self._feature_search_applied = True', source)
        self.assertIn('self._feature_search_applied = False', source)
        self.assertIn('self.widgets._feature_search_applied = False', source)
        self.assertIn('"feature_search_reduction"', source)
        config_source = (ROOT / 'src/Config.py').read_text(encoding='utf-8')
        self.assertIn('feature_search_reduction: str = "NOR"', config_source)
        self.assertIn('JSON_CONFIG_SCHEMA_VERSION = 2', config_source)


if __name__ == '__main__':
    unittest.main()
