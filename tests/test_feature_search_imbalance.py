import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from JBGFeatureSearch import SearchOptions, Search, make_estimator, make_pipeline, investigative_predictions, ranking

class ImbalanceTests(unittest.TestCase):
    def test_class_weights_and_defaults(self):
        for family in ('Extra Trees', 'Random Forest'):
            for weight in ('None', 'balanced', 'balanced_subsample'):
                opts = SearchOptions(model_type='Tree', tree_estimator=family, tree_class_weight=weight)
                estimator = make_estimator(opts)[1]
                self.assertEqual(estimator.class_weight, None if weight == 'None' else weight)
        with self.assertRaises(ValueError):
            SearchOptions(tree_class_weight='oops').validate()
        with self.assertRaises(ValueError):
            SearchOptions(tree_positive_threshold=1.0).validate()

    def test_binary_cutoff_predictions(self):
        values, labels = make_classification(n_samples=100, n_features=5, n_informative=3,
            weights=[0.85, 0.15], random_state=42)
        X = pd.DataFrame(values, columns=[f'f{i}' for i in range(5)])
        y = pd.Series(np.where(labels == 1, 'Inkorrekt', 'Korrekt'))
        opts = SearchOptions(model_type='Tree', tree_estimator='Random Forest',
            tree_n_estimators=20, tree_positive_threshold=0.2)
        pipe = make_pipeline(tuple(X.columns), opts, None)
        pipe.fit(X, y)
        pred = investigative_predictions(pipe, X, 'Inkorrekt', opts)
        probabilities = pipe.predict_proba(X)[:, list(pipe.classes_).index('Inkorrekt')]
        self.assertTrue(np.array_equal(pred == 'Inkorrekt', probabilities >= 0.2))
        self.assertGreaterEqual((pred == 'Inkorrekt').sum(), (pipe.predict(X) == 'Inkorrekt').sum())

    def test_zero_mcc_baseline_wins(self):
        baseline = dict(stage='full_baseline', mean_mcc=0., std_mcc=0., n_columns=99, columns=('a',))
        subset = dict(stage='random', mean_mcc=0., std_mcc=0., n_columns=9, columns=('b',))
        self.assertLess(ranking(baseline), ranking(subset))
        subset['mean_mcc'] = 0.001
        self.assertLess(ranking(subset), ranking(baseline))

if __name__ == '__main__': unittest.main()
