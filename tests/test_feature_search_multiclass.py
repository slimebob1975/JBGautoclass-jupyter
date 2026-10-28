"""Regression coverage for multiclass investigations introduced in revision 132."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

import JBGFeatureSearch as fs


class MulticlassSearchTests(unittest.TestCase):
    def test_multiclass_nested_cv_and_pooled_diagnostics(self):
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        X.index = np.arange(1, len(X) + 1)
        y = pd.Series([iris.target_names[i] for i in iris.target], index=X.index)
        opts = fs.SearchOptions(trials=2, sizes=(1, 2), max_evaluations=3,
                                refine_rounds=0, inner_folds=3, outer_folds=3)
        with tempfile.TemporaryDirectory() as directory:
            result = fs.run_experiment(X, y, 'virginica', Path(directory) / 'iris',
                                       options=opts, progress=None)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['evaluation']['type'], 'multiclass')
            # Includes six fold-local probability-diagnostic fits (2 panels x 3 folds).
            self.assertEqual(result['attempted_fits'], 48)
            self.assertEqual(set(result['nested_validation']), {'selected', 'full_baseline'})
            for name in ('selected', 'full_baseline'):
                metrics = result['nested_validation'][name]
                matrix = np.array(metrics['confusion_matrix'])
                self.assertEqual(matrix.shape, (3, 3))
                self.assertEqual(int(matrix.sum()), 150)
                self.assertEqual(metrics['target_support'], 50)
                self.assertAlmostEqual(metrics['macro_recall'], np.mean([
                    matrix[i, i] / matrix[i].sum() for i in range(3)]))
                self.assertEqual(set(metrics['per_class']), set(iris.target_names))
            self.assertEqual(result['recommendation']['validation_status'], 'candidate')
            self.assertTrue((Path(directory) / 'iris' / 'outer_validation.csv').exists())

    def test_binary_diagnostics_backward_compatible_and_class_fold_check(self):
        X = pd.DataFrame({'x': range(12)})
        y = pd.Series(['a'] * 6 + ['b'] * 6)
        self.assertEqual(len(fs.folds_for(X, y, 3, 42)), 3)
        with self.assertRaisesRegex(ValueError, 'at least 3 rows'):
            fs.folds_for(X, pd.Series(['a'] * 9 + ['b'] * 2 + ['c']), 3, 42)
        self.assertEqual(fs.metrics(fs.classification_counts(['a', 'b'], ['a', 'a'], 'a'))['tp'], 1)
        self.assertEqual(fs.multiclass_diagnostics(['a','b','c'], ['a','b','c'], ['a','b','c'])['macro_f1'], 1.)


if __name__ == '__main__':
    unittest.main()
