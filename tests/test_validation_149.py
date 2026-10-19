import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from JBGValidation149 import group_folds, validation_folds, parse_excluded_columns


class Validation149Tests(unittest.TestCase):
    def setUp(self):
        self.groups = np.repeat(np.arange(30), 4)
        self.labels = np.tile([0, 0, 1, 1], 30)
        self.X = pd.DataFrame({'amount': np.arange(120), 'person': self.groups})

    def test_group_folds_disjoint_and_complete(self):
        folds = group_folds(self.X, self.labels, self.groups, n_splits=5)
        seen = []
        for a, b in folds:
            self.assertFalse(set(self.groups[a]) & set(self.groups[b]))
            self.assertEqual(set(self.labels[b]), {0, 1})
            seen.extend(b.tolist())
        self.assertEqual(sorted(seen), list(range(len(self.X))))

    def test_stable(self):
        a = group_folds(self.X, self.labels, self.groups, n_splits=5)
        b = group_folds(self.X, self.labels, self.groups, n_splits=5)
        self.assertTrue(all(np.array_equal(x[1], y[1]) for x, y in zip(a, b)))

    def test_default_is_stratified_kfold(self):
        cv = validation_folds(self.X, self.labels, n_splits=5)
        self.assertEqual(cv.get_n_splits(), 5)

    def test_missing_group_fails(self):
        with self.assertRaises(ValueError):
            group_folds(self.X, self.labels, self.groups[:5], n_splits=5)

    def test_exclusion_parser(self):
        self.assertEqual(parse_excluded_columns(' iafId,PaymentId, iafId '), ('iafId', 'PaymentId'))


if __name__ == '__main__':
    unittest.main()
