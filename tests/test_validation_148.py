import unittest
import numpy as np
import pandas as pd
from JBGValidation148 import person_disjoint_indices

class PersonDisjointSplitTests(unittest.TestCase):
    def test_repeated_people_never_cross_holdout(self):
        ids = np.repeat(np.arange(100), 3)
        X = pd.DataFrame({'iafId': ids, 'x': np.arange(300)})
        y = pd.Series(np.repeat(np.arange(100) % 2, 3))
        a, b = person_disjoint_indices(X, y)
        self.assertFalse(set(X.iloc[a].iafId) & set(X.iloc[b].iafId))
        self.assertEqual(set(y.iloc[a]), {0, 1})
        self.assertEqual(set(y.iloc[b]), {0, 1})
        self.assertEqual((list(a), list(b)), tuple(map(list, person_disjoint_indices(X, y))))

    def test_missing_group_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'iafId'):
            person_disjoint_indices(pd.DataFrame({'x': [1,2]}), pd.Series([0,1]))

    def test_missing_group_values_fails_closed(self):
        X = pd.DataFrame({'iafId': [1, None, 2, 3]})
        with self.assertRaisesRegex(ValueError, 'non-null'):
            person_disjoint_indices(X, pd.Series([0,1,0,1]))

if __name__ == '__main__': unittest.main()
