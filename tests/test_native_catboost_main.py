"""Smoke tests for the standalone native-category adapter (no JBG GUI imports)."""
import unittest
import pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from JBGNativeCatBoost import OrdinaryNativeCatBoostClassifier
from JBGNativeCatBoost import RawCatBoostColumns, prepare_model_input


class NativeCatBoostMainTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 120
        self.X = pd.DataFrame({
            'KassaNr': rng.choice(['04', '19', '69'], size=n),
            'Belopp': rng.normal(size=n),
            'HarAvvikelse': rng.choice(['Ja', 'Nej'], size=n),
        })
        self.X.loc[[0, 2], 'KassaNr'] = None
        self.X.loc[[1, 5], 'Belopp'] = np.nan
        self.y = np.array(['Korrekt'] * 100 + ['Inkorrekt'] * 20)
        self.pipe = Pipeline([
            ('native_columns', RawCatBoostColumns(tuple(self.X.columns),
                              ('KassaNr', 'HarAvvikelse'))),
            ('CBST', OrdinaryNativeCatBoostClassifier(iterations=20, depth=3, learning_rate=0.1)),
        ])

    def test_dataframe_and_numpy_equivalence(self):
        self.pipe.fit(self.X, self.y)
        p_frame = self.pipe.predict_proba(self.X)
        p_array = self.pipe.predict_proba(self.X.to_numpy())
        np.testing.assert_allclose(p_frame, p_array)
        self.assertIs(prepare_model_input(self.X, self.pipe, prefer_numpy=True), self.X)

    def test_existing_dark_number_correction_accepts_mixed_types(self):
        from JBGDarkNumberCorrectionFactor import DarkNumberCorrectionFactorEstimator
        frame = pd.DataFrame({
            'KassaNr': ['04', '19', '51'] * 50,
            'Belopp': np.arange(150, dtype=float),
        })
        y = np.asarray(['Korrekt'] * 100 + ['Inkorrekt'] * 50)
        model = Pipeline([
            ('native_columns', RawCatBoostColumns(tuple(frame.columns), ('KassaNr',))),
            ('CBST', OrdinaryNativeCatBoostClassifier(iterations=10, depth=3)),
        ])
        correction = DarkNumberCorrectionFactorEstimator(
            estimator=model, flip_fraction=0.2, n_splits=2, n_repeats=1,
            n_jobs=1, random_state=42, positive_class='Inkorrekt',
            sample_size=1, parallel_backend='threads',
        )
        correction.fit(frame, y)
        self.assertEqual(correction.correction_status_, 'estimated')
        self.assertTrue(np.isfinite(correction.score()))

    def test_persistence(self):
        self.pipe.fit(self.X, self.y)
        restored = pickle.loads(pickle.dumps(self.pipe))
        np.testing.assert_allclose(
            self.pipe.predict_proba(self.X), restored.predict_proba(self.X))

    def test_cv_and_clone(self):
        result = cross_val_score(clone(self.pipe), self.X, self.y,
                                 cv=StratifiedKFold(3, shuffle=True, random_state=42))
        self.assertEqual(len(result), 3)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_categorical_columns_are_fixed_when_numpy_is_object(self):
        transformed = RawCatBoostColumns(tuple(self.X.columns),
                                        ('KassaNr', 'HarAvvikelse')).fit_transform(self.X.to_numpy())
        self.assertEqual(str(transformed['Belopp'].dtype), 'float64')
        self.assertEqual(str(transformed['KassaNr'].dtype), 'object')
        self.assertEqual(transformed.loc[0, 'KassaNr'], '__MISSING__')


if __name__ == '__main__':
    unittest.main()
