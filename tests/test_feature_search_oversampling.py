"""Revision 137: reuse JBG sampler definitions, with fold-local isolation."""
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import StratifiedKFold

from JBGFeatureSearch import SearchOptions, make_pipeline


class OversamplingTests(unittest.TestCase):
    def test_config_contract(self):
        self.assertEqual(SearchOptions().oversampling, 'NOG')
        for code in ('NOG', 'RND', 'SME'):
            SearchOptions(oversampling=code).validate()
        with self.assertRaisesRegex(ValueError, 'oversampling'):
            SearchOptions(oversampling='ADASYN').validate()

    def test_pipeline_uses_existing_JBG_sampler_contract(self):
        # Lightweight stand-in for JBGMeta: avoids importing GUI/model extras
        # in the focused backend test environment.
        class Member:
            def __init__(self, klass): self.klass = klass
            def get_callable_oversampler(self):
                return self.klass(sampling_strategy='auto')
        class ExistingOversampling:
            def __class_getitem__(cls, item):
                return {'RND': Member(RandomOverSampler), 'SME': Member(SMOTE)}[item]
        with patch.dict(sys.modules, {'JBGMeta': types.SimpleNamespace(Oversampling=ExistingOversampling)}):
            for code, expected in [('RND', RandomOverSampler), ('SME', SMOTE)]:
                options = SearchOptions(model_type='Linear', oversampling=code)
                pipe = make_pipeline(('f1', 'f2'), options, None)
                self.assertIsInstance(pipe.named_steps['oversampling'], expected)
                self.assertEqual(pipe.named_steps['oversampling'].random_state, options.seed)
                self.assertEqual(list(pipe.named_steps), ['columns', 'impute', 'scale', 'oversampling', 'logistic'])

    def test_validation_rows_not_resampled(self):
        class Member:
            def get_callable_oversampler(self): return RandomOverSampler(sampling_strategy='auto')
        class ExistingOversampling:
            def __class_getitem__(cls, item): return Member()
        X = pd.DataFrame({'f1': np.arange(40, dtype=float), 'f2': np.arange(40, dtype=float) % 3})
        y = pd.Series(['rare'] * 8 + ['normal'] * 32)
        with patch.dict(sys.modules, {'JBGMeta': types.SimpleNamespace(Oversampling=ExistingOversampling)}):
            options = SearchOptions(model_type='Linear', oversampling='RND')
            train, validation = next(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X, y))
            pipe = make_pipeline(tuple(X.columns), options, None)
            pipe.fit(X.iloc[train], y.iloc[train])
            predictions = pipe.predict(X.iloc[validation])
            self.assertEqual(len(predictions), len(validation))
            self.assertGreater(len(pipe.named_steps['oversampling'].sample_indices_), len(train))
            self.assertEqual(sum(y.iloc[validation] == 'rare'), 3)

if __name__ == '__main__':
    unittest.main()
