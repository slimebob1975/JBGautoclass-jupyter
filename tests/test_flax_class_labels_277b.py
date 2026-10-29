"""Flax label contract: predictions and probability columns use original labels.

Extract the production fit method so this contract is testable without optional
JAX/Flax dependencies or costly network training.
"""
import ast
from pathlib import Path
import unittest

import numpy as np
from sklearn.preprocessing import LabelEncoder


class _StopAtNetworkInitialization(Exception):
    pass


class _FakeJnp:
    @staticmethod
    def array(value):
        return np.asarray(value)


class _Encoder(LabelEncoder):
    def transform(self, x):
        return super().transform(x).astype(np.int64)


class FlaxClassLabels277b(unittest.TestCase):
    def test_fitted_classes_match_original_string_labels(self):
        source = Path(__file__).resolve().parents[1] / 'src' / 'JBGJaxTransformers.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        classifier = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FlaxClassifier')
        fit = next(n for n in classifier.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
        fit.returns = None
        for arg in fit.args.args:
            arg.annotation = None
        ns = {'jnp': _FakeJnp()}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fit], type_ignores=[])), str(source), 'exec'), ns)

        class MinimalFlax:
            label_encoder = _Encoder()
            state = None
            def _initialize_model(self, **kwargs):
                raise _StopAtNetworkInitialization()

        model = MinimalFlax()
        labels = np.array(['Iris-virginica', 'Iris-setosa', 'Iris-versicolor'] * 3)
        with self.assertRaises(_StopAtNetworkInitialization):
            ns['fit'](model, np.ones((9, 4)), labels)
        self.assertEqual(list(model.classes_), ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
        np.testing.assert_array_equal(model.label_encoder.inverse_transform(np.arange(3)), model.classes_)
        self.assertFalse(any(str(i) in model.classes_ for i in range(3)))


if __name__ == '__main__':
    unittest.main()
