"""Revision 255: actual sklearn fitting and JBG artifact roundtrip; synthetic data only.

This tests real estimator fitting and the real versioned persistence contract.
The full GUI/AutomaticClassifier/TaskRunner training integration remains a later gate.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from JBGModelPersistence import (save_model_artifact, load_model_artifact_with_compatibility,
                                 load_model_compatibility)


class CSVFittedRoundTrip255(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(255)
        self.x = np.vstack([
            rng.normal(loc=(4.9, 3.4, 1.5, 0.2), scale=0.15, size=(45, 4)),
            rng.normal(loc=(5.9, 2.7, 4.2, 1.3), scale=0.15, size=(45, 4)),
            rng.normal(loc=(6.6, 3.0, 5.5, 2.0), scale=0.15, size=(45, 4)),
        ])
        self.y = np.repeat(['setosa', 'versicolor', 'virginica'], 45)
        self.predict_x = np.vstack([
            rng.normal(loc=(4.9, 3.4, 1.5, 0.2), scale=0.15, size=(5, 4)),
            rng.normal(loc=(5.9, 2.7, 4.2, 1.3), scale=0.15, size=(5, 4)),
            rng.normal(loc=(6.6, 3.0, 5.5, 2.0), scale=0.15, size=(5, 4)),
        ])
        self.compatibility = {'schema_version': 2, 'source_kind': 'csv',
                              'dataset_id': 'synthetic-iris-255', 'source_columns':
                              ('sepal_length', 'sepal_width', 'petal_length', 'petal_width')}

    def fitted(self):
        model = Pipeline([('STA', StandardScaler()), ('LRN', LogisticRegression(max_iter=200))])
        model.fit(self.x, self.y)
        return model

    def test_fitted_model_persists_reloads_and_predicts_fifteen_rows(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'csv-model.sav'
            model = self.fitted()
            expected = model.predict(self.predict_x)
            with patch('socket.create_connection', side_effect=AssertionError('No network permitted')):
                save_model_artifact(path, {'dataset': 'synthetic-iris-255'}, None,
                                    (None, None, 'STA', None, 'LRN'), model, None, 4,
                                    compatibility=self.compatibility)
                (config, converter, components, restored, keras, n_features), metadata = (
                    load_model_artifact_with_compatibility(path))
                actual = restored.predict(self.predict_x)
            self.assertEqual(config['dataset'], 'synthetic-iris-255')
            self.assertEqual(n_features, 4)
            self.assertEqual(metadata, self.compatibility)
            self.assertEqual(len(actual), 15)
            np.testing.assert_array_equal(actual, expected)
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_failed_serialization_does_not_publish_model_or_temp_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'failed.sav'
            with patch('JBGModelPersistence.pickle.dump', side_effect=OSError('synthetic disk failure')):
                with self.assertRaises(OSError):
                    save_model_artifact(path, {}, None, (None,) * 5, self.fitted(), None, 4,
                                        compatibility=self.compatibility)
            self.assertFalse(path.exists())
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_source_compatibility_metadata_is_retained(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'csv-model.sav'
            save_model_artifact(path, {}, None, (None,) * 5, self.fitted(), None, 4,
                                compatibility=self.compatibility)
            self.assertEqual(load_model_compatibility(path)['source_kind'], 'csv')
            self.assertEqual(load_model_compatibility(path)['dataset_id'], 'synthetic-iris-255')


if __name__ == '__main__':
    unittest.main()

class CSVSaveFailureGuard255(unittest.TestCase):
    def test_csv_save_failure_guard_is_present_in_native_handler(self):
        import ast
        source = (Path(__file__).resolve().parents[1] / 'src' / 'JBGHandler.py').read_text(encoding='utf-8')
        root = ast.parse(source)
        cls = next(node for node in root.body if isinstance(node, ast.ClassDef) and node.name == 'ModelHandler')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'save_model_to_file')
        branch = [node for node in ast.walk(method) if isinstance(node, ast.If) and
                  isinstance(node.test, ast.Compare) and 'csv' in ast.unparse(node.test)]
        self.assertTrue(branch)
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(branch[0])))
