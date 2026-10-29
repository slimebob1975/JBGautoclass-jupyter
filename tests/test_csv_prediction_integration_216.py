"""Revision 216: SQL-free CSV prediction I/O integration smoke tests.

Uses a real serialized sklearn model; the JBG Config artifact reader is patched
so this does not claim verification of full JBG model serialization/TaskRunner.
"""
import json
import pickle
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
from uuid import uuid4

import numpy as np
from sklearn.linear_model import LogisticRegression

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.CSVModelConfiguration import save_csv_model_configuration
from Storage.CSVPredictionDataLayer import prepare_csv_prediction_datalayer
from Storage.FileArtifactRepository import FileArtifactRepository


class Rate(Enum):
    PROBABILITY = 1


class SavedConfig:
    def __init__(self, ref):
        self.data_source = SimpleNamespace(kind='csv', dataset_id=ref.dataset_id,
                                           artifact_id=ref.artifact_id)
    def get_column_names(self): return ['value', 'uid', 'label']
    def get_id_column_name(self): return 'uid'
    def get_class_column_name(self): return 'label'
    def should_train(self): return False
    def should_predict(self): return True
    def get_data_limit(self): return 100


class CSVPredictionIntegration216(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.repo = FileArtifactRepository(root / 'repository')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                     'text/csv', dataset_id=str(uuid4()))
        from io import BytesIO
        self.repo.save_artifact(self.scope, self.ref, BytesIO(
            b'value;uid;label\n0;001;A\n1;002;B\n0.1;003;A\n0.9;004;B\n0.2;005;\n0.8;006;\n'))
        self.roles = save_csv_model_configuration(
            self.repo, self.scope, self.ref.artifact_id, id_column='uid',
            class_column='label', data_columns=('value',), text_columns=(),
            csv_format={'delimiter': ';', 'quotechar': '"', 'encoding': 'utf-8'})
        trained = LogisticRegression(random_state=42).fit(
            np.array([[0.], [1.], [0.1], [0.9]]), np.array(['A', 'B', 'A', 'B']))
        self.model_path = root / 'trusted-model.pkl'
        with self.model_path.open('wb') as handle:
            pickle.dump(trained, handle)
        self.runtime = SavedConfig(self.ref)

    def _prepare(self, scope=None, runtime=None):
        with patch.dict(sys.modules, {'Config': SimpleNamespace(Config=SimpleNamespace(
                load_config_from_model_file=lambda filename, config: self.runtime))}):
            return prepare_csv_prediction_datalayer(
                str(self.model_path), runtime or self.runtime, self.repo,
                scope or self.scope, self.roles.configuration_id)

    def test_saved_estimator_to_scoped_artifacts_without_sql(self):
        with self.model_path.open('rb') as handle:
            model = pickle.load(handle)  # Only locally generated trusted test artifact.
        config, layer = self._prepare()
        self.assertIs(config, self.runtime)
        rows = layer.get_dataset()
        self.assertEqual([row[1] for row in rows], ['005', '006'])
        features = np.array([[float(row[0])] for row in rows])
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        result_rows = [dict(key=row[1], prediction=prediction,
                            rate=np.float64(max(probs)),
                            probabilities=','.join(str(x) for x in probs))
                       for row, prediction, probs in zip(rows, predictions, probabilities)]
        response = layer.save_prediction_data(
            result_rows, Rate.PROBABILITY, 'synthetic-model', model.classes_)
        self.assertIsNone(response['error'])
        self.assertEqual(response['results']['row_count'], 2)
        refs = self.repo.list_artifacts(self.scope, run_id=response['results']['run_id'])
        self.assertEqual({r.kind for r in refs}, {ArtifactKind.MANIFEST, ArtifactKind.PREDICTIONS})
        payload_ref = next(r for r in refs if r.kind == ArtifactKind.PREDICTIONS)
        manifest_ref = next(r for r in refs if r.kind == ArtifactKind.MANIFEST)
        with self.repo.open_artifact(self.scope, payload_ref.artifact_id) as handle:
            payload = json.load(handle)
        with self.repo.open_artifact(self.scope, manifest_ref.artifact_id) as handle:
            manifest = json.load(handle)
        self.assertEqual([r['key'] for r in payload['rows']], ['005', '006'])
        self.assertEqual([r['prediction'] for r in payload['rows']], predictions.tolist())
        self.assertEqual(manifest['row_count'], 2)
        self.assertEqual(manifest['predictions_artifact_id'], payload_ref.artifact_id)
        self.assertNotIn('rows', manifest)
        self.assertNotIn('key', manifest)
        self.assertNotIn('prediction', manifest)
        self.assertNotIn('value', manifest)

    def test_invalid_runtime_source_fails_before_reading(self):
        wrong = SavedConfig(self.ref)
        wrong.data_source.kind = 'sql'
        with self.assertRaisesRegex(ValueError, 'CSV'):
            self._prepare(runtime=wrong)
        self.assertFalse(any(r.kind == ArtifactKind.PREDICTIONS
                             for r in self.repo.list_artifacts(self.scope)))

    def test_wrong_project_scope_cannot_load_configuration(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            self._prepare(scope=StorageScope(str(uuid4()), str(uuid4())))

    def test_mismatched_saved_identity_prevents_publication(self):
        broken = SavedConfig(self.ref)
        broken.data_source.artifact_id = str(uuid4())
        with patch.dict(sys.modules, {'Config': SimpleNamespace(Config=SimpleNamespace(
                load_config_from_model_file=lambda filename, config: broken))}):
            with self.assertRaises(ValueError):
                prepare_csv_prediction_datalayer(str(self.model_path), self.runtime,
                    self.repo, self.scope, self.roles.configuration_id)
        self.assertFalse(any(r.kind == ArtifactKind.PREDICTIONS
                             for r in self.repo.list_artifacts(self.scope)))


if __name__ == '__main__':
    unittest.main()
