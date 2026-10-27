"""Revision 217: real JBG artifact I/O and shared prediction-result construction.

The Config facade is narrowed to the real artifact reader plus source guard; the
full Config GUI/runtime loader and TaskRunner are not exercised by this test.
"""
import ast
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from JBGModelPersistence import (load_model_artifact_with_compatibility,
                                 load_model_config, save_model_artifact)
from JBGModelSourceIdentity import check_model_source_identity
from JBGPredictionKeys import prediction_key
from Storage.ArtifactContract import ArtifactKind
from Storage.CSVPredictionDataLayer import prepare_csv_prediction_datalayer
import test_csv_prediction_integration_216 as fixture216


def production_prediction_results():
    """Compile the unmodified production PredictionsHandler method in isolation."""
    path = Path(__file__).resolve().parents[1] / 'src' / 'JBGHandler.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PredictionsHandler')
    method = next(n for n in klass.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'get_prediction_results')
    method.decorator_list = []
    namespace = {'prediction_key': prediction_key, 'pd': pd}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace['get_prediction_results']


class CSVJBGArtifactHandler217(unittest.TestCase):
    def setUp(self):
        # Reuse the synthetic, registered CSV fixture from the previous revision.
        fixture216.CSVPredictionIntegration216.setUp(self)
        self.model = LogisticRegression(random_state=42).fit(
            np.array([[0.], [1.], [0.1], [0.9]]),
            np.array(['A', 'B', 'A', 'B']))
        self.model_path = Path(self.model_path).with_suffix('.sav')
        save_model_artifact(self.model_path, self.runtime, None, (None, None, None, None, None),
                            self.model, None, 1,
                            compatibility={'source': 'synthetic-test'})

    def _prepare_real_artifact(self, runtime=None):
        runtime = runtime or self.runtime
        def load(filename, config):
            saved = load_model_config(filename)
            check_model_source_identity(saved, config)
            return saved
        with patch.dict(sys.modules, {'Config': SimpleNamespace(
                Config=SimpleNamespace(load_config_from_model_file=load))}):
            return prepare_csv_prediction_datalayer(
                str(self.model_path), runtime, self.repo, self.scope,
                self.roles.configuration_id)

    def test_real_jbg_artifact_shared_result_handler_and_private_publication(self):
        artifact, compatibility = load_model_artifact_with_compatibility(self.model_path)
        self.assertEqual(compatibility['source'], 'synthetic-test')
        saved, layer = self._prepare_real_artifact()
        self.assertEqual(saved.data_source.dataset_id, self.ref.dataset_id)
        rows = layer.get_dataset()
        self.assertEqual([r[1] for r in rows], ['005', '006'])
        features = np.asarray([[float(r[0])] for r in rows])
        estimator = artifact[3]
        predictions = estimator.predict(features)
        probabilities = estimator.predict_proba(features)
        values = SimpleNamespace(
            handler=SimpleNamespace(config=saved), predictions=predictions,
            rates=np.max(probabilities, axis=1), probabilites=probabilities,
            get_probablities_as_string=lambda p: ','.join(str(v) for v in p))
        results = production_prediction_results()(values, pd.Series([r[1] for r in rows]))
        self.assertEqual([item['key'] for item in results], ['005', '006'])
        response = layer.save_prediction_data(results, fixture216.Rate.PROBABILITY,
                                              'synthetic-model', estimator.classes_)
        self.assertIsNone(response['error'])
        run_id = response['results']['run_id']
        refs = self.repo.list_artifacts(self.scope, run_id=run_id)
        self.assertEqual({r.kind for r in refs},
                         {ArtifactKind.PREDICTIONS, ArtifactKind.MANIFEST})
        payload_ref = next(r for r in refs if r.kind == ArtifactKind.PREDICTIONS)
        with self.repo.open_artifact(self.scope, payload_ref.artifact_id) as handle:
            payload = json.load(handle)
        self.assertEqual([r['key'] for r in payload['rows']], ['005', '006'])
        self.assertEqual([r['prediction'] for r in payload['rows']], predictions.tolist())

    def test_real_artifact_rejects_wrong_dataset_before_publication(self):
        wrong = fixture216.SavedConfig(self.ref)
        wrong.data_source.dataset_id = 'other-dataset'
        with self.assertRaisesRegex(ValueError, 'different dataset'):
            self._prepare_real_artifact(wrong)
        self.assertFalse(any(r.kind == ArtifactKind.PREDICTIONS
                             for r in self.repo.list_artifacts(self.scope)))

    def test_real_artifact_rejects_cross_source_reuse(self):
        wrong = fixture216.SavedConfig(self.ref)
        wrong.data_source.kind = 'sql'
        with self.assertRaisesRegex(ValueError, 'CSV data source'):
            self._prepare_real_artifact(wrong)

    def test_shared_handler_keeps_historical_sql_numeric_keys(self):
        state = SimpleNamespace(
            handler=SimpleNamespace(config=SimpleNamespace()),
            predictions=np.array(['A']), rates=np.array([0.75]),
            probabilites=np.array([[0.75, 0.25]]),
            get_probablities_as_string=lambda p: ','.join(str(v) for v in p))
        records = production_prediction_results()(state, pd.Series(['0007']))
        self.assertEqual(records[0]['key'], 7)


if __name__ == '__main__':
    unittest.main()
