"""Revision 231: real DatasetHandler row processing to real prediction persistence.

Uses production method bodies and a fitted sklearn estimator. Full application
bootstrap, ModelHandler.load_model, and TaskRunner.run are NOT exercised here.
"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from Storage.ArtifactContract import ArtifactKind
import test_csv_dataset_ingest_230 as ingestion
import test_csv_real_handlers_221 as handlers


class CSVHandlerBridge231(handlers.CSVRealHandlers221):
    def setUp(self):
        super().setUp()
        cfg = self.config
        cfg.column_is_text = lambda name: False
        cfg.get_text_column_names = lambda: []
        cfg.should_train = lambda: False
        cfg.should_predict = lambda: True
        cfg.get_column_names = lambda: ['value', 'uid', 'label']
        cfg.get_id_column_name = lambda: 'uid'
        cfg.get_class_column_name = lambda: 'label'
        data = ingestion.RealDatasetCSVIngest230.build(self, source='csv')
        # The source-level fixture builds a handler with its own default config;
        # use the saved runtime config so the row contract is the real one.
        data.handler.config = cfg
        data.handler.logger = Mock()
        data.handler.logger.print_privacy_safe = Mock()
        data.load_data(self.layer.get_dataset())
        data.separate_dataset()
        data.split_dataset_for_training_and_validation()
        self.handler.handlers['dataset'] = data
        self.dataset = data
        features = data.X_prediction[['value']].to_numpy(dtype=float)
        self.predictions.predictions = self.estimator.predict(features)
        self.predictions.probabilites = self.estimator.predict_proba(features)
        self.predictions.rates = np.max(self.predictions.probabilites, axis=1)

    def test_saved_csv_rows_to_real_prediction_handlers(self):
        self.assertEqual(self.dataset.unpredicted_keys.tolist(), ['005', '006'])
        result = self.handler.save_predictions()
        self.assertIsNone(result['error'])
        refs = self.repo.list_artifacts(self.scope, run_id=result['results']['run_id'])
        self.assertEqual({r.kind for r in refs}, {ArtifactKind.PREDICTIONS, ArtifactKind.MANIFEST})
        record = next(r for r in refs if r.kind == ArtifactKind.PREDICTIONS)
        with self.repo.open_artifact(self.scope, record.artifact_id) as stream:
            rows = json.load(stream)['rows']
        self.assertEqual([r['key'] for r in rows], ['005', '006'])

    def test_dataset_index_and_prediction_keys_remain_aligned(self):
        self.assertEqual(self.dataset.X_prediction.index.tolist(), ['005', '006'])
        self.assertEqual(self.dataset.unpredicted_keys.tolist(), ['005', '006'])
        self.assertEqual(len(self.predictions.predictions), len(self.dataset.X_prediction))


if __name__ == '__main__':
    unittest.main()
