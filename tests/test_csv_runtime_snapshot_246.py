"""246: real scoped CSV correction snapshot crosses the prediction composition boundary."""
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from Storage.CSVPredictionDataLayer import CSVPredictionDataLayer
from Storage.CSVApplicationRuntime import prepare_csv_prediction_classifier
from Storage.CSVReclassification import correct_csv_class
from test_csv_correction_reader_226 import CSVCorrectionReader226


class CSVSnapshotComposition246(CSVCorrectionReader226):
    def _runtime(self):
        source = SimpleNamespace(kind='csv', dataset_id=self.roles.dataset_id,
                                 artifact_id=self.roles.artifact_id)
        return SimpleNamespace(data_source=source, should_train=lambda: False,
                               should_predict=lambda: True)

    def _compose(self, reader):
        config = self._runtime()
        app = types.ModuleType('AutomaticClassifier')
        def construct(**kw):
            return SimpleNamespace(datalayer=kw['datalayer'], config=kw['config'],
                                   handler=SimpleNamespace(datalayer=kw['datalayer']))
        app.AutomaticClassifier = construct
        with patch('Storage.CSVPredictionIO.prepare_csv_model_runtime',
                   return_value=(config, reader)), patch.dict(sys.modules, {'AutomaticClassifier': app}):
            return prepare_csv_prediction_classifier('trusted-model.sav', config, self.repo,
                                                     self.scope, self.roles.configuration_id,
                                                     Mock())

    def test_original_snapshot_survives_subsequent_edit(self):
        classifier = self._compose(self.adapter(train=False, predict=True))
        self.assertIsNone(classifier.datalayer.correction_revision)
        self.edit('ABC-17', 'A')
        self.assertEqual(classifier.datalayer.get_dataset(), [('2', 'ABC-17', '')])

    def test_pinned_revision_survives_further_edits(self):
        event = self.edit('005', 'B')
        classifier = self._compose(self.adapter(train=False, predict=True))
        self.assertEqual(classifier.datalayer.correction_revision, event.artifact_id)
        self.edit('ABC-17', 'A')
        self.assertEqual(classifier.datalayer.get_dataset(), [('2', 'ABC-17', '')])
        fresh = self._compose(self.adapter(train=False, predict=True))
        self.assertEqual(fresh.datalayer.get_dataset(), [])

    def test_unpinned_reader_refused_before_application_creation(self):
        config = self._runtime()
        reader = SimpleNamespace(get_dataset=lambda num_rows=None: [])
        layer = CSVPredictionDataLayer(reader, Mock(save_prediction_data=Mock()))
        with patch('Storage.CSVApplicationRuntime.prepare_csv_prediction_datalayer',
                   return_value=(config, layer)), patch.dict(
                       sys.modules, {'AutomaticClassifier': types.ModuleType('AutomaticClassifier')}):
            with self.assertRaisesRegex(ValueError, 'pinned correction snapshot'):
                prepare_csv_prediction_classifier('trusted-model.sav', config, self.repo,
                                                  self.scope, self.roles.configuration_id,
                                                  Mock())

    def test_sql_source_rejected_before_repository_or_model_reads(self):
        config = self._runtime()
        config.data_source.kind = 'sql'
        with patch('Storage.CSVApplicationRuntime.prepare_csv_prediction_datalayer') as prepare:
            with self.assertRaisesRegex(ValueError, 'prediction-only CSV'):
                prepare_csv_prediction_classifier('trusted-model.sav', config, self.repo,
                                                  self.scope, self.roles.configuration_id,
                                                  Mock())
            prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
