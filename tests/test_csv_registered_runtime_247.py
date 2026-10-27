"""247: repository-backed registration gate before CSV classifier construction."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from Storage.CSVApplicationRuntime import prepare_registered_csv_prediction_classifier
from test_csv_correction_reader_226 import CSVCorrectionReader226


class CSVRegisteredRuntime247(CSVCorrectionReader226):
    def runtime(self):
        source = SimpleNamespace(kind='csv', dataset_id=self.roles.dataset_id,
                                 artifact_id=self.roles.artifact_id)
        return SimpleNamespace(
            data_source=source, should_train=lambda: False,
            should_predict=lambda: True,
            get_id_column_name=lambda: self.roles.id_column,
            get_class_column_name=lambda: self.roles.class_column,
            get_column_names=lambda: list(self.roles.text_columns) +
                list(self.roles.data_columns) +
                [self.roles.id_column, self.roles.class_column],
        )

    def compose(self, runtime):
        with patch('Storage.CSVApplicationRuntime.prepare_csv_prediction_classifier',
                   return_value=object()) as prepared:
            result = prepare_registered_csv_prediction_classifier(
                'trusted-model.sav', runtime, self.repo, self.scope,
                self.roles.configuration_id, Mock())
            self.assertIs(result, prepared.return_value)
            prepared.assert_called_once()

    def test_registered_source_and_roles_reach_composition(self):
        self.compose(self.runtime())

    def test_dataset_identity_mismatch_stops_before_composition(self):
        runtime = self.runtime()
        runtime.data_source.artifact_id = 'different-artifact'
        with patch('Storage.CSVApplicationRuntime.prepare_csv_prediction_classifier') as prepared:
            with self.assertRaisesRegex(ValueError, 'source identity'):
                prepare_registered_csv_prediction_classifier(
                    'trusted-model.sav', runtime, self.repo, self.scope,
                    self.roles.configuration_id, Mock())
            prepared.assert_not_called()

    def test_role_mismatch_stops_before_composition(self):
        runtime = self.runtime()
        runtime.get_id_column_name = lambda: 'wrong-id'
        with patch('Storage.CSVApplicationRuntime.prepare_csv_prediction_classifier') as prepared:
            with self.assertRaisesRegex(ValueError, 'model roles'):
                prepare_registered_csv_prediction_classifier(
                    'trusted-model.sav', runtime, self.repo, self.scope,
                    self.roles.configuration_id, Mock())
            prepared.assert_not_called()

    def test_sql_source_cannot_read_registered_csv(self):
        runtime = self.runtime()
        runtime.data_source.kind = 'sql'
        with patch('Storage.CSVApplicationRuntime.prepare_csv_prediction_classifier') as prepared:
            with self.assertRaisesRegex(ValueError, 'prediction-only CSV'):
                prepare_registered_csv_prediction_classifier(
                    'trusted-model.sav', runtime, self.repo, self.scope,
                    self.roles.configuration_id, Mock())
            prepared.assert_not_called()


if __name__ == '__main__':
    unittest.main()
