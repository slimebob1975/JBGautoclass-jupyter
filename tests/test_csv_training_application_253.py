"""Revision 253: guarded CSV training application assembly (synthetic fixtures)."""
import sys
import types
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from Storage.ArtifactContract import StorageScope
from Storage.CSVTrainingApplication import prepare_registered_csv_training_classifier
from Storage.CSVTrainingAdapter import CSVTrainingAdapter
from Storage.CSVTrainingDataLayer import CSVTrainingDataLayer


class CSVTrainingApplication253(unittest.TestCase):
    def setUp(self):
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.runtime = Mock()
        self.runtime.data_source.kind = 'csv'
        self.runtime.data_source.dataset_id = 'dataset'
        self.runtime.data_source.artifact_id = 'artifact'
        self.runtime.should_train.return_value = True
        self.runtime.should_predict.return_value = False
        self.runtime.get_column_names.return_value = ['measurement', 'row_id', 'species']
        self.runtime.get_id_column_name.return_value = 'row_id'
        self.runtime.get_class_column_name.return_value = 'species'
        self.saved = types.SimpleNamespace(dataset_id='dataset', artifact_id='artifact',
                                           id_column='row_id', class_column='species',
                                           text_columns=(), data_columns=('measurement',))
        self.reader = object.__new__(CSVTrainingAdapter)
        self.reader.correction_revision = None
        self.layer = CSVTrainingDataLayer(self.reader)

    def build(self):
        app = types.ModuleType('AutomaticClassifier')
        def constructor(**kwargs):
            instance = types.SimpleNamespace(**kwargs)
            instance.handler = types.SimpleNamespace(datalayer=kwargs['datalayer'])
            return instance
        app.AutomaticClassifier = Mock(side_effect=constructor)
        with patch('Storage.CSVTrainingApplication.load_csv_model_configuration',
                   return_value=self.saved), patch(
                       'Storage.CSVTrainingApplication.prepare_csv_training_datalayer',
                       return_value=self.layer), patch.dict(sys.modules, {'AutomaticClassifier': app}):
            classifier = prepare_registered_csv_training_classifier(
                self.runtime, Mock(), self.scope, 'config', Mock())
        return classifier, app.AutomaticClassifier

    def test_registered_training_injects_same_layer_into_classifier_and_handler(self):
        classifier, constructor = self.build()
        self.assertIs(classifier.datalayer, self.layer)
        self.assertIs(classifier.handler.datalayer, self.layer)
        self.assertIs(classifier.config, self.runtime)
        self.assertFalse(constructor.call_args.kwargs['regression_suite'])

    def test_sql_or_mixed_mode_rejected_before_access(self):
        with patch('Storage.CSVTrainingApplication.load_csv_model_configuration') as loader:
            self.runtime.data_source.kind = 'sql'
            with self.assertRaisesRegex(ValueError, 'train-only CSV'):
                prepare_registered_csv_training_classifier(self.runtime, Mock(), self.scope,
                                                           'config', Mock())
            self.runtime.data_source.kind = 'csv'
            self.runtime.should_predict.return_value = True
            with self.assertRaisesRegex(ValueError, 'train-only CSV'):
                prepare_registered_csv_training_classifier(self.runtime, Mock(), self.scope,
                                                           'config', Mock())
            loader.assert_not_called()

    def test_invalid_scope_rejected_before_access(self):
        with self.assertRaisesRegex(ValueError, 'scope'):
            prepare_registered_csv_training_classifier(self.runtime, Mock(), None,
                                                       'config', Mock())

    def test_changed_columns_rejected_before_layer_preparation(self):
        self.runtime.get_column_names.return_value = ['wrong', 'row_id', 'species']
        with patch('Storage.CSVTrainingApplication.load_csv_model_configuration',
                   return_value=self.saved), patch(
                       'Storage.CSVTrainingApplication.prepare_csv_training_datalayer') as prepare:
            with self.assertRaisesRegex(ValueError, 'roles mismatch'):
                prepare_registered_csv_training_classifier(self.runtime, Mock(), self.scope,
                                                           'config', Mock())
            prepare.assert_not_called()

    def test_changed_dataset_identity_rejected_before_layer_preparation(self):
        self.runtime.data_source.dataset_id = 'other'
        with patch('Storage.CSVTrainingApplication.load_csv_model_configuration',
                   return_value=self.saved), patch(
                       'Storage.CSVTrainingApplication.prepare_csv_training_datalayer') as prepare:
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                prepare_registered_csv_training_classifier(self.runtime, Mock(), self.scope,
                                                           'config', Mock())
            prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
