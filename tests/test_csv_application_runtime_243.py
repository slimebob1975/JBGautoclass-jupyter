"""Revision 243: guarded CSV application composition, without GUI activation."""
import sys
import types
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from Storage.ArtifactContract import StorageScope
from Storage.CSVApplicationRuntime import prepare_csv_prediction_classifier
from Storage.CSVPredictionDataLayer import CSVPredictionDataLayer


class CSVApplicationRuntime243(unittest.TestCase):
    def setUp(self):
        # Repository authorization happens in the caller; use a valid scope.
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.runtime = Mock()
        self.runtime.data_source.kind = 'csv'
        self.runtime.should_train.return_value = False
        self.runtime.should_predict.return_value = True
        self.saved = Mock()
        self.saved.data_source.kind = 'csv'
        self.saved.should_train.return_value = False
        self.saved.should_predict.return_value = True
        self.layer = CSVPredictionDataLayer(Mock(get_dataset=Mock()), Mock(save_prediction_data=Mock()))
        self.logger = Mock()

    def _prepare(self):
        with patch('Storage.CSVApplicationRuntime.prepare_csv_prediction_datalayer',
                   return_value=(self.saved, self.layer)) as factory:
            app = types.ModuleType('AutomaticClassifier')
            def build(**kwargs):
                classifier = Mock()
                classifier.datalayer = kwargs['datalayer']
                classifier.config = kwargs['config']
                classifier.handler.datalayer = kwargs['datalayer']
                return classifier
            app.AutomaticClassifier = Mock(side_effect=build)
            with patch.dict(sys.modules, {'AutomaticClassifier': app}):
                classifier = prepare_csv_prediction_classifier(
                    'trusted.pkl', self.runtime, Mock(), self.scope, 'config-id', self.logger)
            return classifier, factory, app.AutomaticClassifier

    def test_validated_csv_data_layer_is_injected_into_application(self):
        classifier, factory, constructor = self._prepare()
        self.assertIs(classifier.datalayer, self.layer)
        self.assertIs(classifier.handler.datalayer, self.layer)
        self.assertIs(classifier.config, self.saved)
        self.assertFalse(constructor.call_args.kwargs['regression_suite'])
        self.assertEqual(factory.call_count, 1)

    def test_sql_source_rejected_before_preparation(self):
        self.runtime.data_source.kind = 'sql'
        with patch('Storage.CSVApplicationRuntime.prepare_csv_prediction_datalayer') as factory:
            with self.assertRaisesRegex(ValueError, 'prediction-only CSV'):
                prepare_csv_prediction_classifier('trusted.pkl', self.runtime, Mock(),
                                                  self.scope, 'config-id', self.logger)
            factory.assert_not_called()

    def test_training_and_invalid_scope_rejected(self):
        self.runtime.should_train.return_value = True
        with self.assertRaises(ValueError):
            prepare_csv_prediction_classifier('trusted.pkl', self.runtime, Mock(),
                                              self.scope, 'config-id', self.logger)
        self.runtime.should_train.return_value = False
        with self.assertRaises(ValueError):
            prepare_csv_prediction_classifier('trusted.pkl', self.runtime, Mock(),
                                              object(), 'config-id', self.logger)

    def test_restored_sql_model_cannot_enter_application(self):
        self.saved.data_source.kind = 'sql'
        with patch('Storage.CSVApplicationRuntime.prepare_csv_prediction_datalayer',
                   return_value=(self.saved, self.layer)):
            with self.assertRaisesRegex(ValueError, 'Restored model'):
                prepare_csv_prediction_classifier('trusted.pkl', self.runtime, Mock(),
                                                  self.scope, 'config-id', self.logger)


if __name__ == '__main__':
    unittest.main()
