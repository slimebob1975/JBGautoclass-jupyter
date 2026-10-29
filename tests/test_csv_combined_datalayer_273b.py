"""273b: shared CSV read/write boundary, no GUI or mixed-mode dispatch."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from Storage.CSVCombinedDataLayer import CSVCombinedDataLayer, prepare_csv_combined_datalayer
from Storage.CSVTrainingAdapter import CSVTrainingAdapter
from Storage.CSVPredictionResults import CSVPredictionResultSink
from Storage.ArtifactContract import StorageScope


class CombinedBoundary273b(unittest.TestCase):
    def setUp(self):
        self.scope = StorageScope('00000000-0000-4000-8000-000000000001',
                                  '00000000-0000-4000-8000-000000000002')
        self.repository = Mock()
        self.config = Mock()
        self.config.data_source.kind = 'csv'
        self.config.data_source.dataset_id = 'dataset'
        self.config.data_source.artifact_id = 'artifact'
        self.config.should_train.return_value = True
        self.config.should_predict.return_value = True
        self.config.get_id_column_name.return_value = 'uid'
        self.config.get_class_column_name.return_value = 'target'
        self.config.get_column_names.return_value = ['measurement', 'uid', 'target']
        self.saved = Mock(dataset_id='dataset', artifact_id='artifact',
                          id_column='uid', class_column='target',
                          text_columns=(), data_columns=('measurement',))

    def test_mode_and_identity_fail_before_readers_or_writers(self):
        with patch('Storage.CSVCombinedDataLayer.load_csv_model_configuration') as load:
            self.config.should_predict.return_value = False
            with self.assertRaisesRegex(ValueError, 'train and predict'):
                prepare_csv_combined_datalayer(self.config, self.repository, self.scope, 'conf')
            self.config.should_predict.return_value = True
            self.config.data_source.kind = 'sql'
            with self.assertRaisesRegex(ValueError, 'train and predict'):
                prepare_csv_combined_datalayer(self.config, self.repository, self.scope, 'conf')
            load.assert_not_called()
        self.config.data_source.kind = 'csv'
        self.config.data_source.artifact_id = 'wrong'
        with patch('Storage.CSVCombinedDataLayer.load_csv_model_configuration', return_value=self.saved), patch('Storage.CSVCombinedDataLayer.CSVTrainingAdapter') as reader:
            with self.assertRaisesRegex(ValueError, 'identity|roles'):
                prepare_csv_combined_datalayer(self.config, self.repository, self.scope, 'conf')
            reader.assert_not_called()

    def test_composition_keeps_reader_and_sink_bound_to_same_dataset(self):
        # Test real validation without constructing storage infrastructure.
        with patch('Storage.CSVCombinedDataLayer.CSVTrainingAdapter', CSVTrainingAdapter), patch('Storage.CSVCombinedDataLayer.CSVPredictionResultSink', CSVPredictionResultSink):
            # Lightweight objects that are genuine instances of the required types.
            actual_reader = object.__new__(CSVTrainingAdapter)
            actual_reader.repository = self.repository
            actual_reader.scope = self.scope
            actual_reader.configuration = self.saved
            actual_reader.correction_revision = None
            actual_reader.get_dataset = Mock(return_value=[('1', '001', 'A'), ('2', '002', 'A'), ('3', '003', 'B'), ('4', '004', 'B'), ('5', '005', '')])
            actual_sink = object.__new__(CSVPredictionResultSink)
            actual_sink.repository = self.repository
            actual_sink.scope = self.scope
            actual_sink.context = Mock(dataset_id='dataset', artifact_id='artifact')
            actual_sink.save_prediction_data = Mock(return_value={'results': 1})
            layer = CSVCombinedDataLayer(actual_reader, actual_sink)
            self.assertIsNone(layer.correction_revision)
            self.assertEqual(len(layer.get_dataset()), 5)
            self.assertEqual(layer.save_prediction_data([], None, 'm', []), {'results': 1})
            actual_sink.context.artifact_id = 'other'
            with self.assertRaisesRegex(ValueError, 'scope mismatch'):
                CSVCombinedDataLayer(actual_reader, actual_sink)

    def test_training_dispatch_remains_gated(self):
        from Storage.CSVTrainingDataLayer import prepare_csv_training_datalayer
        with self.assertRaisesRegex(ValueError, 'train-only'):
            prepare_csv_training_datalayer(self.config, self.repository, self.scope, 'conf')


if __name__ == '__main__':
    unittest.main()
