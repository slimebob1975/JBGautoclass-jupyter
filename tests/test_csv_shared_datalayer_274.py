"""Regression: CSV modes share an I/O contract without granting unsafe writes."""
import unittest
from unittest.mock import Mock

from Storage.CSVDataLayer import CSVDataLayer, CSVResultDataLayer
from Storage.CSVPredictionDataLayer import CSVPredictionDataLayer
from Storage.CSVTrainingDataLayer import CSVTrainingDataLayer
from Storage.CSVTrainingAdapter import CSVTrainingAdapter


class CSVSharedDataLayer274(unittest.TestCase):
    def test_common_read_and_scoped_write_delegation(self):
        reader = Mock(correction_revision='synthetic-revision')
        reader.get_dataset.return_value = [('example', '1')]
        sink = Mock()
        layer = CSVResultDataLayer(reader, sink)
        self.assertEqual(layer.correction_revision, 'synthetic-revision')
        self.assertEqual(layer.get_dataset(num_rows=4), [('example', '1')])
        reader.get_dataset.assert_called_once_with(num_rows=4)
        layer.save_prediction_data('rows', 'rate', 'model', ['yes'])
        sink.save_prediction_data.assert_called_once_with(
            'rows', 'rate', 'model', ['yes'], create_tables=True, test_run=0)
        layer.save_mispredicted_data('frame')
        sink.save_mispredicted_data.assert_called_once_with('frame')

    def test_read_only_training_does_not_expose_writers(self):
        reader = object.__new__(CSVTrainingAdapter)
        reader.get_dataset = Mock(return_value=[])
        reader.correction_revision = None
        layer = CSVTrainingDataLayer(reader)
        self.assertIsInstance(layer, CSVDataLayer)
        self.assertFalse(hasattr(layer, "save_prediction_data"))
        self.assertFalse(hasattr(layer, "save_mispredicted_data"))

    def test_prediction_diagnostics_remain_blocked(self):
        layer = CSVPredictionDataLayer(Mock(), Mock())
        self.assertIsInstance(layer, CSVDataLayer)
        self.assertFalse(callable(layer.save_mispredicted_data))

    def test_missing_sink_fails_closed(self):
        layer = CSVResultDataLayer(Mock())
        with self.assertRaisesRegex(ValueError, 'not prepared'):
            layer.save_prediction_data([], 'rate', 'model', [])
        with self.assertRaisesRegex(ValueError, 'not prepared'):
            layer.save_mispredicted_data([])


if __name__ == '__main__':
    unittest.main()
