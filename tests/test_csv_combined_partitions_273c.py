"""273c: preflight split of a single pinned CSV snapshot, no dispatch unlock."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from Storage.CSVCombinedDataLayer import CSVCombinedDataLayer
from Storage.CSVTrainingAdapter import CSVTrainingAdapter
from Storage.CSVPredictionResults import CSVPredictionResultSink


class CombinedPartitions273c(unittest.TestCase):
    def layer(self, rows):
        repo, scope = Mock(), Mock()
        reader = object.__new__(CSVTrainingAdapter)
        reader.repository = repo
        reader.scope = scope
        reader.configuration = Mock(dataset_id='dataset', artifact_id='artifact')
        reader.correction_revision = 'pinned'
        reader.get_dataset = Mock(return_value=rows)
        sink = object.__new__(CSVPredictionResultSink)
        sink.repository = repo
        sink.scope = scope
        sink.context = Mock(dataset_id='dataset', artifact_id='artifact')
        return CSVCombinedDataLayer(reader, sink), reader

    def test_single_read_separates_labelled_and_unknown_with_ids_preserved(self):
        layer, reader = self.layer([
            ('x', '001', 'A'), ('y', '002', 'B'), ('z', '003', 'A'),
            ('q', '004', 'B'), ('r', '005', ''), ('s', '006', 'NULL')])
        training, prediction = layer.read_partitions()
        self.assertEqual([r[1] for r in training], ['001', '002', '003', '004'])
        self.assertEqual([r[1] for r in prediction], ['005', '006'])
        self.assertEqual(layer.correction_revision, 'pinned')
        reader.get_dataset.assert_called_once_with(num_rows=None)

    def test_preflight_rejects_insufficient_training_classes(self):
        layer, _ = self.layer([('x', '1', 'A'), ('y', '2', 'B'), ('z', '3', '')])
        with self.assertRaisesRegex(ValueError, 'two labelled classes'):
            layer.read_partitions()

    def test_preflight_rejects_no_prediction_rows(self):
        layer, _ = self.layer([('x', '1', 'A'), ('y', '2', 'B'),
                               ('z', '3', 'A'), ('q', '4', 'B')])
        with self.assertRaisesRegex(ValueError, 'unlabelled'):
            layer.read_partitions()

    def test_preflight_rejects_invalid_rows(self):
        layer, _ = self.layer([('x', '1', 'A'), ('y', '2', 'B'),
                               ('z', '3', 'A'), ('q', '4', 'B'), 'bad'])
        with self.assertRaisesRegex(ValueError, 'invalid row'):
            layer.read_partitions()


if __name__ == '__main__':
    unittest.main()
