"""Synthetic CSV training guard regression; no source rows or SQL access."""
import unittest
from unittest.mock import Mock
from Storage.CSVTrainingAdapter import CSVTrainingAdapter
from Storage.CSVTrainingDataLayer import CSVTrainingDataLayer
from test_csv_training_classifier_254 import CSVTrainingFetchPortable254
from types import SimpleNamespace

class CSVPreflight257(unittest.TestCase):
    @staticmethod
    def config():
        cfg = Mock()
        cfg.data_source.kind = 'csv'
        cfg.should_train.return_value = True
        cfg.should_predict.return_value = False
        return cfg

    def layer(self, rows):
        reader = object.__new__(CSVTrainingAdapter)
        reader.correction_revision = None
        reader.get_dataset = Mock(return_value=rows)
        return CSVTrainingDataLayer(reader)

    def test_iris_like_rows_preserve_column_order_and_ids(self):
        rows = [(5.1, 3.5, '001', 'setosa'), (4.9, 3.0, '002', 'setosa'),
                (6.2, 2.9, '003', 'versicolor'), (5.9, 3.0, '004', 'versicolor'),
                (6.6, 3.0, '005', 'virginica'), (6.4, 2.8, '006', 'virginica')]
        layer = self.layer(rows)
        self.assertEqual(layer.get_dataset(), rows)
        runner = SimpleNamespace(config=self.config(), datalayer=layer, logger=Mock())
        self.assertEqual(CSVTrainingFetchPortable254.fetch_method()(runner)['data'], rows)
        self.assertFalse(hasattr(layer, 'save_prediction_data'))

    def test_unusable_class_support_rejected(self):
        for rows in ([], [(1, 'a', '')], [(1, 'a', 'x'), (2, 'b', 'x')],
                     [(1, 'a', 'x'), (2, 'b', 'x'), (3, 'c', 'y')]):
            with self.subTest(size=len(rows)):
                with self.assertRaisesRegex(ValueError, 'no labelled rows|two labelled classes'):
                    runner = SimpleNamespace(config=self.config(), datalayer=self.layer(rows), logger=Mock())
                    CSVTrainingFetchPortable254.fetch_method()(runner)

    def test_snapshot_and_limit_forwarded(self):
        rows = [(1, '001', 'a'), (2, '002', 'a'), (3, '003', 'b'), (4, '004', 'b')]
        layer = self.layer(rows)
        layer._reader.correction_revision = 'synthetic-head'
        self.assertEqual(layer.correction_revision, 'synthetic-head')
        self.assertEqual(layer.get_dataset(num_rows=4), rows)
        layer._reader.get_dataset.assert_called_once_with(num_rows=4)

if __name__ == '__main__':
    unittest.main()
