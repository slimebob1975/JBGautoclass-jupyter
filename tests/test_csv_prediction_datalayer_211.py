"""Focused tests for the opt-in CSV prediction data-layer facade."""
import unittest
from unittest.mock import Mock, patch

from Storage.CSVPredictionDataLayer import (
    CSVPredictionDataLayer, prepare_csv_prediction_datalayer,
)


class CSVPredictionDataLayerTests(unittest.TestCase):
    def test_reader_delegation_preserves_rows(self):
        reader = Mock()
        reader.get_dataset.return_value = [("0001", "")]
        layer = CSVPredictionDataLayer(reader, Mock())
        self.assertEqual(layer.get_dataset(3), [("0001", "")])
        reader.get_dataset.assert_called_once_with(num_rows=3)

    def test_sink_delegation_preserves_prediction_payload(self):
        sink = Mock()
        sink.save_prediction_data.return_value = {"error": None, "results": {"row_count": 1}}
        layer = CSVPredictionDataLayer(Mock(), sink)
        rows = [{"key": "0001", "prediction": "A", "rate": 0.8, "probabilities": None}]
        answer = layer.save_prediction_data(rows, "rate", "model", ["A"], test_run=2)
        self.assertEqual(answer["results"]["row_count"], 1)
        sink.save_prediction_data.assert_called_once_with(
            rows, "rate", "model", ["A"], create_tables=True, test_run=2)

    def test_invalid_reader_fails_closed(self):
        with self.assertRaises(ValueError):
            CSVPredictionDataLayer(object(), Mock())

    def test_invalid_sink_fails_closed(self):
        with self.assertRaises(ValueError):
            CSVPredictionDataLayer(Mock(), object())

    def test_factory_uses_existing_composition(self):
        reader, sink, config = Mock(), Mock(), object()
        with patch("Storage.CSVPredictionDataLayer.prepare_csv_prediction_io",
                   return_value=(config, reader, sink)) as prepare:
            actual_config, layer = prepare_csv_prediction_datalayer(
                "model", "runtime", "repo", "scope", "config-id")
        self.assertIs(actual_config, config)
        self.assertIsInstance(layer, CSVPredictionDataLayer)
        prepare.assert_called_once_with("model", "runtime", "repo", "scope", "config-id")


if __name__ == "__main__":
    unittest.main()
