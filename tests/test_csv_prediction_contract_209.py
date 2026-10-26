"""Revision 209: CSV prediction payload semantics before artifact publication."""
import unittest

import test_csv_prediction_sink_207 as fixtures


class CSVPredictionContractTests(unittest.TestCase):
    setUp = fixtures.CSVPredictionSinkTests.setUp
    def test_valid_probability_order_and_opaque_id(self):
        result = self.sink.save_prediction_data(
            [{"key": "0007", "prediction": "A", "rate": .7,
              "probabilities": "0.7,0.3"}], fixtures.Rate.PROBABILITY, "model", ["A", "B"])
        self.assertEqual(result["results"]["row_count"], 1)

    def test_unknown_predicted_class_rejected_without_writes(self):
        self._reject([{"key": "007", "prediction": "other", "rate": .5,
                       "probabilities": "0.5,0.5"}], ["A", "B"])

    def test_probability_vector_length_checked_before_writes(self):
        self._reject([{"key": "007", "prediction": "A", "rate": .5,
                       "probabilities": "0.5"}], ["A", "B"])

    def test_malformed_probability_numbers_rejected_without_writes(self):
        for probabilities in ("NaN,0.1", "-0.1,1.1", "abc,0.2", "inf,0"):
            with self.subTest(probabilities=probabilities):
                self._reject([{"key": "007", "prediction": "A", "rate": .5,
                               "probabilities": probabilities}], ["A", "B"])

    def test_duplicate_model_labels_rejected(self):
        self._reject([{"key": "007", "prediction": "A", "rate": .5,
                       "probabilities": "0.5,0.5"}], ["A", "A"])

    def _reject(self, rows, labels):
        before = len(self.repo.list_artifacts(self.scope))
        with self.assertRaises(ValueError):
            self.sink.save_prediction_data(rows, fixtures.Rate.PROBABILITY, "model", labels)
        self.assertEqual(len(self.repo.list_artifacts(self.scope)), before)


if __name__ == "__main__":
    unittest.main()
