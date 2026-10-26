"""Revision 215: sklearn/NumPy prediction values at the CSV storage boundary."""
import json
import unittest

import numpy as np

import test_csv_prediction_sink_207 as fixtures


class CSVNumPyPredictionTests(unittest.TestCase):
    setUp = fixtures.CSVPredictionSinkTests.setUp

    def _save(self, prediction, rate, labels, probabilities='0.2,0.8'):
        return self.sink.save_prediction_data(
            [{'key': '0008', 'prediction': prediction, 'rate': rate,
              'probabilities': probabilities}],
            fixtures.Rate.PROBABILITY, 'model', labels)

    def test_numpy_string_labels_and_rates_serializable(self):
        result = self._save(np.str_('B'), np.float64(.8),
                            np.array(['A', 'B'], dtype=np.str_))
        refs = self.repo.list_artifacts(self.scope, run_id=result['results']['run_id'])
        from Storage.ArtifactContract import ArtifactKind
        row_ref = next(ref for ref in refs if ref.kind == ArtifactKind.PREDICTIONS)
        with self.repo.open_artifact(self.scope, row_ref.artifact_id) as handle:
            row = json.load(handle)['rows'][0]
        self.assertEqual(row['key'], '0008')
        self.assertEqual(row['prediction'], 'B')
        self.assertEqual(row['rate'], .8)
        self.assertEqual(row['probabilities'], '0.2,0.8')

    def test_integer_numpy_classes_keep_numeric_identity(self):
        self.assertEqual(self._save(np.int64(2), np.float32(.8),
                                    np.array([1, 2], dtype=np.int64))['results']['row_count'], 1)

    def test_numpy_class_order_not_sorted(self):
        self.assertEqual(self._save(np.str_('Z'), np.float64(.8),
                                    np.array(['Z', 'A']))['results']['row_count'], 1)

    def test_unknown_numpy_class_rejected_before_write(self):
        before = len(self.repo.list_artifacts(self.scope))
        with self.assertRaises(ValueError):
            self._save(np.str_('X'), np.float64(.8), np.array(['A', 'B']))
        self.assertEqual(before, len(self.repo.list_artifacts(self.scope)))

    def test_nonfinite_numpy_rate_rejected_before_write(self):
        before = len(self.repo.list_artifacts(self.scope))
        with self.assertRaises(ValueError):
            self._save(np.str_('B'), np.float64('nan'), np.array(['A', 'B']))
        self.assertEqual(before, len(self.repo.list_artifacts(self.scope)))


if __name__ == '__main__':
    unittest.main()
