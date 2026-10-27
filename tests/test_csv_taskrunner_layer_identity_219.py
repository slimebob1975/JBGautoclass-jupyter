"""Guard the CSV read/write facade identity before any task dispatch."""
import unittest
from unittest.mock import Mock

from Storage.CSVPredictionDataLayer import CSVPredictionDataLayer
from test_taskrunner_source_gate_213 import _runner, _runner_method

TASKS = ['load_model', 'fetch_data', 'load_dataset', 'separate_dataset', 'make_predictions']


class CSVTaskRunnerLayerIdentity219(unittest.TestCase):
    def test_same_facade_is_accepted(self):
        facade = CSVPredictionDataLayer(Mock(), Mock())
        runner = _runner('csv', facade)
        runner.load_model__task = Mock(return_value={'early_exit': True})
        self.assertTrue(_runner_method()(runner, TASKS))
        runner.logger.initiate_progress.assert_called_once()

    def test_sql_writer_is_rejected_before_any_task(self):
        runner = _runner('csv', CSVPredictionDataLayer(Mock(), Mock()))
        runner.handler.datalayer = Mock()
        runner.load_model__task = Mock()
        with self.assertRaisesRegex(ValueError, 'data-layer mismatch'):
            _runner_method()(runner, TASKS)
        runner.load_model__task.assert_not_called()
        runner.logger.initiate_progress.assert_not_called()

    def test_different_csv_facade_is_rejected(self):
        facade = CSVPredictionDataLayer(Mock(), Mock())
        runner = _runner('csv', facade)
        runner.handler.datalayer = CSVPredictionDataLayer(Mock(), Mock())
        with self.assertRaisesRegex(ValueError, 'data-layer mismatch'):
            _runner_method()(runner, TASKS)

    def test_missing_handler_data_layer_is_rejected(self):
        runner = _runner('csv', CSVPredictionDataLayer(Mock(), Mock()))
        del runner.handler.datalayer
        with self.assertRaisesRegex(ValueError, 'data-layer mismatch'):
            _runner_method()(runner, TASKS)

    def test_sql_legacy_path_not_subject_to_identity_guard(self):
        runner = _runner('sql', Mock())
        runner.handler.datalayer = Mock()
        self.assertFalse(_runner_method()(runner, []))


if __name__ == '__main__':
    unittest.main()
