"""CSV TaskRunner must stop after failing, even if the logger does not raise.

Uses the production TaskRunner methods extracted by the 232 integration fixture.
It does not claim full application bootstrap or SQL-connection interception.
"""
import unittest
from unittest.mock import Mock

from test_csv_taskrunner_handler_bridge_232 import CSVTaskRunnerHandlerBridge232, TASKS


class CSVTaskRunnerFailClosed239(CSVTaskRunnerHandlerBridge232):
    # Parent smoke tests are separately exercised by the same discovery suite.
    def test_model_load_error_stops_before_csv_read(self):
        self.logger.abort_cleanly.side_effect = None
        self.model.load_model = Mock(side_effect=ValueError('synthetic load failure'))
        self.layer.get_dataset = Mock(side_effect=AssertionError('CSV should not be read'))
        with self.assertRaisesRegex(ValueError, 'synthetic load failure'):
            self.runner.run(TASKS)
        self.layer.get_dataset.assert_not_called()
        self.assertFalse(self._artifacts())

    def test_csv_reader_error_stops_before_prediction(self):
        self.logger.abort_cleanly.side_effect = None
        self.layer.get_dataset = Mock(side_effect=ValueError('synthetic read failure'))
        self.handler.save_predictions = Mock(side_effect=AssertionError('publication attempted'))
        with self.assertRaisesRegex(ValueError, 'synthetic read failure'):
            self.runner.run(TASKS)
        self.handler.save_predictions.assert_not_called()
        self.assertFalse(self._artifacts())

    def test_csv_publication_error_is_not_silent(self):
        self.logger.abort_cleanly.side_effect = None
        self.handler.save_predictions = Mock(return_value={'error': 'synthetic save failure'})
        with self.assertRaisesRegex(Exception, 'CSV prediction publication failed'):
            self.runner.run(TASKS)
        self.handler.save_predictions.assert_called_once()
        self.assertFalse(self._artifacts())


if __name__ == '__main__':
    unittest.main()
