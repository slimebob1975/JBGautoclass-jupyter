"""Revision 240: native JBG constructor and TaskRunner CSV prediction probe.

Unlike earlier AST-extracted tests, this imports complete production classes.
A missing optional ML dependency skips this probe (not a passing end-to-end test).
The saved-model loader is still controlled; full application bootstrap is next.
"""
import unittest
from unittest.mock import Mock

from test_csv_taskrunner_handler_bridge_232 import CSVTaskRunnerHandlerBridge232, TASKS

try:
    from JBGHandler import JBGHandler
    from JBGTaskRunner import TaskRunner
except (ImportError, OSError) as error:
    IMPORT_ERROR = str(error)
    JBGHandler = None
    TaskRunner = None
else:
    IMPORT_ERROR = None


@unittest.skipIf(JBGHandler is None, 'Native JBG import unavailable: ' + str(IMPORT_ERROR))
class CSVNativeConstructor240(CSVTaskRunnerHandlerBridge232):
    # Parent's task-bridge tests are not inherited into this class.
    test_dispatch_loads_rows_predicts_and_publishes = None
    test_sql_layer_mismatch_rejected_before_read = None
    test_training_sequence_rejected_before_read = None
    test_prediction_fails_closed_for_invalid_class = None

    def setUp(self):
        super().setUp()
        original = self.handler
        native = JBGHandler(datalayer=self.layer, config=self.config, logger=self.logger)
        native.add_handler('dataset')
        native.add_handler('model')
        native.add_handler('predictions')
        dataset = native.get_handler('dataset')
        model = native.get_handler('model')
        predictions = native.get_handler('predictions')
        # Keep the real constructors and handler registry. Use the already
        # prepared synthetic dataset and model loader to limit this probe.
        dataset.__dict__.update(self.dataset.__dict__)
        dataset.handler = native
        predictions.__dict__.update(self.predictions.__dict__)
        predictions.handler = native
        model.load_model = self.model.load_model
        model.model = self.model.model
        self.handler = native
        self.model = model
        self.dataset = dataset
        self.predictions = predictions
        self.runner = TaskRunner(datalayer=self.layer, config=self.config,
                                 logger=self.logger, handler=native)
        self.assertIsNot(native, original)

    def test_native_handlers_and_runner_publish_csv(self):
        self.assertIs(self.handler.get_handler('dataset'), self.dataset)
        self.assertIs(self.handler.get_handler('model'), self.model)
        self.assertIs(self.handler.get_handler('predictions'), self.predictions)
        self.assertFalse(self.runner.run(TASKS))
        self.assertEqual(len(self._artifacts()), 1)

    def test_native_runner_rejects_wrong_datalayer_before_read(self):
        self.handler.datalayer = Mock()
        self.layer.get_dataset = Mock(side_effect=AssertionError('unexpected CSV read'))
        with self.assertRaisesRegex(ValueError, 'data-layer mismatch'):
            self.runner.run(TASKS)
        self.layer.get_dataset.assert_not_called()
        self.assertFalse(self._artifacts())


if __name__ == '__main__':
    unittest.main()
