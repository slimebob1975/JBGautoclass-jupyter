"""Dependency-light source boundary checks for the common task runner."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from Storage.CSVPredictionDataLayer import CSVPredictionDataLayer


def _runner_method():
    path = Path(__file__).resolve().parents[1] / "src" / "JBGTaskRunner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TaskRunner")
    method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    method.decorator_list = []
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), ns)
    return ns["run"]


def _runner(kind, layer, *, training=False, suite=False):
    config = SimpleNamespace(data_source=SimpleNamespace(kind=kind), should_train=lambda: training, get_text_column_names=lambda: [])
    logger = Mock()
    handler = Mock()
    handler.get_handler.return_value = Mock()
    runner = SimpleNamespace(config=config, datalayer=layer, logger=logger, handler=handler, regression_suite=suite)
    return runner


class TaskRunnerSourceGateTests(TestCase):
    def test_csv_prediction_with_prepared_layer_enters_common_runner(self):
        layer = CSVPredictionDataLayer(Mock(), Mock())
        runner = _runner("csv", layer)
        runner.load_model__task = Mock(return_value={"early_exit": True})
        tasks = ["load_model", "fetch_data", "load_dataset", "separate_dataset", "make_predictions"]
        self.assertTrue(_runner_method()(runner, tasks))
        runner.logger.initiate_progress.assert_called_once_with(number_of_tasks=5)

    def test_csv_rejects_sql_layer_before_task(self):
        runner = _runner("csv", Mock())
        with self.assertRaisesRegex(ValueError, "prepared prediction"):
            _runner_method()(runner, [])
        runner.logger.initiate_progress.assert_not_called()

    def test_csv_training_remains_disabled(self):
        runner = _runner("csv", CSVPredictionDataLayer(Mock(), Mock()), training=True)
        with self.assertRaises(ValueError):
            _runner_method()(runner, [])

    def test_csv_regression_remains_disabled(self):
        runner = _runner("csv", CSVPredictionDataLayer(Mock(), Mock()), suite=True)
        with self.assertRaises(ValueError):
            _runner_method()(runner, [])

    def test_sql_legacy_and_unknown_source(self):
        sql = _runner("sql", Mock(), training=True)
        self.assertFalse(_runner_method()(sql, []))
        sql.logger.initiate_progress.assert_called_once()
        unsupported = _runner("other", Mock())
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _runner_method()(unsupported, [])
