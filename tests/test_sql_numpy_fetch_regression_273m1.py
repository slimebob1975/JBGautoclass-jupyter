"""SQL source must accept NumPy data without evaluating its truth value."""
import ast
import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


def _fetch_task():
    # Isolate the production method without importing optional ML dependencies.
    path = pathlib.Path(__file__).resolve().parents[1] / "src" / "JBGTaskRunner.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "TaskRunner")
    method = next(node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "fetch_data__task")
    namespace = {"DatasetException": type("DatasetException", (Exception,), {})}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["fetch_data__task"]


class SQLNumPyFetchRegression273m1(unittest.TestCase):
    def setUp(self):
        self.fetch = _fetch_task()

    def _runner(self, kind, data):
        config = SimpleNamespace(data_source=SimpleNamespace(kind=kind), should_train=lambda: True, should_predict=lambda: False)
        return SimpleNamespace(config=config, datalayer=SimpleNamespace(get_dataset=lambda: data), logger=SimpleNamespace(print_task_header=Mock()))

    def test_sql_numpy_matrix_is_not_tested_for_truth_value(self):
        rows = np.arange(858).reshape(286, 3)
        result = self.fetch(self._runner("sql", rows))
        self.assertIs(result["data"], rows)

    def test_sql_none_returns_early_exit(self):
        self.assertEqual(self.fetch(self._runner("sql", None)), {"early_exit": True})

    def test_csv_empty_list_returns_early_exit(self):
        # Validation of train-only CSV rows is owned by CSVTrainingDataLayer.
        self.assertEqual(self.fetch(self._runner("csv", [])), {"early_exit": True})


if __name__ == "__main__":
    unittest.main()
