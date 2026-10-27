"""Regression coverage of source-aware misprediction analysis and reporting."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def production_methods():
    source = Path(__file__).resolve().parents[1] / "src" / "JBGHandler.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    wanted = {"evaluate_mispredictions", "most_mispredicted", "_get_rid_of_TOP_stuffs"}
    nodes = []
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "PredictionsHandler":
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name in wanted:
                    node.decorator_list = []
                    node.returns = None
                    for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                        arg.annotation = None
                    nodes.append(node)
    assert {node.name for node in nodes} == wanted
    namespace = {"pd": pd, "np": np, "accuracy_score": accuracy_score,
                 "prepare_model_input": lambda x, model: x,
                 "call_with_sparse_input_retry": lambda fn, x, **kw: fn(x),
                 "Helpers": SimpleNamespace(save_matrix_as_csv=Mock(),
                                            create_download_link=Mock(return_value="link"))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(source), "exec"), namespace)
    return {name: namespace[name] for name in wanted}


METHODS = production_methods()


class Mispredicted223(unittest.TestCase):
    def handler(self, kind):
        config = SimpleNamespace(data_source=SimpleNamespace(kind=kind),
                                 should_display_mispredicted=lambda: True,
                                 get_id_column_name=lambda: "row_id")
        logger = Mock()
        obj = SimpleNamespace(handler=SimpleNamespace(config=config, logger=logger,
                                                        read_data_query="SELECT TOP(3) * FROM source"),
                              LIMIT_MISPREDICTED=10, NO_MODEL="none",
                              CROSS_TRAINED_MODEL="cross", RETRAINED_MODEL="retrained")
        for name, func in METHODS.items():
            setattr(obj, name, func.__get__(obj))
        return obj, logger

    def data(self):
        ids = ["005", "ABC-17", "010"]
        x = pd.DataFrame({"value": [0, 1, 2]}, index=ids)
        y = pd.Series(["A", "B", "A"], index=ids)
        model = SimpleNamespace(predict=lambda X: np.array([{"005": "B", "ABC-17": "B", "010": "A"}[k] for k in X.index]),
                                predict_proba=lambda X: np.tile([0.2, 0.8], (len(X), 1)),
                                classes_=np.array(["A", "B"]))
        return x, y, model

    def test_csv_mispredicted_preserves_opaque_ids_and_excludes_sql(self):
        obj, logger = self.handler("csv")
        x, y, model = self.data()
        obj.most_mispredicted(x, model, model, x, y)
        self.assertEqual(obj.num_mispredicted, 1)
        self.assertEqual(obj.X_most_mispredicted.index.tolist(), ["005"])
        obj.evaluate_mispredictions("synthetic.csv")
        logger.print_query.assert_not_called()
        namespace_helpers = METHODS["evaluate_mispredictions"].__globals__["Helpers"]
        namespace_helpers.save_matrix_as_csv.assert_called()
        self.assertEqual(obj.X_mispredicted.index.tolist(), ["005"])

    def test_csv_reporting_never_reads_sql_query(self):
        obj, logger = self.handler("csv")
        obj.handler.read_data_query = None
        obj.X_most_mispredicted = pd.DataFrame({"a": [1]}, index=["ABC-17"])
        obj.X_mispredicted = obj.X_most_mispredicted.copy()
        obj.num_mispredicted = 1
        obj.evaluate_mispredictions("synthetic.csv")
        logger.print_query.assert_not_called()

    def test_sql_retains_numeric_query(self):
        obj, logger = self.handler("sql")
        obj.X_most_mispredicted = pd.DataFrame({"a": [1]}, index=[5])
        obj.X_mispredicted = obj.X_most_mispredicted.copy()
        obj.num_mispredicted = 1
        obj.evaluate_mispredictions("synthetic.csv")
        self.assertIn("WHERE row_id IN (5)", logger.print_query.call_args.args[1])

    def test_empty_mispredictions_write_nothing(self):
        obj, logger = self.handler("csv")
        obj.X_most_mispredicted = pd.DataFrame()
        obj.evaluate_mispredictions("synthetic.csv")
        logger.print_query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
