"""Regression tests for source-aware IDs in the real DatasetHandler methods."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from JBGPredictionKeys import prediction_key


def extract_methods():
    source = Path(__file__).resolve().parents[1] / "src" / "JBGHandler.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    methods = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "DatasetHandler":
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name in (
                    "shuffle_dataset", "split_keys_from_dataset", "_normalize_class_column"
                ):
                    node.returns = None
                    for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                        arg.annotation = None
                    methods[node.name] = node
    helpers = SimpleNamespace(get_rid_of_decimals=lambda v: int(v))
    namespace = {"pd": pd, "np": np, "Helpers": helpers,
                 "prediction_key": prediction_key, "DatasetException": ValueError}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=list(methods.values()), type_ignores=[])), str(source), "exec"), namespace)
    return namespace


METHODS = extract_methods()


class SourceAwareDataset222(unittest.TestCase):
    def handler(self, kind):
        config = SimpleNamespace(data_source=SimpleNamespace(kind=kind),
                                 get_id_column_name=lambda: "row_id",
                                 get_class_column_name=lambda: "label")
        logger = Mock()
        obj = SimpleNamespace(handler=SimpleNamespace(config=config, logger=logger))
        obj.shuffle_dataset = METHODS["shuffle_dataset"].__get__(obj)
        obj.split_keys_from_dataset = METHODS["split_keys_from_dataset"].__get__(obj)
        return obj

    def data(self):
        return pd.DataFrame({"row_id": ["010", "002", "A-1"],
                             "label": ["yes", None, None], "value": [1, 2, 3]})

    def test_csv_preserves_opaque_and_leading_zero_ids(self):
        obj = self.handler("csv")
        frame, keys = obj.split_keys_from_dataset(self.data(), "row_id")
        self.assertEqual(keys.tolist(), ["010", "002", "A-1"])
        self.assertEqual(frame.index.tolist(), keys.tolist())
        self.assertNotIn("row_id", frame.columns)

    def test_csv_sorts_known_before_unknown_with_opaque_ids(self):
        obj = self.handler("csv")
        sorted_data = obj.shuffle_dataset(self.data())
        self.assertEqual(sorted_data.row_id.tolist(), ["010", "002", "A-1"])

    def test_csv_rejects_duplicate_and_empty_identifiers(self):
        obj = self.handler("csv")
        for identifiers in (["005", "005"], ["005", ""]):
            with self.subTest(identifiers=identifiers):
                frame = pd.DataFrame({"row_id": identifiers, "label": [None, None]})
                with self.assertRaises(ValueError):
                    obj.split_keys_from_dataset(frame, "row_id")

    def test_sql_preserves_integer_index_contract(self):
        obj = self.handler("sql")
        data = pd.DataFrame({"row_id": ["010", "002"], "label": [None, "yes"]})
        frame, keys = obj.split_keys_from_dataset(data, "row_id")
        self.assertEqual(keys.tolist(), [10, 2])
        self.assertEqual(frame.index.tolist(), [10, 2])
        self.assertTrue(pd.api.types.is_integer_dtype(frame.index.dtype))

    def test_sql_shuffle_retains_legacy_numeric_sort(self):
        obj = self.handler("sql")
        data = pd.DataFrame({"row_id": ["010", "002", "003"],
                             "label": [None, "yes", "no"]})
        self.assertEqual(obj.shuffle_dataset(data).row_id.tolist(),
                         ["002", "003", "010"])


if __name__ == "__main__":
    unittest.main()
