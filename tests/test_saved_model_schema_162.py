"""Revision 162: source-schema safety and legacy compatibility."""
import ast
from pathlib import Path
import unittest

import pandas as pd
from JBGModelCompatibility import pandas_dtype_family

SOURCE = Path(__file__).resolve().parents[1] / "src" / "JBGTransformers.py"


def load_converter_methods():
    """Isolate the two methods to test schema behavior without torch/keras imports."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "TextDataToNumbersConverter")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "transform")
    # Run the actual method with lightweight stand-ins for the legacy transform.
    block = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(block)
    space = {"pd": pd, "check_is_fitted": lambda obj: None, "pandas_dtype_family": pandas_dtype_family}
    exec(compile(block, str(SOURCE), "exec"), space)
    return space["transform"]


class Schema162(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transform_method = staticmethod(load_converter_methods())

    def make_converter(self, names=None):
        class Converter:
            text_columns_ = []
            category_columns_ = []
        converter = Converter()
        if names is not None:
            converter.input_feature_columns_ = names
        return converter

    def test_same_schema_and_different_order(self):
        conv = self.make_converter(("a", "b"))
        actual = self.transform_method(conv, pd.DataFrame({"b": [2], "a": [1]}))
        self.assertEqual(list(actual.columns), ["a", "b"])

    def test_missing_and_extra_columns_refused(self):
        conv = self.make_converter(("a", "b"))
        with self.assertRaisesRegex(ValueError, "missing=.*'b'"):
            self.transform_method(conv, pd.DataFrame({"a": [1]}))
        with self.assertRaisesRegex(ValueError, "unexpected=.*'c'"):
            self.transform_method(conv, pd.DataFrame({"a": [1], "b": [2], "c": [3]}))

    def test_duplicate_columns_refused(self):
        conv = self.make_converter(("a", "b"))
        df = pd.DataFrame([[1, 2]], columns=["a", "a"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.transform_method(conv, df)

    def test_legacy_without_schema_still_works(self):
        conv = self.make_converter()
        df = pd.DataFrame({"b": [2], "a": [1]})
        actual = self.transform_method(conv, df)
        self.assertEqual(list(actual.columns), ["b", "a"])

    def test_dtype_family_mismatch_is_refused_for_new_models(self):
        conv = self.make_converter(("a", "b"))
        conv.input_feature_type_families_ = {"a": "numeric", "b": "numeric"}
        with self.assertRaisesRegex(ValueError, "changed dtype family"):
            self.transform_method(conv, pd.DataFrame({"a": [1], "b": ["two"]}))

    def test_schema_is_recorded_on_fit(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('self.input_feature_columns_ = tuple(X.columns)', source)
        self.assertIn('self.input_feature_type_families_', source)
        self.assertIn('self.configured_text_columns_', source)
