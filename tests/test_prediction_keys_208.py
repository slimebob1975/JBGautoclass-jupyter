"""Revision 208: preserve CSV keys without altering legacy SQL identifiers."""
import ast
from pathlib import Path
import unittest

from JBGPredictionKeys import prediction_key


class PredictionKeysTests(unittest.TestCase):
    def test_csv_leading_zeroes_preserved(self):
        self.assertEqual(prediction_key("00123", "csv"), "00123")
        self.assertEqual(prediction_key("abc-007", "csv"), "abc-007")

    def test_csv_invalid_keys_fail_closed(self):
        for value in ("", None, 1, True, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prediction_key(value, "csv")

    def test_sql_integer_conversion_unchanged(self):
        self.assertEqual(prediction_key("00123", "sql"), 123)
        self.assertEqual(prediction_key(12, "sql"), 12)
        with self.assertRaises(ValueError):
            prediction_key("ABC", "sql")

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):
            prediction_key("00123", "files")

    def test_handler_uses_source_aware_conversion_and_shared_sink(self):
        src = (Path(__file__).resolve().parents[1] / "src" / "JBGHandler.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        candidates = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_prediction_results"]
        self.assertEqual(len(candidates), 1)
        body = ast.get_source_segment(src, candidates[0])
        self.assertIn('prediction_key(k, source_kind)', body)
        self.assertNotIn('"key": int(k)', body)
        self.assertIn('self.handler.datalayer.save_prediction_data(', src)


if __name__ == "__main__":
    unittest.main()
