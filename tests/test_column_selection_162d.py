"""Regression checks for class NULL semantics and atomic selector refresh."""
import ast
import pathlib
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
SQL = (ROOT / "src/SQLDataLayer.py").read_text(encoding="utf-8")
GUI = (ROOT / "src/GUI/Widgets.py").read_text(encoding="utf-8")

class Selection162d(unittest.TestCase):
    def test_sql_null_not_counted_as_class(self):
        self.assertIn("COUNT(DISTINCT {expression}) AS cardinality_{index}", SQL)
        self.assertNotIn("+ MAX(CASE WHEN", SQL)
    def test_unique_ids_initialized_before_class_refresh(self):
        node = ast.parse(GUI)
        method = next(n for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n.name == "update_class_id_data_columns")
        source = ast.get_source_segment(GUI, method)
        self.assertLess(source.index("self.unique_id_columns ="), source.index("self.refresh_class_column_options()"))
    def test_reentrant_observer_suppressed(self):
        self.assertIn('getattr(self.widgets, "_updating_column_options", False)', GUI)
        self.assertIn('if self.widgets.class_column.value is None:', GUI)
    def test_class_refresh_state_resets(self):
        self.assertIn('self._updating_column_options = True', GUI)
        self.assertIn('finally:', GUI)
        self.assertIn('self._updating_column_options = False', GUI)

if __name__ == "__main__": unittest.main()
