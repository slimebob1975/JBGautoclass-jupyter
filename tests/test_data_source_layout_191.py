"""Regression checks for the source chooser and run-history presentation."""
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DataSourceLayoutTests(unittest.TestCase):
    def test_display_order(self):
        settings = json.loads((ROOT / "src/GUI/default_settings.json").read_text(encoding="utf-8"))
        order = settings["display"]
        self.assertLess(order.index("connection_form"), order.index("models_form"))
        self.assertLess(order.index("models_form"), order.index("data_form"))
        self.assertEqual(order.index("regression_suite_form"), order.index("class_summary") + 1)

    def test_source_selector_has_no_redundant_label(self):
        tree = ast.parse((ROOT / "src/GUI/CSVSourcePanel.py").read_text(encoding="utf-8"))
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DataSourceSelector")
        init = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        toggles = [n for n in ast.walk(init) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "ToggleButtons"]
        self.assertEqual(len(toggles), 1)
        self.assertFalse(any(k.arg == "description" and isinstance(k.value, ast.Constant) and k.value.value == "Source:" for k in toggles[0].keywords))

    def test_run_history_hides_retired_button(self):
        tree = ast.parse((ROOT / "src/GUI/Widgets.py").read_text(encoding="utf-8"))
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Widgets")
        method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "regression_suite_form")
        attributes = {n.attr for n in ast.walk(method) if isinstance(n, ast.Attribute)}
        self.assertIn("repeat_last_run_button", attributes)
        self.assertNotIn("regression_suite_button", attributes)


if __name__ == "__main__":
    unittest.main()
