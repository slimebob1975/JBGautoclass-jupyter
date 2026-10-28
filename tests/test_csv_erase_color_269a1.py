"""Verify CSV model deletion is always an orange warning action."""
import ast
from pathlib import Path
import unittest

TREE = ast.parse((Path(__file__).resolve().parents[1] / "src/GUI/Widgets.py").read_text())

class EraseModelColorTests(unittest.TestCase):
    def _method(self, name):
        return next(node for node in ast.walk(TREE) if isinstance(node, ast.FunctionDef) and node.name == name)

    def test_button_created_blocked(self):
        self.assertIn("set_action_state(self._csv_erase_model_control, 'blocked')", ast.unparse(self._method('_csv_erase_model_button')))

    def test_enabled_button_orange(self):
        self.assertIn("set_action_state(button, 'warning' if enabled else 'blocked')", ast.unparse(self._method('_csv_update_erase_model_button')))

    def test_confirmation_remains_orange(self):
        self.assertIn("set_action_state(button, 'warning')", ast.unparse(self._method('_csv_erase_model')))
