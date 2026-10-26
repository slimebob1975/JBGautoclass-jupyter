"""CSV GUI inspection regression checks; no live SQL dependency."""
import tempfile
import unittest
from pathlib import Path

import sys
import types
from unittest.mock import patch

# Minimal widget facsimiles for environments without ipywidgets.
class FakeWidget:
    def __init__(self, value="", description="", **kwargs):
        self.value = value
        self.description = description
        self.accept = kwargs.get("accept", "")
        self.multiple = kwargs.get("multiple", False)
        self.children = kwargs.get("children", ())
        self._observers = []
    def on_click(self, callback):
        self.callback = callback
    def observe(self, callback, names=None):
        self._observers.append(callback)

class FakeBox(FakeWidget):
    def __init__(self, children, **kwargs):
        super().__init__(children=children, **kwargs)

fake_widgets = types.ModuleType("ipywidgets")
fake_widgets.FileUpload = FakeWidget
fake_widgets.Layout = FakeWidget
fake_widgets.Text = FakeWidget
fake_widgets.HTML = FakeWidget
fake_widgets.Button = FakeWidget
fake_widgets.VBox = FakeBox
fake_widgets.HBox = FakeBox
fake_widgets.ToggleButtons = FakeWidget
# Exercise real ipywidgets when installed (Windows/Voila). Only stub the
# dependency in minimal CI environments that do not provide it.
try:
    import ipywidgets  # noqa: F401
except ImportError:
    with patch.dict(sys.modules, {"ipywidgets": fake_widgets}):
        from GUI.CSVSourcePanel import CSVSourcePanel, DataSourceSelector
else:
    from GUI.CSVSourcePanel import CSVSourcePanel, DataSourceSelector


class CSVSourcePanelTests(unittest.TestCase):
    def test_validation_counts_without_data_or_path_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "confidential.csv"
            path.write_text("uid;target;secret\n001;A;private_value\n002;B;private_value\n", encoding="utf-8")
            panel = CSVSourcePanel()
            panel.path.value = str(path)
            panel.delimiter.value = ";"
            self.assertTrue(panel.validate())
            self.assertIn("2 rows and 3 columns", panel.status.value)
            self.assertNotIn(str(path), panel.status.value)
            self.assertNotIn("private_value", panel.status.value)
            self.assertIn("Dataset columns", panel.status.value)
            self.assertFalse(hasattr(panel, "id_column"))
            self.assertFalse(hasattr(panel, "class_column"))

    def test_invalid_input_is_fail_closed_and_private(self):
        panel = CSVSourcePanel()
        panel.path.value = "/some/private/missing.csv"
        self.assertFalse(panel.validate())
        self.assertNotIn("private", panel.status.value)
        panel.delimiter.value = "too-long"
        self.assertFalse(panel.validate())

    def test_no_widget_execution_action(self):
        panel = CSVSourcePanel()
        self.assertEqual(panel.validate_button.description, "Validate CSV")
        self.assertFalse(hasattr(panel, "run_button"))

    def test_source_view_switches_without_changing_execution(self):
        try:
            import ipywidgets as real_widgets
        except ImportError:
            sql = FakeWidget(value="SQL connection")
            csv = FakeWidget(value="CSV inspection")
        else:
            sql = real_widgets.HTML(value="SQL connection")
            csv = real_widgets.HTML(value="CSV inspection")
        selector = DataSourceSelector(sql, csv)
        self.assertEqual(tuple(selector.body.children), (sql,))
        selector.choice.value = "csv"
        selector._source_changed({"name": "value", "new": "csv"})
        self.assertEqual(tuple(selector.body.children), (csv,))
        selector.choice.value = "sql"
        selector._source_changed({"name": "value", "new": "sql"})
        self.assertEqual(tuple(selector.body.children), (sql,))

    def test_rejects_bad_csv_shape_without_leaking_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "confidential.csv"
            path.write_text("id;target\n1;private;extra\n", encoding="utf-8")
            panel = CSVSourcePanel()
            panel.path.value = str(path)
            panel.delimiter.value = ";"
            self.assertFalse(panel.validate())
            self.assertNotIn(str(path), panel.status.value)
            self.assertNotIn("private", panel.status.value)

    def test_data_source_container_uses_supported_box_type(self):
        # Static GUI construction regression: create_form only supports Box/HBox.
        import ast
        module = ast.parse((Path(__file__).resolve().parents[1] / "src" / "GUI" / "Widgets.py").read_text(encoding="utf-8"))
        widgets_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Widgets")
        connection = next(node for node in widgets_class.body if isinstance(node, ast.FunctionDef) and node.name == "connection_form")
        section_calls = [node for node in ast.walk(connection) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "create_section_form"]
        data_source_calls = [node for node in section_calls if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "Data source"]
        self.assertEqual(len(data_source_calls), 1)
        boxtype = data_source_calls[0].args[2]
        self.assertIsInstance(boxtype, ast.Attribute)
        self.assertEqual(boxtype.attr, "Box")
