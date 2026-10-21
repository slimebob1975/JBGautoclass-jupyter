"""Revision 163: bounded widget lifecycle for the misprediction preview."""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WIDGETS_SOURCE = ROOT / "src" / "GUI" / "Widgets.py"
GUI_HANDLER_SOURCE = ROOT / "src" / "GUIHandler.py"


class FakeWidget:
    def __init__(self, value="", **kwargs):
        self.value = value
        self.kwargs = kwargs
        self.closed = False
        self.classes = []

    def add_class(self, class_name):
        self.classes.append(class_name)
        return self

    def close(self):
        self.closed = True


class FakeWidgets:
    Label = FakeWidget
    HTML = FakeWidget


class FakeLayout:
    def __init__(self):
        self.grid_template_columns = ""


class FakeGrid:
    def __init__(self):
        self.children = ()
        self.layout = FakeLayout()


class FakeOutput:
    def __init__(self):
        self.clear_calls = []

    def clear_output(self, wait=False):
        self.clear_calls.append(wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def load_widgets_contract():
    """Execute only the revision-163 constants/methods with lightweight stand-ins."""
    tree = ast.parse(WIDGETS_SOURCE.read_text(encoding="utf-8"))
    source_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Widgets"
    )
    wanted_methods = {"clear_mispredicted_output", "handle_mispredicted"}
    wanted_constants = {"MISPREDICTED_RENDER_LIMIT", "MISPREDICTED_WIDGET_LIMIT"}
    body = []
    for node in source_class.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_methods:
            body.append(node)
        elif isinstance(node, ast.Assign):
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if names & wanted_constants:
                body.append(node)
    isolated = ast.Module(
        body=[ast.ClassDef(
            name="Widgets",
            bases=[],
            keywords=[],
            body=body,
            decorator_list=[],
        )],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated)
    namespace = {
        "DataFrame": pd.DataFrame,
        "display": lambda *args, **kwargs: None,
        "logging": logging,
        "widgets": FakeWidgets,
    }
    exec(compile(isolated, str(WIDGETS_SOURCE), "exec"), namespace)
    return namespace["Widgets"]


def make_view(widget_budget=240, row_limit=40):
    return SimpleNamespace(
        MISPREDICTED_RENDER_LIMIT=row_limit,
        MISPREDICTED_WIDGET_LIMIT=widget_budget,
        mispredicted_output=FakeOutput(),
        mispredicted_gridbox=FakeGrid(),
        get_text_widget=lambda value: FakeWidget(value=str(value)),
        reclassify_dropdown=lambda classes, name: FakeWidget(value=(tuple(classes), name)),
        create_inpage_styles=lambda section: FakeWidget(value=section),
    )


class WidgetRender163(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Widgets = load_widgets_contract()

    def test_widget_budget_bounds_wide_preview(self):
        view = make_view(widget_budget=100)
        data = pd.DataFrame({f"c{i}": range(50) for i in range(20)})

        rendered = self.Widgets.handle_mispredicted(view, data, ["ok", "bad"])

        # 20 data columns + index + reclassify = 22 widgets per rendered grid row.
        # The header consumes one row, leaving room for three data rows: 4 * 22 = 88.
        self.assertTrue(rendered)
        self.assertEqual(len(view.mispredicted_gridbox.children), 88)
        self.assertLessEqual(len(view.mispredicted_gridbox.children), 100)

    def test_existing_40_row_behavior_is_preserved_for_small_tables(self):
        view = make_view(widget_budget=240, row_limit=40)
        data = pd.DataFrame({"text": ["x"] * 60, "score": range(60)})

        rendered = self.Widgets.handle_mispredicted(view, data, ["wrong", "correct"])

        self.assertTrue(rendered)
        self.assertEqual(len(view.mispredicted_gridbox.children), (40 + 1) * 4)

    def test_too_many_columns_skips_interactive_grid(self):
        view = make_view(widget_budget=20)
        data = pd.DataFrame({f"c{i}": [i] for i in range(25)})

        rendered = self.Widgets.handle_mispredicted(view, data, ["ok"])

        self.assertFalse(rendered)
        self.assertEqual(view.mispredicted_gridbox.children, ())

    def test_second_render_closes_first_render_models(self):
        view = make_view()
        data = pd.DataFrame({"text": ["a", "b"], "score": [1, 2]})

        self.assertTrue(self.Widgets.handle_mispredicted(view, data, ["ok"]))
        old_grid_widgets = tuple(view.mispredicted_gridbox.children)
        old_display_widgets = tuple(view._mispredicted_display_widgets)

        self.assertTrue(self.Widgets.handle_mispredicted(view, data.iloc[:1], ["ok"]))

        self.assertTrue(all(widget.closed for widget in old_grid_widgets))
        self.assertTrue(all(widget.closed for widget in old_display_widgets))
        self.assertGreaterEqual(len(view.mispredicted_output.clear_calls), 2)
        self.assertTrue(all(wait is False for wait in view.mispredicted_output.clear_calls))

    def test_render_failure_releases_partial_models_and_returns_fallback(self):
        view = make_view()
        created = []

        def fail_after_widget(value):
            widget = FakeWidget(value=str(value))
            created.append(widget)
            if len(created) == 2:
                raise RuntimeError("synthetic render failure")
            return widget

        view.get_text_widget = fail_after_widget
        rendered = self.Widgets.handle_mispredicted(
            view, pd.DataFrame({"x": [1], "y": [2]}), ["ok"]
        )

        self.assertFalse(rendered)
        self.assertEqual(view.mispredicted_gridbox.children, ())
        self.assertEqual(view._mispredicted_display_widgets, ())
        self.assertTrue(created[0].closed)

    def test_gui_paths_use_cleanup_and_disk_diagnostics(self):
        widgets_source = WIDGETS_SOURCE.read_text(encoding="utf-8")
        handler_source = GUI_HANDLER_SOURCE.read_text(encoding="utf-8")
        self.assertGreaterEqual(widgets_source.count("self.clear_mispredicted_output()"), 2)
        self.assertIn('"mispredicted widget: begin"', handler_source)
        self.assertIn('("rendered" if rendered else "fallback")', handler_source)


if __name__ == "__main__":
    unittest.main()
