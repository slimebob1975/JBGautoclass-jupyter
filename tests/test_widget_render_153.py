"""Regression tests for guarded reclassification display (revision 153)."""
import logging
from types import SimpleNamespace

import ipywidgets as widgets
import pandas as pd

from GUI.Widgets import Widgets


def _view():
    output = widgets.Output()
    grid = widgets.GridBox()
    return SimpleNamespace(
        MISPREDICTED_RENDER_LIMIT=40,
        mispredicted_output=output,
        mispredicted_gridbox=grid,
        get_text_widget=lambda value: widgets.Label(value=str(value)),
        reclassify_dropdown=lambda classes, name: widgets.Dropdown(
            options=[("Keep", 0)] + [(c, (c, name)) for c in classes], value=0
        ),
        create_inpage_styles=lambda section: widgets.HTML(value=""),
    )


def test_bounded_render_and_columns():
    view = _view()
    data = pd.DataFrame({"text": ["x"] * 60, "score": range(60)})
    Widgets.handle_mispredicted(view, data, ["wrong", "correct"])
    assert len(view.mispredicted_gridbox.children) == (40 + 1) * 4
    assert view.mispredicted_gridbox.layout.grid_template_columns == "repeat(4, max-content)"


def test_duplicate_index_is_one_row_per_position():
    view = _view()
    data = pd.DataFrame({"x": [1, 2]}, index=["dup", "dup"])
    Widgets.handle_mispredicted(view, data, ["good"])
    assert len(view.mispredicted_gridbox.children) == 3 * 3


def test_render_failure_is_logged_and_swallowed(caplog):
    view = _view()
    view.get_text_widget = lambda value: (_ for _ in ()).throw(RuntimeError("test failure"))
    with caplog.at_level(logging.ERROR):
        Widgets.handle_mispredicted(view, pd.DataFrame({"x": [1]}), ["ok"])
    assert "Mispredicted widget rendering failed" in caplog.text
    assert "test failure" in caplog.text
