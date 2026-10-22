"""Revision 174a: static decorations must not allocate widget comm models."""
from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
WIDGETS_SOURCE = ROOT / "src" / "GUI" / "Widgets.py"


class WidgetRender174a(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = WIDGETS_SOURCE.read_text(encoding="utf-8")

    def test_mispredicted_decorations_use_static_ipython_html(self):
        self.assertIn("from IPython.display import HTML as DisplayHTML, display", self.source)
        self.assertIn('DisplayHTML("<h3>Reclassification table for most mispredicted</h3>")', self.source)
        self.assertIn("display_items.append(DisplayHTML(", self.source)
        self.assertNotIn('widgets.HTML("<h3>Reclassification table for most mispredicted</h3>")', self.source)

    def test_new_render_does_not_track_comm_backed_decorations(self):
        self.assertIn("self._mispredicted_display_widgets = ()", self.source)
        self.assertIn("return DisplayHTML(html_string)", self.source)

    def test_interactive_grid_is_retained(self):
        self.assertIn("grid.children = tuple(items)", self.source)
        self.assertIn("self.reclassify_dropdown(unique_classes, row_name)", self.source)
        self.assertIn("display(grid)", self.source)


if __name__ == "__main__":
    unittest.main()
