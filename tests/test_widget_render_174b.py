"""Revision 174b: wider reclassification preview budget."""
from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
WIDGETS_SOURCE = ROOT / "src" / "GUI" / "Widgets.py"


class WidgetRender174b(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = WIDGETS_SOURCE.read_text(encoding="utf-8")

    def test_default_widget_budget_is_raised_but_still_bounded(self):
        self.assertIn("MISPREDICTED_WIDGET_LIMIT = 640", self.source)
        self.assertIn('getattr(self, "MISPREDICTED_WIDGET_LIMIT", 640)', self.source)
        self.assertIn("MISPREDICTED_RENDER_LIMIT = 40", self.source)


if __name__ == "__main__":
    unittest.main()
