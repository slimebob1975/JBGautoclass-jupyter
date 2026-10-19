import unittest
from pathlib import Path


class InitializationDiagnosticsTests(unittest.TestCase):
    def test_stage_markers_and_failure_logging_exist(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "GUIHandler.py").read_text(encoding="utf-8")
        for marker in ("callback entered", "persist last run: begin", "data layer construction: begin",
                       "data layer construction: end", "classifier construction: begin",
                       "classifier.run: begin", "classifier.run: end", "traceback.format_exc()"):
            with self.subTest(marker=marker):
                self.assertIn(marker, source)

    def test_button_entry_marker(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "GUI" / "Widgets.py").read_text(encoding="utf-8")
        self.assertIn('self.guihandler._classifier_diagnostic("start button action entered")', source)


if __name__ == "__main__":
    unittest.main()
