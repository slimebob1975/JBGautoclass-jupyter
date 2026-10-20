"""Focused source-level checks for the non-invasive revision 158 diagnostics."""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parents[1]


class VoilaOutput158Tests(unittest.TestCase):
    def test_gui_handler_compiles_and_tracks_capture_lifecycle(self):
        source = (ROOT / "src/GUIHandler.py").read_text(encoding="utf-8")
        compile(source, "src/GUIHandler.py", "exec")
        self.assertIn('Script output capture: entering', source)
        self.assertIn('Script output capture: exited', source)
        self.assertIn('post-run GUI: completed', source)

    def test_output_clear_boundaries_are_instrumented(self):
        source = (ROOT / "src/GUI/Widgets.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsInstance(tree, ast.Module)
        self.assertIn('Script output widget: clear begin', source)
        self.assertIn('Script output widget: clear end', source)

    def test_launch_keeps_xsrf_protection_and_marks_shutdown_403(self):
        source = (ROOT / "start_application_template.ps1").read_text(encoding="utf-8")
        self.assertIn('403 POST /voila/api/shutdown/', source)
        self.assertIn('Runtime versions:', source)
        self.assertNotIn('disable_check_xsrf', source)
        self.assertNotIn('xsrf_cookies=False', source)


if __name__ == "__main__":
    unittest.main()
