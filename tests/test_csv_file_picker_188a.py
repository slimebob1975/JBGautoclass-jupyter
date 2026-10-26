"""CSV browser picker behavior using synthetic uploads only."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_csv_source_panel_184 import CSVSourcePanel


class CSVFilePickerTests(unittest.TestCase):
    def test_upload_restricts_file_types_and_stages_csv(self):
        panel = CSVSourcePanel()
        self.assertEqual(panel.file_picker.accept, ".csv")
        self.assertFalse(panel.file_picker.multiple)
        panel._file_selected({"new": ({"name": "example.csv", "content": memoryview(b"uid,target\n001,A\n")},)})
        self.assertTrue(panel.path.value.endswith("dataset.csv"))
        self.assertEqual(panel.selected_file.value, "example.csv")
        self.assertNotIn(str(Path(panel.path.value).parent), panel.selected_file.value)
        self.assertTrue(panel.validate())
        self.assertEqual(Path(panel.path.value).read_bytes(), b"uid,target\n001,A\n")
        panel._upload_directory.cleanup()

    def test_rejects_mismatched_extension_and_multiple_files(self):
        panel = CSVSourcePanel()
        panel._file_selected({"new": ({"name": "example.txt", "content": b"uid,target\n1,A\n"},)})
        self.assertFalse(panel.path.value)
        self.assertFalse(panel.selected_file.value)
        panel._file_selected({"new": ({"name": "a.csv", "content": b"a"}, {"name": "b.csv", "content": b"b"})})
        self.assertFalse(panel.path.value)
        panel._upload_directory.cleanup()

    def test_semicolon_and_quoted_text_are_detected(self):
        panel = CSVSourcePanel()
        content = b'id;target;note\n001;A;"hello;world"\n002;B;"more;text"\n'
        panel._file_selected({"new": ({"name": "example.csv", "content": content},)})
        self.assertEqual(panel.delimiter.value, ";")
        self.assertEqual(panel.quotechar.value, '"')
        self.assertTrue(panel.validate())
        self.assertIn("2 rows and 3 columns", panel.status.value)
        panel._upload_directory.cleanup()

    def test_unknown_dialect_does_not_guess_and_manual_override_works(self):
        panel = CSVSourcePanel()
        panel._file_selected({"new": ({"name": "example.csv", "content": b"single\nvalue\n"},)})
        self.assertIn("Confirm delimiter", panel.status.value)
        panel._upload_directory.cleanup()

    def test_explicit_text_qualifier_validates_embedded_delimiter(self):
        panel = CSVSourcePanel()
        panel._file_selected({"new": ({"name": "example.csv", "content": b"id;txt\n1;'a;b'\n"},)})
        panel.delimiter.value = ";"
        panel.quotechar.value = "'"
        self.assertTrue(panel.validate())
        panel.quotechar.value = ";"
        self.assertFalse(panel.validate())
        panel._upload_directory.cleanup()

    def test_legacy_upload_mapping_supported(self):
        panel = CSVSourcePanel()
        panel._file_selected({"new": {"test.csv": {"metadata": {"name": "test.csv"}, "content": b"uid,target\n1,A\n"}}})
        self.assertTrue(panel.validate())
        panel._upload_directory.cleanup()


if __name__ == "__main__":
    unittest.main()
