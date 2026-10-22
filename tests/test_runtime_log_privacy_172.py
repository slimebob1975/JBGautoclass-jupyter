"""Revision 172: broader application-runtime persistent-log privacy."""
from pathlib import Path
import os
import unittest
from unittest.mock import patch

from JBGLogPrivacy import (
    redact_local_paths,
    safe_code_log_message,
    safe_local_path,
    safe_prediction_values_log_message,
    safe_reclassification_log_message,
)


ROOT = Path(__file__).resolve().parents[1]
STREAMED = (ROOT / "src" / "JBGStreamedLogger.py").read_text(encoding="utf-8")
TERMINAL = (ROOT / "src" / "JBGLogger.py").read_text(encoding="utf-8")
HANDLER = (ROOT / "src" / "JBGHandler.py").read_text(encoding="utf-8")
TASKS = (ROOT / "src" / "JBGTaskRunner.py").read_text(encoding="utf-8")


class RuntimeLogPrivacy172Tests(unittest.TestCase):
    def test_local_paths_are_redacted_by_default_and_restored_by_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(safe_local_path(r"C:\\Users\\example\\project"), "<local-path>")
        with patch.dict(os.environ, {"JBG_LOG_SENSITIVE_DATA": "1"}, clear=True):
            self.assertEqual(
                safe_local_path(r"C:\\Users\\example\\project"),
                r"C:\\Users\\example\\project",
            )

    def test_absolute_paths_and_traceback_paths_are_redacted_without_eating_urls(self):
        windows = 'File "C:\\Users\\example\\project\\module.py", line 12, in run'
        posix = "artifact: /home/example/private/model.sav"
        url = "documentation: https://example.invalid/path"
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("example\\project", redact_local_paths(windows))
            self.assertIn('File "<local-path>"', redact_local_paths(windows))
            self.assertEqual(redact_local_paths(posix), "artifact: <local-path>")
            self.assertEqual(redact_local_paths(url), url)

    def test_prediction_payload_is_summarized_without_ids_or_labels(self):
        predictions = {"row-123": "private-label", "row-456": "other-label"}
        with patch.dict(os.environ, {}, clear=True):
            message = safe_prediction_values_log_message(predictions)
        self.assertIn("entries=2", message)
        self.assertNotIn("row-123", message)
        self.assertNotIn("private-label", message)

    def test_reclassification_summary_omits_row_and_class(self):
        with patch.dict(os.environ, {}, clear=True):
            message = safe_reclassification_log_message(123456, "private-label")
        self.assertNotIn("123456", message)
        self.assertNotIn("private-label", message)
        self.assertIn("omitted", message)

    def test_embedded_download_payload_and_paths_are_not_persisted(self):
        embedded = '<a download="x.csv" href="data:text/csv;base64,U0VDUkVU">download</a>'
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("U0VDUkVU", safe_code_log_message("Export", embedded))
            self.assertEqual(
                safe_code_log_message("Loading pipeline from", r"C:\\private\\model.sav"),
                "Loading pipeline from: <local-path>",
            )

    def test_streamed_logger_redacts_config_schema_predictions_and_class_matrices(self):
        self.assertIn("safe_code_log_message", STREAMED)
        self.assertIn("safe_prediction_values_log_message", STREAMED)
        self.assertIn("safe_reclassification_log_message", STREAMED)
        self.assertIn("subclass_title, df, print_always=False,", STREAMED)
        self.assertGreaterEqual(STREAMED.count("log_matrix=sensitive_log_data_enabled()"), 8)

    def test_handler_uses_privacy_safe_split_and_schema_diagnostics(self):
        self.assertIn("Dataset order: known labels first, then configured unique ID", HANDLER)
        self.assertIn("partition signature omitted from persistent log", HANDLER)
        self.assertIn("column names omitted from persistent log", HANDLER)
        self.assertIn("target/formula/value details omitted from persistent log", HANDLER)
        self.assertIn("log_matrix=sensitive_log_data_enabled()", HANDLER)

    def test_feature_importance_names_are_ui_only_by_default(self):
        self.assertIn("Feature importance — top 20 original inputs", TASKS)
        self.assertIn("log_matrix=sensitive_log_data_enabled()", TASKS)

    def test_terminal_logger_has_same_privacy_helpers(self):
        self.assertIn("safe_code_log_message", TERMINAL)
        self.assertIn("safe_prediction_values_log_message", TERMINAL)
        self.assertIn("safe_reclassification_log_message", TERMINAL)


if __name__ == "__main__":
    unittest.main()
