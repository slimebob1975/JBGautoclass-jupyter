"""Revision 170: privacy-safe persistent logs and feature-search worker diagnostics."""
from pathlib import Path
import os
import unittest
from unittest.mock import patch

import pandas as pd

from JBGFeatureSearch import describe_feature_search_parallelism
from JBGLogPrivacy import (
    redacted_table_log_message,
    safe_prediction_query_log_message,
    safe_query_log_message,
    sensitive_log_data_enabled,
)


ROOT = Path(__file__).resolve().parents[1]
STREAMED_SOURCE = (ROOT / "src" / "JBGStreamedLogger.py").read_text(encoding="utf-8")
LOGGER_SOURCE = (ROOT / "src" / "JBGLogger.py").read_text(encoding="utf-8")
HANDLER_SOURCE = (ROOT / "src" / "JBGHandler.py").read_text(encoding="utf-8")
SERVICE_SOURCE = (ROOT / "src" / "JBGFeatureSearchService.py").read_text(encoding="utf-8")
ENV_TEMPLATE = (ROOT / ".env_template").read_text(encoding="utf-8")


class LogPrivacy170Tests(unittest.TestCase):
    def test_sensitive_logging_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(sensitive_log_data_enabled())
        for value in ("1", "true", "YES", "on"):
            self.assertTrue(sensitive_log_data_enabled(value))
        for value in ("", "0", "false", "no"):
            self.assertFalse(sensitive_log_data_enabled(value))

    def test_sql_is_omitted_by_default_but_can_be_enabled(self):
        query = "SELECT * FROM [private_db].[private_schema].[private_table]"
        with patch.dict(os.environ, {"JBG_LOG_SENSITIVE_DATA": "0"}):
            message = safe_query_log_message("Classification data", query)
            self.assertNotIn("private_db", message)
            self.assertIn("SQL omitted", message)
        with patch.dict(os.environ, {"JBG_LOG_SENSITIVE_DATA": "1"}):
            self.assertIn(query, safe_query_log_message("Classification data", query))

    def test_prediction_retrieval_sql_is_omitted_by_default(self):
        query = "SELECT * FROM [private_prediction_table]"
        with patch.dict(os.environ, {"JBG_LOG_SENSITIVE_DATA": "0"}):
            message = safe_prediction_query_log_message(7, query)
            self.assertIn("Added 7 rows", message)
            self.assertNotIn("private_prediction_table", message)

    def test_row_level_table_placeholder_contains_shape_not_values(self):
        frame = pd.DataFrame({"secret_name": ["PERSON_A", "PERSON_B"], "x": [1, 2]})
        message = redacted_table_log_message(frame)
        self.assertIn("shape=2x2", message)
        self.assertNotIn("PERSON_A", message)
        self.assertNotIn("secret_name", message)

    def test_streamed_logger_keeps_full_ui_but_redacts_persistent_paths(self):
        self.assertIn('print_html(f"Query for <em>{type}</em>: <pre>{query}</pre>")', STREAMED_SOURCE)
        self.assertIn("self.log_file.write_message(safe_log)", STREAMED_SOURCE)
        self.assertIn('"Head", dataset.head(20), log_matrix=sensitive_log_data_enabled()', STREAMED_SOURCE)
        self.assertIn("if not log_matrix:", STREAMED_SOURCE)
        self.assertIn("redacted_table_log_message(matrix)", STREAMED_SOURCE)

    def test_terminal_logger_uses_safe_sql_and_row_preview(self):
        self.assertIn("safe_query_log_message(type, query)", LOGGER_SOURCE)
        self.assertIn("safe_prediction_query_log_message", LOGGER_SOURCE)
        self.assertIn("if self.in_terminal and not log_matrix:", LOGGER_SOURCE)

    def test_mispredicted_sql_uses_query_privacy_path(self):
        self.assertIn(
            'self.handler.logger.print_query("most mispredicted data", most_mispredicted_query)',
            HANDLER_SOURCE,
        )
        self.assertNotIn(
            'print_code("Get the most misplaced data by SQL query", most_mispredicted_query)',
            HANDLER_SOURCE,
        )

    def test_sensitive_logging_switch_is_documented_in_env_template(self):
        self.assertIn('JBG_LOG_SENSITIVE_DATA="0"', ENV_TEMPLATE)


class FeatureSearchDiagnostics170Tests(unittest.TestCase):
    def test_parallelism_message_explains_budget_and_global_cap(self):
        message = describe_feature_search_parallelism(
            5, 4, logical_cpus=24, configured_cap=4
        )
        self.assertIn("workers=4", message)
        self.assertIn("logical_cpus=24", message)
        self.assertIn("global_cap=4", message)
        self.assertIn("candidate_budget=5", message)
        self.assertIn("pre_memory_cap=4", message)
        self.assertIn("memory_guard=clear", message)

    def test_parallelism_message_identifies_memory_reduction(self):
        message = describe_feature_search_parallelism(
            20, 1, logical_cpus=16, configured_cap=8
        )
        self.assertIn("pre_memory_cap=8", message)
        self.assertIn("memory_guard=triggered", message)

    def test_parallelism_message_handles_unlimited_global_cap(self):
        message = describe_feature_search_parallelism(
            3, 3, logical_cpus=12, configured_cap=-1
        )
        self.assertIn("global_cap=unlimited", message)
        self.assertIn("pre_memory_cap=3", message)

    def test_service_writes_parallelism_diagnostic_to_persistent_log(self):
        self.assertIn("describe_feature_search_parallelism", SERVICE_SOURCE)
        self.assertIn("logger.log_file.write_message(parallelism_message)", SERVICE_SOURCE)


if __name__ == "__main__":
    unittest.main()
