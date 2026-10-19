"""Regression coverage for the revision 155 startup failure."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Config import Config


class JsonExport156bTests(unittest.TestCase):
    def test_extra_trees_is_exported_without_parsing_template(self):
        config = Config(save=False)
        config.mode.feature_search_tree_estimator = "Extra Trees"
        config.connection.sql_password = "secret"
        payload = config._config_to_json()
        self.assertEqual(payload["mode"]["feature_search_tree_estimator"], "Extra Trees")
        self.assertEqual(payload["connection"]["sql_password"], "")
        self.assertEqual(payload["schema_version"], 1)
        json.dumps(payload)

    def test_json_export_failure_does_not_break_legacy_config_save(self):
        config = Config(save=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.py"
            with patch.object(Config, "_config_to_json", side_effect=RuntimeError("simulated")):
                with self.assertLogs("Config", level="WARNING"):
                    config.save_to_file(path)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(".json").exists())

    def test_json_save_accepts_multiword_setting(self):
        config = Config(save=False)
        config.mode.feature_search_tree_estimator = "Extra Trees"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.py"
            config.save_to_file(path)
            payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"]["feature_search_tree_estimator"], "Extra Trees")
