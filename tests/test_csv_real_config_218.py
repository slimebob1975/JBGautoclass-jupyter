"""Revision 218: exercise the real JBG Config restore and CSV I/O boundary.

Requires the normal application dependencies; missing optional bootstrap packages
are reported as skipped tests, not successful integration verification.
"""
import ast
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from JBGModelPersistence import save_model_artifact
from Storage.CSVPredictionDataLayer import prepare_csv_prediction_datalayer
import test_csv_prediction_integration_216 as fixture216


class ConfigSourceBoundary218(unittest.TestCase):
    def test_real_loader_is_used_by_csv_runtime(self):
        path = Path(__file__).resolve().parents[1] / "src" / "Storage" / "CSVModelRuntime.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(c.func, ast.Attribute) and
                            c.func.attr == "load_config_from_model_file"
                            for c in calls))


class CSVConfigRestore218(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from Config import Config
        except ImportError as exc:
            raise unittest.SkipTest('Full JBG Config dependencies unavailable: ' + type(exc).__name__)
        cls.Config = Config

    def setUp(self):
        fixture216.CSVPredictionIntegration216.setUp(self)
        config = self.Config()
        config.data_source.kind = 'csv'
        config.data_source.dataset_id = self.ref.dataset_id
        config.data_source.artifact_id = self.ref.artifact_id
        config.data_source.delimiter = ';'
        config.connection.id_column = 'uid'
        config.connection.class_column = 'label'
        config.connection.data_numerical_columns = ['value']
        config.connection.data_text_columns = []
        config.mode.train = False
        config.mode.predict = True
        config.mode.mispredicted = False
        config.mode.calculate_dark_numbers = False
        self.saved = config
        self.runtime = copy.deepcopy(config)
        self.model_path = Path(self.model_path).with_suffix('.sav')
        save_model_artifact(self.model_path, config, None, (None, None, None, None, None),
                            None, None, 1, compatibility={'source': 'synthetic-test'})

    def _prepare(self, config=None):
        return prepare_csv_prediction_datalayer(
            str(self.model_path), config or self.runtime,
            self.repo, self.scope, self.roles.configuration_id)

    def test_real_config_restores_and_reads_scoped_csv_rows(self):
        restored, layer = self._prepare()
        self.assertIsInstance(restored, self.Config)
        self.assertEqual(restored.data_source.dataset_id, self.ref.dataset_id)
        self.assertEqual(restored.get_column_names(), ['value', 'uid', 'label'])
        self.assertEqual([row[1] for row in layer.get_dataset()], ['005', '006'])
        self.assertEqual(restored.get_data_limit(), 0)

    def test_real_config_rejects_different_dataset(self):
        config = copy.deepcopy(self.runtime)
        from uuid import uuid4
        config.data_source.dataset_id = str(uuid4())
        with self.assertRaisesRegex(Exception, "different dataset"):
            self._prepare(config)
        self.assertFalse(self.repo.list_artifacts(self.scope, run_id=str(uuid4())))

    def test_real_config_rejects_cross_source_reuse(self):
        config = copy.deepcopy(self.runtime)
        config.data_source.kind = 'sql'
        config.data_source.dataset_id = ''
        config.data_source.artifact_id = ''
        with self.assertRaisesRegex(Exception, "data source"):
            self._prepare(config)

    def test_runtime_mail_settings_survive_real_model_restore(self):
        self.runtime.mail.notification_email = 'recipient@example.invalid'
        restored, _ = self._prepare()
        self.assertEqual(restored.mail.notification_email, 'recipient@example.invalid')
        self.assertEqual(restored.data_source.artifact_id, self.ref.artifact_id)


if __name__ == '__main__':
    unittest.main()
