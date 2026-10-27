"""Revision 244: native AutomaticClassifier.run() with SQL boundaries closed.

Uses the native 241 handler fixture; full unassisted application bootstrap and
production CSV configuration restoration remain separate acceptance criteria.
"""
import json
import unittest
from unittest.mock import patch

from Storage.ArtifactContract import ArtifactKind
from test_csv_native_model_reload_241 import CSVNativeReload241, JBGHandler, ModelHandler
from test_csv_taskrunner_handler_bridge_232 import TASKS

try:
    from AutomaticClassifier import AutomaticClassifier
    from SQLDataLayer import DataLayer as SQLDataLayer
    from SqlHelper.JBGSqlHelper import JBGSqlHelper
    import pyodbc
except (ImportError, OSError):
    AutomaticClassifier = None


@unittest.skipIf(AutomaticClassifier is None or JBGHandler is None or ModelHandler is None,
                 'Full JBG optional dependencies unavailable')
class CSVClassifierRun244(CSVNativeReload241):
    test_native_runner_reloads_validated_jbg_model_and_publishes = None
    test_corrupt_artifact_prevents_prediction_publication = None

    def _classifier(self):
        self.config.should_display_mispredicted = lambda: False
        # We reuse native handlers prepared by the 241 fixture. Verify the
        # actual application constructor and run() without claiming that this
        # fixture represents unassisted production application startup.
        def registry(instance):
            instance.handler = self.handler
            return self.dataset, self.model, self.predictions

        with patch.object(AutomaticClassifier, 'create_handlers', registry), \
             patch('Config.Config.load_config_from_model_file', return_value=self.config):
            classifier = AutomaticClassifier(config=self.config, logger=self.logger,
                                             datalayer=self.layer)
        return classifier

    def _run_no_sql(self, classifier):
        def forbidden(*args, **kwargs):
            raise AssertionError('CSV classifier attempted SQL access')

        with patch.object(pyodbc, 'connect', side_effect=forbidden) as odbc, \
             patch.object(SQLDataLayer, 'get_dataset', side_effect=forbidden) as sql_read, \
             patch.object(SQLDataLayer, 'save_prediction_data', side_effect=forbidden) as sql_write, \
             patch.object(JBGSqlHelper, 'connect', side_effect=forbidden) as helper, \
             patch('JBGTaskRunner.get_tasks', return_value=TASKS):
            result = classifier.run()
            odbc.assert_not_called()
            sql_read.assert_not_called()
            sql_write.assert_not_called()
            helper.assert_not_called()
            return result

    def test_application_run_reloads_model_and_publishes_without_sql(self):
        result = self._run_no_sql(self._classifier())
        self.assertEqual(result, {'mispredicted': None})
        refs = [item for item in self.repo.list_artifacts(self.scope)
                if item.kind == ArtifactKind.PREDICTIONS]
        self.assertEqual(len(refs), 1)
        with self.repo.open_artifact(self.scope, refs[0].artifact_id) as stream:
            rows = json.load(stream)['rows']
        self.assertEqual([row['key'] for row in rows], ['005', '006'])

    def test_application_run_rejects_damaged_model_before_publication(self):
        classifier = self._classifier()
        self.jbg_model_path.write_bytes(b'invalid artifact')
        with self.assertRaises(Exception) as failure:
            self._run_no_sql(classifier)
        self.assertNotIn('attempted SQL access', str(failure.exception))
        self.assertFalse([item for item in self.repo.list_artifacts(self.scope)
                          if item.kind == ArtifactKind.PREDICTIONS])


if __name__ == '__main__':
    unittest.main()
