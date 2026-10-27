"""Revision 242: block SQL entry points during native-handler CSV prediction.

This deliberately reuses the synthetic application context from revision 241.
It does not claim to cover application startup, user authentication, or CSV training.
"""
import json
import unittest
from unittest.mock import patch

from Storage.ArtifactContract import ArtifactKind
from test_csv_native_model_reload_241 import CSVNativeReload241, JBGHandler, ModelHandler
from test_csv_taskrunner_handler_bridge_232 import TASKS


@unittest.skipIf(JBGHandler is None or ModelHandler is None,
                 'Native JBG dependencies unavailable')
class CSVBlockedSQL242(CSVNativeReload241):
    test_native_runner_reloads_validated_jbg_model_and_publishes = None
    test_corrupt_artifact_prevents_prediction_publication = None

    def _run_without_sql(self):
        # Fail at the actual connection boundary and at the legacy SQL facade.
        # A mistaken SQL fallback cannot silently pass this integration test.
        import pyodbc
        from SQLDataLayer import DataLayer as SQLDataLayer
        from SqlHelper.JBGSqlHelper import JBGSqlHelper

        def forbidden(*args, **kwargs):
            raise AssertionError('CSV prediction attempted SQL access')

        with (patch.object(pyodbc, 'connect', side_effect=forbidden) as odbc,
              patch.object(SQLDataLayer, 'get_dataset', side_effect=forbidden) as sql_read,
              patch.object(SQLDataLayer, 'save_prediction_data', side_effect=forbidden) as sql_write,
              patch.object(JBGSqlHelper, 'connect', side_effect=forbidden) as helper):
            outcome = self.runner.run(TASKS)
            odbc.assert_not_called()
            sql_read.assert_not_called()
            sql_write.assert_not_called()
            helper.assert_not_called()
            return outcome

    def test_native_versioned_prediction_is_sql_free(self):
        self.assertFalse(self._run_without_sql())
        refs = self._published()
        self.assertEqual(len(refs), 1)
        with self.repo.open_artifact(self.scope, refs[0].artifact_id) as stream:
            rows = json.load(stream)['rows']
        self.assertEqual([r['key'] for r in rows], ['005', '006'])

    def test_sql_blocked_corrupt_artifact_cannot_publish(self):
        self.jbg_model_path.write_bytes(b'invalid JBG artifact bytes')
        with self.assertRaises(Exception) as failure:
            self._run_without_sql()
        self.assertNotIn("attempted SQL access", str(failure.exception))
        self.assertFalse(self._published())


if __name__ == '__main__':
    unittest.main()
