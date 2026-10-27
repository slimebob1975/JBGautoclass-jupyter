"""245: exercise the CSV composition entry point and native classifier.run together.

The registered CSV I/O builder and handler initialization are still controlled by
previous native fixtures; this is not an unassisted production bootstrap test.
"""
import json
import unittest
from unittest.mock import patch

from Storage.ArtifactContract import ArtifactKind
from Storage.CSVApplicationRuntime import prepare_csv_prediction_classifier
from test_csv_classifier_run_244 import CSVClassifierRun244, AutomaticClassifier
from test_csv_taskrunner_handler_bridge_232 import TASKS


@unittest.skipIf(AutomaticClassifier is None, 'Full JBG optional dependencies unavailable')
class CSVApplicationRun245(CSVClassifierRun244):
    test_application_run_reloads_model_and_publishes_without_sql = None
    test_application_run_rejects_damaged_model_before_publication = None

    def _prepared_classifier(self):
        self.config.should_display_mispredicted = lambda: False
        # Keep the real AutomaticClassifier constructor; only native handlers
        # prepared by the 241 fixture and external configuration I/O are injected.
        def registry(instance):
            instance.handler = self.handler
            return self.dataset, self.model, self.predictions

        with patch('Storage.CSVApplicationRuntime.prepare_csv_prediction_datalayer',
                   return_value=(self.config, self.layer)) as prepare, \
             patch.object(AutomaticClassifier, 'create_handlers', registry), \
             patch('Config.Config.load_config_from_model_file', return_value=self.config):
            classifier = prepare_csv_prediction_classifier(
                str(self.jbg_model_path), self.config, self.repo,
                self.scope, self.layer._reader.configuration.configuration_id,
                self.logger)
        prepare.assert_called_once()
        self.assertIs(classifier.datalayer, self.layer)
        self.assertIs(classifier.handler.datalayer, self.layer)
        return classifier

    def test_composed_application_runs_with_sql_closed(self):
        result = self._run_no_sql(self._prepared_classifier())
        self.assertEqual(result, {'mispredicted': None})
        refs = [ref for ref in self.repo.list_artifacts(self.scope)
                if ref.kind == ArtifactKind.PREDICTIONS]
        self.assertEqual(len(refs), 1)
        with self.repo.open_artifact(self.scope, refs[0].artifact_id) as stream:
            self.assertEqual([r['key'] for r in json.load(stream)['rows']],
                             ['005', '006'])

    def test_composed_application_rejects_corrupt_model_without_publish(self):
        classifier = self._prepared_classifier()
        self.jbg_model_path.write_bytes(b'bad model artifact')
        with self.assertRaises(Exception):
            self._run_no_sql(classifier)
        self.assertFalse([r for r in self.repo.list_artifacts(self.scope)
                          if r.kind == ArtifactKind.PREDICTIONS])

    def test_composed_application_rejects_sql_source_before_construct(self):
        self.config.data_source.kind = 'sql'
        with patch('Storage.CSVApplicationRuntime.prepare_csv_prediction_datalayer') as prepare:
            with self.assertRaisesRegex(ValueError, 'prediction-only CSV'):
                prepare_csv_prediction_classifier(
                    str(self.jbg_model_path), self.config, self.repo,
                    self.scope, self.layer._reader.configuration.configuration_id,
                    self.logger)
            prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
