"""Revision 241: native handler constructors plus real JBG artifact reload.

Synthetic, trusted artifacts only. Dataset and prediction method state are inherited
from the 240 probe; this is not a full application bootstrap or Azure smoke test.
"""
import unittest
from unittest.mock import patch

from JBGModelCompatibility import build_csv_model_compatibility_metadata
from JBGModelPersistence import save_model_artifact
from Storage.ArtifactContract import ArtifactKind
from test_csv_model_compatibility_234 import SyntheticCompatibilityConfig, fitted_converter
from test_csv_native_constructors_240 import CSVNativeConstructor240, JBGHandler

try:
    from JBGHandler import ModelHandler
except (ImportError, OSError):
    ModelHandler = None


@unittest.skipIf(JBGHandler is None or ModelHandler is None,
                 'Native JBG dependencies unavailable')
class CSVNativeReload241(CSVNativeConstructor240):
    test_native_handlers_and_runner_publish_csv = None
    test_native_runner_rejects_wrong_datalayer_before_read = None

    def setUp(self):
        super().setUp()
        from sklearn.pipeline import Pipeline
        self.saved_config = SyntheticCompatibilityConfig(self.ref)
        converter = fitted_converter()
        source = self.layer._reader.configuration
        metadata = build_csv_model_compatibility_metadata(
            self.saved_config, converter, source)
        self.jbg_model_path = self.model_path.with_name('trusted-native-241.pkl')
        save_model_artifact(
            self.jbg_model_path, self.saved_config, converter,
            (None, None, None, None, None),
            Pipeline([('classifier', self.estimator)]), None, 1,
            compatibility=metadata)
        self.config.get_model_filename = lambda: str(self.jbg_model_path)
        self.config.set_num_selected_features = lambda value: None
        # Restore the actual production loader, which revision 240 replaces with
        # its controlled loader. This method does full source/semantic review.
        self.model.load_model = ModelHandler.load_model.__get__(self.model)
        self.model.load_model_from_file = ModelHandler.load_model_from_file.__get__(self.model)

    def _published(self):
        return [ref for ref in self.repo.list_artifacts(self.scope)
                if ref.kind == ArtifactKind.PREDICTIONS]

    def test_native_runner_reloads_validated_jbg_model_and_publishes(self):
        from test_csv_taskrunner_handler_bridge_232 import TASKS
        self.assertFalse(self.runner.run(TASKS))
        self.assertIsNotNone(self.model.model)
        self.assertEqual(list(self.dataset.unpredicted_keys), ['005', '006'])
        self.assertEqual(len(self._published()), 1)

    def test_corrupt_artifact_prevents_prediction_publication(self):
        from test_csv_taskrunner_handler_bridge_232 import TASKS
        self.jbg_model_path.write_bytes(b'corrupt synthetic model')
        with self.assertRaises(Exception):
            self.runner.run(TASKS)
        self.assertFalse(self._published())


if __name__ == '__main__':
    unittest.main()
