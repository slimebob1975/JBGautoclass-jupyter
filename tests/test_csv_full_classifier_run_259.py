"""259: full native CSV train-only classifier path in the Windows ML environment.

All rows and artifact identifiers are synthetic. This is an acceptance probe:
production selection, evaluation, checkpoint reload and retraining are NOT mocked.
CSV GUI execution remains disabled regardless of test outcome.
"""
from contextlib import ExitStack
from pathlib import Path
from unittest import skipIf
from unittest.mock import patch

from test_csv_native_handler_bridge_258 import (
    AutomaticClassifier, CSVNativeHandlerBridge258,
)


@skipIf(AutomaticClassifier is None, 'Full JBG ML dependencies unavailable')
class CSVFullClassifierRun259(CSVNativeHandlerBridge258):
    test_native_dataset_handler_and_real_estimator_roundtrip = None
    test_registered_train_only_reader_does_not_include_unlabelled_rows = None

    def test_full_native_csv_train_run_and_checkpoint(self):
        from JBGMeta import (
            Algorithm, AlgorithmTuple, Oversampling, Undersampling,
            Preprocess, PreprocessTuple, Reduction, ReductionTuple,
        )
        from JBGModelPersistence import load_model_artifact_with_compatibility
        from Storage.CSVTrainingApplication import prepare_registered_csv_training_classifier

        config = self.config
        config.script_path = self.root
        config.io.model_path = 'models'
        config.mode.algorithm = AlgorithmTuple([Algorithm.LRN])
        config.mode.preprocessor = PreprocessTuple([Preprocess.NOS])
        config.mode.feature_selection = ReductionTuple([Reduction.NOR])
        config.mode.oversampler = Oversampling.NOG
        config.mode.undersampler = Undersampling.NUG
        config.mode.oversamplers = ['NOG']
        config.mode.undersamplers = ['NUG']
        config.mode.max_iterations = 300
        (self.root / 'models').mkdir()
        (self.root / 'output' / 'csvs').mkdir(parents=True)
        model_path = Path(config.get_model_filename())
        self.assertFalse(model_path.exists())

        with ExitStack() as guards:
            guards.enter_context(patch('socket.create_connection', side_effect=AssertionError('Network access forbidden')))
            import pyodbc
            guards.enter_context(patch.object(pyodbc, 'connect', side_effect=AssertionError('SQL connection forbidden')))
            from SQLDataLayer import DataLayer as SQLDataLayer
            guards.enter_context(patch.object(SQLDataLayer, '__init__', side_effect=AssertionError('SQL layer forbidden')))
            classifier = prepare_registered_csv_training_classifier(
                config, self.repo, self.scope, self.roles.configuration_id, self.logger)
            self.assertIs(classifier.datalayer, classifier.handler.datalayer)
            result = classifier.run()
            self.assertIsInstance(result, dict, 'Classifier did not finish the native train-only task sequence')
            self.assertTrue(model_path.is_file(), 'Native CSV model checkpoint was not saved')
            payload, compatibility = load_model_artifact_with_compatibility(model_path)
            self.assertEqual(compatibility['source_kind'], 'csv')
            self.assertEqual(compatibility['schema_version'], 2)
            self.assertIn('source_identity_sha256', compatibility)
            self.assertNotIn('dataset_id', compatibility)
            model = payload[3]
            self.assertIsNotNone(model)
            self.assertEqual(len(classifier.dh.X), 135)
            self.assertEqual(len(model.predict(classifier.dh.X)), 135)
            self.assertIsNot(model, classifier.mh.model.pipeline)


if __name__ == '__main__':
    import unittest
    unittest.main()
