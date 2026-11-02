"""281c: registered CSV -> application preparation -> real native ML roundtrip.

No notebook, GUI, live SQL server, or real customer data is needed.
The registered CSV, model and output files live under a temporary directory.
"""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from test_csv_full_classifier_run_259 import CSVFullClassifierRun259


class CSVApplicationEndToEnd281c(CSVFullClassifierRun259):
    # Reuse the real native 259 fixture and its dependency-aware skip marker,
    # but execute it through the application service, not the direct entrypoint.
    test_full_native_csv_train_run_and_checkpoint = None

    def _configure_training(self):
        from JBGMeta import (Algorithm, AlgorithmTuple, Oversampling,
                             Undersampling, Preprocess, PreprocessTuple,
                             Reduction, ReductionTuple)
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
        from Helpers import build_model_performance_matrix, save_matrix_as_csv
        self.logger.print_test_performance.side_effect = lambda results, filepath: (
            save_matrix_as_csv(build_model_performance_matrix(results), filepath))
        self.logger.get_log_output.return_value = ''
        self.logger.get_log_output_html.return_value = ''
        return Path(config.get_model_filename())

    def _application(self, scope=None):
        from JBGApplicationContracts import DataSourceKind, DatasetReference
        from JBGDatasetApplicationService import DatasetApplicationService
        from JBGDatasetResolvers import ScopedCSVDatasetResolver
        from JBGRunPreparationService import RunConfiguration, RunPreparationService
        from JBGRunConfigurationValidators import CSVRunConfigurationValidator
        from JBGRunExecutionService import RunExecutionService, RegisteredCSVClassifierRuntime

        authorized_scope = scope if scope is not None else self.scope
        datasets = DatasetApplicationService(csv=ScopedCSVDatasetResolver(
            self.repo, authorized_scope))
        preparation = RunPreparationService(
            datasets, csv=CSVRunConfigurationValidator())
        execution = RunExecutionService(preparation, csv=RegisteredCSVClassifierRuntime(
            self.repo, authorized_scope, self.roles.configuration_id, self.logger))
        reference = DatasetReference(DataSourceKind.CSV, self.ref.artifact_id)
        configuration = RunConfiguration(DataSourceKind.CSV, self.config)
        return preparation, execution, reference, configuration

    def test_registered_csv_trains_and_reloads_model_through_application(self):
        from JBGModelPersistence import load_model_artifact_with_compatibility
        model_path = self._configure_training()
        self.assertFalse(model_path.exists())
        preparation, execution, reference, configuration = self._application()
        captured_classifiers = []
        from Storage.CSVTrainingApplication import prepare_registered_csv_training_classifier
        def capture_real_classifier(*args, **kwargs):
            classifier = prepare_registered_csv_training_classifier(*args, **kwargs)
            captured_classifiers.append(classifier)
            return classifier
        with ExitStack() as guards:
            guards.enter_context(patch('socket.create_connection',
                                       side_effect=AssertionError('Network access forbidden')))
            import pyodbc
            guards.enter_context(patch.object(pyodbc, 'connect',
                                              side_effect=AssertionError('SQL connection forbidden')))
            from SQLDataLayer import DataLayer as SQLDataLayer
            guards.enter_context(patch.object(SQLDataLayer, '__init__',
                                              side_effect=AssertionError('SQL layer forbidden')))
            mock_send_email = guards.enter_context(
                patch('JBGTaskRunner.send_email', return_value=False))
            guards.enter_context(patch(
                'Storage.CSVTrainingApplication.prepare_registered_csv_training_classifier',
                side_effect=capture_real_classifier))
            prepared = preparation.prepare(reference, configuration)
            self.assertEqual(prepared.resolved_dataset.scope, self.scope)
            self.assertEqual(prepared.resolved_dataset.dataset_id, self.ref.dataset_id)
            self.assertEqual(prepared.resolved_dataset.artifact_id, self.ref.artifact_id)
            result = execution.execute(prepared)
            self.assertIsInstance(result, dict)
            mock_send_email.assert_called_once()

        self.assertTrue(model_path.is_file(), 'Application run did not save a model')
        payload, compatibility = load_model_artifact_with_compatibility(model_path)
        self.assertEqual(compatibility['source_kind'], 'csv')
        self.assertEqual(compatibility['schema_version'], 2)
        self.assertIn('source_identity_sha256', compatibility)
        self.assertNotIn('dataset_id', compatibility)
        restored_model = payload[3]
        self.assertIsNotNone(restored_model)
        # Assert on the fitted native handler's feature matrix, as in 259.
        self.assertEqual(len(captured_classifiers), 1)
        classifier = captured_classifiers[0]
        self.assertIs(classifier.datalayer, classifier.handler.datalayer)
        self.assertEqual(len(classifier.dh.X), 135)
        self.assertEqual(len(restored_model.predict(classifier.dh.X)), 135)
        self.assertIsNot(restored_model, classifier.mh.model.pipeline)

    def test_other_scope_cannot_prepare_csv_run(self):
        from Storage.ArtifactContract import StorageScope
        self._configure_training()
        foreign_scope = StorageScope(str(uuid4()), str(uuid4()))
        preparation, _, reference, configuration = self._application(scope=foreign_scope)
        with self.assertRaises((FileNotFoundError, ValueError)):
            preparation.prepare(reference, configuration)
        self.assertFalse(Path(self.config.get_model_filename()).exists())
