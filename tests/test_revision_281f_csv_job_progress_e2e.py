"""281f: real registered CSV training emits job-bound engine progress.

Uses revision 281c's synthetic native-ML fixture. No GUI or SQL connection.
"""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import test_revision_281c_csv_application_e2e as revision_281c


class CSVJobProgressEndToEnd281f(revision_281c.CSVApplicationEndToEnd281c):
    # Inherited acceptance probes are independently exercised by revision 281c.
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def test_real_csv_run_reports_original_progress_per_job(self):
        from JBGApplicationContracts import RunStatus
        from JBGDatasetApplicationService import DatasetApplicationService
        from JBGDatasetResolvers import ScopedCSVDatasetResolver
        from JBGModelPersistence import load_model_artifact_with_compatibility
        from JBGRunConfigurationValidators import CSVRunConfigurationValidator
        from JBGRunExecutionService import RunExecutionService, RegisteredCSVClassifierRuntime
        from JBGRunJobService import RunJobService
        from JBGRunPreparationService import RunPreparationService
        from JBGRunProgress import ProgressObservingLogger, RunProgressService

        model_path = self._configure_training()
        progress = RunProgressService()
        observing_logger = ProgressObservingLogger(self.logger, progress)
        dataset_service = DatasetApplicationService(
            csv=ScopedCSVDatasetResolver(self.repo, self.scope))
        preparation = RunPreparationService(
            dataset_service, csv=CSVRunConfigurationValidator())
        execution = RunExecutionService(
            preparation,
            csv=RegisteredCSVClassifierRuntime(
                self.repo, self.scope, self.roles.configuration_id, observing_logger))
        jobs = RunJobService(execution, progress=progress)
        _, _, reference, configuration = self._application()
        prepared = preparation.prepare(reference, configuration)
        first = jobs.create(prepared)
        second = jobs.create(prepared)  # Not executed: must not inherit first job's events.
        self.assertIsNone(jobs.latest_progress(first))
        self.assertEqual(jobs.progress_history(second), ())

        from Storage.CSVTrainingApplication import prepare_registered_csv_training_classifier
        trained_classifiers = []

        def capture_classifier(*args, **kwargs):
            classifier = prepare_registered_csv_training_classifier(*args, **kwargs)
            trained_classifiers.append(classifier)
            return classifier

        with ExitStack() as guards:
            guards.enter_context(patch('socket.create_connection',
                                       side_effect=AssertionError('Network forbidden')))
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
                side_effect=capture_classifier))
            result = jobs.execute(first)
            mock_send_email.assert_called_once()

        self.assertIsInstance(result, dict)
        self.assertIs(jobs.result(first), result)
        self.assertEqual(jobs.snapshot(first).status, RunStatus.SUCCEEDED)
        self.assertEqual(jobs.snapshot(second).status, RunStatus.PENDING)
        self.assertIsNone(jobs.error(first))
        events = jobs.progress_history(first)
        self.assertTrue(events, 'Native ML engine emitted no observable progress')
        self.assertEqual(events[0].message, 'Starting up ...')
        self.assertEqual(events[0].percent, 0.0)
        self.assertTrue(any(event.message.startswith('Ending program after ') and event.percent == 1.0
                            for event in events if event.message is not None))
        self.assertEqual(jobs.latest_progress(first), events[-1])
        self.assertTrue(all(event.run == first and event.timestamp.tzinfo is not None
                            for event in events))
        self.assertEqual(jobs.progress_history(second), ())

        # The wrapper must preserve the ML logger's messages and percentages.
        original_calls = self.logger.print_progress.call_args_list
        self.assertEqual(len(events), len(original_calls))
        for event, call in zip(events, original_calls):
            arguments, keywords = call
            message = keywords.get('message', arguments[0] if arguments else None)
            percent = keywords.get('percent', arguments[1] if len(arguments) > 1 else None)
            self.assertEqual((event.message, event.percent), (message, percent))

        self.assertTrue(Path(model_path).is_file())
        payload, compatibility = load_model_artifact_with_compatibility(model_path)
        self.assertEqual(compatibility['source_kind'], 'csv')
        self.assertEqual(compatibility['schema_version'], 2)
        self.assertIn('source_identity_sha256', compatibility)
        self.assertIsNotNone(payload[3])
        self.assertEqual(len(trained_classifiers), 1)
        self.assertEqual(len(payload[3].predict(trained_classifiers[0].dh.X)), 135)
