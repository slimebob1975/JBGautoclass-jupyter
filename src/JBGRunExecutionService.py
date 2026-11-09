"""GUI-independent execution of a source-bound, prepared ML run (281b).

The composition root supplies trusted runtime adapters. The execution boundary
re-resolves and revalidates immediately before starting because Config is mutable.
No training logic, SQL connection, or CSV staging is implemented here.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from JBGApplicationContracts import DataSourceKind
from JBGRunPreparationService import PreparedRun, RunPreparationService


@runtime_checkable
class RunRuntime(Protocol):
    def execute(self, prepared: PreparedRun): ...


class RunExecutionService:
    def __init__(self, preparation: RunPreparationService, *,
                 sql: RunRuntime | None = None, csv: RunRuntime | None = None):
        if not isinstance(preparation, RunPreparationService):
            raise TypeError("preparation must be RunPreparationService")
        self._preparation = preparation
        self._runtimes = {}
        for source, runtime in ((DataSourceKind.SQL, sql), (DataSourceKind.CSV, csv)):
            if runtime is not None:
                if not isinstance(runtime, RunRuntime):
                    raise TypeError("runtime must implement execute(prepared)")
                self._runtimes[source] = runtime

    def execute(self, prepared: PreparedRun):
        if not isinstance(prepared, PreparedRun):
            raise TypeError("expected PreparedRun")
        source = prepared.dataset.source
        if (not isinstance(source, DataSourceKind)
                or prepared.configuration.source is not source):
            raise ValueError("prepared run source mismatch")
        runtime = self._runtimes.get(source)
        if runtime is None:
            raise ValueError("run runtime for source is not configured")
        # Recheck mutable Config and the source's authorized dataset resolver.
        checked = self._preparation.prepare(prepared.dataset, prepared.configuration)
        if (checked.configuration.payload is not prepared.configuration.payload
                or checked.dataset != prepared.dataset
                or (checked.resolved_dataset is not prepared.resolved_dataset
                    and checked.resolved_dataset != prepared.resolved_dataset)):
            raise ValueError("prepared dataset binding changed before execution")
        return runtime.execute(checked)


class SQLClassifierRuntime:
    """Use the already validated SQL DataLayer, never construct another one."""
    def __init__(self, logger):
        self._logger = logger

    def execute(self, prepared: PreparedRun):
        if prepared.dataset.source is not DataSourceKind.SQL:
            raise ValueError("SQL runtime requires SQL run")
        config = prepared.configuration.payload
        layer = prepared.resolved_dataset
        if (getattr(getattr(config, "data_source", None), "kind", None) != "sql"
                or getattr(layer, "config", None) is not config):
            raise ValueError("SQL runtime configuration/layer mismatch")
        from AutomaticClassifier import AutomaticClassifier
        return AutomaticClassifier(config=config, logger=self._logger,
                                   datalayer=layer, regression_suite=False).run()


class RegisteredCSVClassifierRuntime:
    """Delegate to existing registered train-only CSV preparation boundary.

    repository/scope/configuration_id are trusted inputs from an authorized
    composition root, NEVER derived from an opaque dataset key.
    """
    def __init__(self, repository, scope, configuration_id, logger):
        from Storage.ArtifactContract import StorageScope
        if not isinstance(scope, StorageScope):
            raise TypeError("trusted StorageScope required")
        if not isinstance(configuration_id, str) or not configuration_id.strip():
            raise ValueError("configuration_id required")
        self._repository = repository
        self._scope = scope
        self._configuration_id = configuration_id
        self._logger = logger

    def execute(self, prepared: PreparedRun):
        from Storage.ArtifactContract import ArtifactKind, ArtifactReference
        if prepared.dataset.source is not DataSourceKind.CSV:
            raise ValueError("CSV runtime requires CSV run")
        config = prepared.configuration.payload
        reference = prepared.resolved_dataset
        if (not isinstance(reference, ArtifactReference)
                or reference.scope != self._scope
                or reference.artifact_id != prepared.dataset.key
                or reference.kind != ArtifactKind.DATASET
                or reference.content_type != "text/csv"
                or not reference.dataset_id):
            raise ValueError("CSV runtime dataset/scope mismatch")
        source = getattr(config, "data_source", None)
        if (getattr(source, "kind", None) != "csv"
                or getattr(source, "artifact_id", None) != reference.artifact_id
                or getattr(source, "dataset_id", None) != reference.dataset_id):
            raise ValueError("CSV runtime configuration mismatch")
        predicts = getattr(config, "should_predict", None)
        trains = getattr(config, "should_train", None)
        if callable(predicts) and callable(trains) and predicts() and not trains():
            # 296b: predict with a saved model through the same CSV boundary
            # as the Voilà GUI (identity and roles checked against the
            # training record); the model path comes from the server-built spec.
            from Storage.CSVApplicationRuntime import prepare_registered_csv_prediction_classifier
            # The CSV boundary takes the trusted file name as text (as the
            # Voilà GUI passes it); get_model_filename() returns a Path.
            classifier = prepare_registered_csv_prediction_classifier(
                str(config.get_model_filename()), config, self._repository, self._scope,
                self._configuration_id, self._logger)
            # The CSV boundary runs with the configuration restored from the
            # model file; carry only the run's own output folder over (295a),
            # so the predictions are also kept there (296b).
            classifier.config.output_path = getattr(config, "output_path", None)
            return classifier.run()
        from Storage.CSVTrainingApplication import prepare_registered_csv_training_classifier
        classifier = prepare_registered_csv_training_classifier(
            config, self._repository, self._scope, self._configuration_id, self._logger)
        return classifier.run()
