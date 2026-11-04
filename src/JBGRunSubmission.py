"""Turn a reference to a saved configuration into a queued, authorized run (285d).

Server side only: the client names a configuration from the catalog
(JBGSavedConfigurations); every path, scope and role record is derived here.

* SQL: load the Voila configuration file, open the run's DataLayer with the
  user's own SQL session (284c), prepare, authorize and queue.
* CSV: build a new training configuration from a saved record exactly as Voila
  does when training starts (``prepare_csv_gui_training_config``: new record,
  Voila's model location), then prepare, authorize and queue.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from JBGApplicationContracts import DataSourceKind, DatasetReference
from JBGDurableRuns import DurableRunService
from JBGRunComposition import compose_csv_run, compose_sql_run
from JBGRunJobStore import RunJobStore
from JBGRunPreparationService import RunConfiguration
from JBGSavedConfigurations import ROOT, SavedConfigurations
from JBGSQLSessions import SQLSessionRegistry
from Storage.ArtifactContract import StorageScope

DEFAULT_CSV_MODEL_ROOT = ROOT / "output" / "csv_artifacts" / "models"
# SQL runs have no storage scope; DurableRunService still requires one.
_SQL_SCOPE = StorageScope("00000000-0000-4000-8000-000000000000",
                          "00000000-0000-4000-8000-000000000000")


class _QuietLogger:
    """Preparation in the web process logs nothing; training logs in the worker."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class RunSubmission:
    def __init__(self, store: RunJobStore, *, configurations: SavedConfigurations,
                 sessions: SQLSessionRegistry, on_submit: Callable[[], None] | None = None,
                 csv_model_root: Path = DEFAULT_CSV_MODEL_ROOT) -> None:
        if not isinstance(configurations, SavedConfigurations):
            raise TypeError("SavedConfigurations required")
        if not isinstance(sessions, SQLSessionRegistry):
            raise TypeError("SQLSessionRegistry required")
        self._store = store
        self._configurations = configurations
        self._sessions = sessions
        self._on_submit = on_submit
        self._csv_model_root = Path(csv_model_root)

    @property
    def configurations(self) -> SavedConfigurations:
        return self._configurations

    def _service(self, scope: StorageScope, **options) -> DurableRunService:
        return DurableRunService(self._store, scope_for=lambda principal: scope,
                                 on_submit=self._on_submit, **options)

    def submit_sql(self, principal, configuration_name: str) -> dict:
        """PermissionError without SQL sign-in; UnknownConfiguration for unknown names."""
        config = self._configurations.load_sql(configuration_name)
        composed = compose_sql_run(self._sessions, principal, config, _QuietLogger())
        prepared = composed.preparation.prepare(
            DatasetReference(DataSourceKind.SQL, composed.sql_dataset_key()),
            RunConfiguration(DataSourceKind.SQL, composed.configuration))
        service = self._service(_SQL_SCOPE, authorize_dataset=self._sessions.authorize)
        return service.submit(principal, prepared)

    def submit_csv(self, principal, project: str, configuration_id: str) -> dict:
        """Train again with the settings and roles of a saved CSV configuration."""
        from Storage.CSVGUITrainingConfig import prepare_csv_gui_training_config
        from JBGRunWorker import _config_from_spec, SPEC_VERSION
        repository, scope, saved = self._configurations.load_csv(project, configuration_id)
        base = _config_from_spec({"spec_version": SPEC_VERSION,
                                  "configuration": saved.shared_json}, None)
        runtime, record = prepare_csv_gui_training_config(
            base, repository, scope, saved.artifact_id,
            id_column=saved.id_column, class_column=saved.class_column,
            data_columns=tuple(saved.data_columns), text_columns=tuple(saved.text_columns),
            csv_format=saved.csv_format, model_root=self._csv_model_root,
            predict=bool(base.mode.predict), mispredicted=bool(base.mode.mispredicted))
        composed = compose_csv_run(repository, scope, record.configuration_id, _QuietLogger())
        prepared = composed.preparation.prepare(
            DatasetReference(DataSourceKind.CSV, saved.artifact_id),
            RunConfiguration(DataSourceKind.CSV, runtime))
        return self._service(scope).submit(principal, prepared,
                                           configuration_id=record.configuration_id)
