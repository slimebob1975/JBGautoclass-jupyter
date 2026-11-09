"""Start runs from a project's configuration and dataset (revision 288a).

Agreed 2026-11-04 (BACKLOG Phase 0 item 3):
1. A run request names a configuration (optionally a version; default the
   latest) and optionally a dataset (default the project's active training
   dataset). The run records the configuration version, dataset and model it
   uses (``spec["project_run"]``).
3. One model file per training run: ``<model root>/<project ID>/<model ID>.sav``,
   separate from Voilà's model folders.
4. Training runs only (train-and-predict included); prediction-only runs need
   a chosen saved model and follow with the Models tab.

The CSV path (288a) builds the run exactly as Voilà does at training start
(``prepare_csv_gui_training_config``) and checks the column roles against the
file. The SQL path (288b) takes host and driver from the server's settings
(decision 2: one sign-in per server) and catalog and table from the dataset;
it needs the user's SQL sign-in for that server.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from uuid import uuid4

from JBGApplicationContracts import DataSourceKind, DatasetReference
from JBGDatasets import DatasetRepository, DatasetRole
from JBGDurableRuns import DurableRunService
from JBGProjectService import ProjectService
from JBGProjectStorage import (DEFAULT_CSV_ROOT, check_csv_roles, csv_upload_format,
                               project_csv_storage)
from JBGRunJobStore import RunJobStore
from JBGRunPreparationService import RunConfiguration

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ROOT = ROOT / "output" / "models"


class RunRequestRefused(ValueError):
    """The project cannot start this run (fixed message, 409)."""


class RunDoesNotMatchDataset(ValueError):
    """The configuration's column roles do not fit the dataset (422)."""


class _QuietLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _load_config(settings: dict):
    from Config import Config
    with tempfile.TemporaryDirectory(prefix="jbg_run_") as folder:
        path = Path(folder) / "configuration.json"
        path.write_text(json.dumps(settings), encoding="utf-8")
        return Config.load_config_from_json(path)


class ProjectRunService:
    def __init__(self, store: RunJobStore, *, projects: ProjectService,
                 datasets: DatasetRepository, configurations, on_submit=None,
                 csv_root: Path = DEFAULT_CSV_ROOT,
                 model_root: Path = DEFAULT_MODEL_ROOT,
                 sessions=None, sql_server=None) -> None:
        self._store = store
        self._projects = projects
        self._datasets = datasets
        self._configurations = configurations
        self._on_submit = on_submit
        self._csv_root = Path(csv_root)
        self._model_root = Path(model_root)
        self._sessions = sessions
        self._sql_server = sql_server

    def _dataset(self, owner: str, project_id: str, dataset_id: str | None):
        if dataset_id is None:
            active = [d for d in self._datasets.list_for_project(owner, project_id)
                      if d.active_training]
            if not active:
                raise RunRequestRefused("the project has no active training dataset")
            return active[0]
        try:
            return self._datasets.get_owned(owner, project_id, dataset_id)
        except KeyError:
            from JBGDatasetService import DatasetUnavailable
            raise DatasetUnavailable("dataset unavailable") from None

    def submit(self, principal, project_id: str, *, configuration_id: str,
               version: int | None = None, dataset_id: str | None = None) -> dict:
        owner = principal.principal_key
        project = self._projects.require(owner, project_id)
        if project.source is DataSourceKind.SQL and (
                self._sql_server is None or not self._sql_server.configured
                or self._sessions is None):
            raise RunRequestRefused("no SQL Server is configured on this server")
        configuration = self._configurations.stored_configuration(
            owner, project_id, configuration_id, version)
        settings = configuration.settings
        if not settings["mode"].get("train"):
            raise RunRequestRefused(
                "prediction-only runs need a saved model and are not available yet")
        if project.source is DataSourceKind.CSV and settings["mode"].get("predict"):
            # 294a: the engine's CSV training path is train-only
            # (Storage.CSVTrainingApplication); refuse before a worker starts.
            raise RunRequestRefused("CSV training runs cannot also predict")
        dataset = self._dataset(owner, project_id, dataset_id)
        if dataset.role is DatasetRole.PREDICTION:
            raise RunRequestRefused("a training run needs a training or mixed dataset")
        # 293a (decision A): the column roles come from the dataset.
        roles = self._datasets.get_roles(owner, project_id, dataset.dataset_id)
        if roles is None:
            raise RunRequestRefused(
                "the dataset has no column roles; set them on the Dataset tab first")

        # Decision 3: one model file per training run, outside Voilà's folders.
        model_id = str(uuid4())
        model_folder = self._model_root / project.project_id
        metadata = {"project_id": project.project_id,
                    "configuration_id": configuration.configuration_id,
                    "configuration_version": configuration.version,
                    "configuration_name": configuration.name,
                    "dataset_id": dataset.dataset_id, "model_id": model_id}
        if project.source is DataSourceKind.SQL:
            return self._submit_sql(principal, settings, dataset, model_folder, model_id,
                                    metadata, roles)
        return self._submit_csv(principal, project, settings, dataset, model_folder,
                                model_id, metadata, roles)

    def submit_prediction(self, principal, project_id: str, *, model_id: str,
                          dataset_id: str | None = None) -> dict:
        """295c: predict with a saved model (SQL). The training run's own engine
        configuration (roles and model file included) is reused in prediction-
        only mode on the chosen dataset, or on the training dataset's rows
        without a class. A table may lack the class column; it is generated."""
        from JBGApplicationContracts import RunReference
        from JBGModelCatalog import ModelCatalog, ModelState
        owner = principal.principal_key
        project = self._projects.require(owner, project_id)
        if project.source is DataSourceKind.CSV:
            raise RunRequestRefused("predicting with a saved model is not available for CSV "
                                    "projects yet")
        if self._sql_server is None or not self._sql_server.configured or self._sessions is None:
            raise RunRequestRefused("no SQL Server is configured on this server")
        try:
            entry = ModelCatalog(self._store.engine, schema=self._store.schema).get_owned(
                owner, project_id, model_id)
        except KeyError:                    # unknown, or another user's model
            raise RunRequestRefused("the model is not available") from None
        if entry.state is not ModelState.AVAILABLE or not entry.run_id:
            raise RunRequestRefused("the model is not available")
        try:
            training = self._store.get_owned(owner, RunReference(entry.run_id))
        except (KeyError, ValueError):
            raise RunRequestRefused("the model is not available") from None
        trained = training.spec.get("project_run") or {}
        config = _load_config(training.spec["configuration"])
        if not Path(config.get_model_filename()).is_file():
            raise RunRequestRefused("the model file is missing")
        dataset = self._dataset(owner, project_id, dataset_id or trained.get("dataset_id"))
        config.mode.train, config.mode.predict = False, True
        config.mode.mispredicted = False
        config.mode.calculate_feature_importance = False
        self._sql_server.apply_to(config)
        config.connection.data_catalog = dataset.catalog
        config.connection.data_table = dataset.table
        metadata = {"project_id": project.project_id, "kind": "prediction", "model_id": model_id,
                    "dataset_id": dataset.dataset_id,
                    "configuration_name": trained.get("configuration_name"),
                    "configuration_version": trained.get("configuration_version")}
        from JBGRunComposition import compose_sql_run
        from JBGRunSubmission import _SQL_SCOPE
        composed = compose_sql_run(self._sessions, principal, config, _QuietLogger())
        prepared = composed.preparation.prepare(
            DatasetReference(DataSourceKind.SQL, composed.sql_dataset_key()),
            RunConfiguration(DataSourceKind.SQL, composed.configuration))
        service = DurableRunService(self._store, scope_for=lambda principal: _SQL_SCOPE,
                                    authorize_dataset=self._sessions.authorize,
                                    on_submit=self._on_submit)
        return service.submit(principal, prepared, run_metadata=metadata)

    def _submit_sql(self, principal, settings, dataset, model_folder, model_id, metadata,
                    roles):
        from JBGRunComposition import compose_sql_run
        from JBGRunSubmission import _SQL_SCOPE, validate_saved_sql_run_config
        config = _load_config(settings)
        roles.apply_to(config)
        self._sql_server.apply_to(config)
        config.connection.data_catalog = dataset.catalog
        config.connection.data_table = dataset.table
        try:
            validate_saved_sql_run_config(config)
        except ValueError as error:
            # Fixed text naming the field only (285h), never its value.
            raise RunDoesNotMatchDataset(str(error)) from None
        model_folder.mkdir(parents=True, exist_ok=True)
        config.io.model_path = str(model_folder)
        config.io.model_name = model_id
        # Opens the run's DataLayer with the user's own login for this server;
        # PermissionError("SQL sign-in required") without one.
        composed = compose_sql_run(self._sessions, principal, config, _QuietLogger())
        prepared = composed.preparation.prepare(
            DatasetReference(DataSourceKind.SQL, composed.sql_dataset_key()),
            RunConfiguration(DataSourceKind.SQL, composed.configuration))
        service = DurableRunService(self._store, scope_for=lambda principal: _SQL_SCOPE,
                                    authorize_dataset=self._sessions.authorize,
                                    on_submit=self._on_submit)
        return service.submit(principal, prepared, run_metadata=metadata)

    def _submit_csv(self, principal, project, settings, dataset, model_folder, model_id,
                    metadata, roles):
        repository, scope = project_csv_storage(project.project_id, root=self._csv_root)
        artifact_id = dataset.registration_id
        csv_format = csv_upload_format(repository, scope, artifact_id)
        base = _load_config(dict(settings, data_source=dict(
            settings["data_source"], kind="csv", artifact_id=artifact_id,
            dataset_id=_upload_dataset_id(repository, scope, artifact_id), **csv_format)))
        roles.apply_to(base)
        connection = base.connection
        text_columns = tuple(connection.data_text_columns)
        data_columns = tuple(c for c in connection.data_numerical_columns
                             if c not in text_columns)

        from Storage.CSVGUITrainingConfig import prepare_csv_gui_training_config
        try:
            # The roles were not checked at upload (agreed Dataset flow); check
            # them against the stored file now, as Voilà does at registration.
            check_csv_roles(repository, scope, artifact_id, csv_format,
                            id_column=connection.id_column, class_column=connection.class_column,
                            feature_columns=data_columns + text_columns)
            # Voila's model folders are replaced below (decision 3); a temporary
            # root keeps them from being created.
            with tempfile.TemporaryDirectory(prefix="jbg_models_") as unused_root:
                runtime, record = prepare_csv_gui_training_config(
                    base, repository, scope, artifact_id, id_column=connection.id_column,
                    class_column=connection.class_column, data_columns=data_columns,
                    text_columns=text_columns, csv_format=csv_format,
                    model_root=Path(unused_root), predict=bool(base.mode.predict),
                    mispredicted=bool(base.mode.mispredicted))
        except (ValueError, TypeError, KeyError):
            # Missing or wrongly typed columns; details stay server-side.
            raise RunDoesNotMatchDataset("the column roles do not fit the dataset") from None

        model_folder.mkdir(parents=True, exist_ok=True)
        runtime.io.model_path = str(model_folder)
        runtime.io.model_name = model_id

        from JBGRunComposition import compose_csv_run
        composed = compose_csv_run(repository, scope, record.configuration_id, _QuietLogger())
        prepared = composed.preparation.prepare(
            DatasetReference(DataSourceKind.CSV, artifact_id),
            RunConfiguration(DataSourceKind.CSV, runtime))
        service = DurableRunService(self._store, scope_for=lambda principal: scope,
                                    on_submit=self._on_submit)
        return service.submit(principal, prepared, configuration_id=record.configuration_id,
                              run_metadata=metadata)


def _upload_dataset_id(repository, scope, artifact_id: str) -> str:
    from Storage.ArtifactContract import ArtifactKind
    for ref in repository.list_artifacts(scope):
        if ref.artifact_id == artifact_id and ref.kind == ArtifactKind.DATASET:
            return ref.dataset_id
    raise FileNotFoundError("registered CSV not found")
