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


def keeps_groups_apart(settings: dict) -> bool:
    """298g: whether a configuration needs a group column (grouped
    cross-validation or a holdout that keeps groups apart)."""
    mode = (settings or {}).get("mode") or {}
    return bool(mode.get("grouped_cross_validation")
                or mode.get("holdout_split_mode") == "person_disjoint")


from JBGDatasetRoles import INTEGER_SQL_TYPES  # noqa: E402  (298i)


class RunRequestRefused(ValueError):
    """The project cannot start this run (fixed message, 409)."""


class IdColumnNotInteger(RunRequestRefused):
    """298i: an SQL run's ID column is not a whole-number column."""


class RunDoesNotMatchDataset(ValueError):
    """The configuration's column roles do not fit the dataset (422)."""


class NothingToPredict(RunRequestRefused):
    """297f: every row of the chosen dataset already has a class (409)."""


class _QuietLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


# 297b: the row limit an SQL run gets when the configuration leaves "Row limit"
# empty ("Empty reads all rows"). The SQL engine reads an empty limit as zero
# rows (Voilà always fills it in with the table's row count) but caps any limit
# at the rows eligible when the run fetches, so this reads them all, also from a
# table that has grown since it was registered. Runtime only: the saved
# configuration keeps its empty limit.
SQL_ALL_ROWS = 2_147_483_647            # the largest SQL Server int


def _read_all_rows_unless_limited(config) -> None:
    limit = config.debug.data_limit
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        config.debug.data_limit = SQL_ALL_ROWS


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
        if keeps_groups_apart(settings) and not roles.group_column:
            # 298g: the engine would stop at the split; refuse before a worker starts.
            raise RunRequestRefused(
                "the configuration keeps groups apart, but the dataset has no group column")

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
        if project.source is DataSourceKind.SQL and (
                self._sql_server is None or not self._sql_server.configured or self._sessions is None):
            raise RunRequestRefused("no SQL Server is configured on this server")
        try:
            entry = ModelCatalog(self._store.engine, schema=self._store.schema).get_owned(
                owner, project_id, model_id)
        except KeyError:                    # unknown, or another user's model
            raise RunRequestRefused("the model is not available") from None
        if entry.state is not ModelState.AVAILABLE or not entry.run_id:
            raise RunRequestRefused("the model is not available")
        try:
            # 299e: also when the training run was deleted (its settings are kept).
            training = self._store.get_owned(owner, RunReference(entry.run_id),
                                             include_hidden=True)
        except (KeyError, ValueError):
            raise RunRequestRefused("the model is not available") from None
        trained = training.spec.get("project_run") or {}
        config = _load_config(training.spec["configuration"])
        if not Path(config.get_model_filename()).is_file():
            raise RunRequestRefused("the model file is missing")
        if project.source is DataSourceKind.CSV:
            return self._submit_csv_prediction(principal, project, training, trained, config,
                                               model_id, dataset_id)
        dataset = self._dataset(owner, project_id, dataset_id or trained.get("dataset_id"))
        config.mode.train, config.mode.predict = False, True
        config.mode.mispredicted = False
        config.mode.calculate_feature_importance = False
        self._sql_server.apply_to(config)
        config.connection.data_catalog = dataset.catalog
        config.connection.data_table = dataset.table
        # 297b: also for models trained before 297b, whose stored configuration
        # may hold the empty limit (it would predict no rows).
        _read_all_rows_unless_limited(config)
        metadata = {"project_id": project.project_id, "kind": "prediction", "model_id": model_id,
                    "dataset_id": dataset.dataset_id,
                    "configuration_name": trained.get("configuration_name"),
                    "configuration_version": trained.get("configuration_version")}
        from JBGRunSubmission import _SQL_SCOPE
        composed = self._compose_checked_sql(principal, config)
        prepared = composed.preparation.prepare(
            DatasetReference(DataSourceKind.SQL, composed.sql_dataset_key()),
            RunConfiguration(DataSourceKind.SQL, composed.configuration))
        service = DurableRunService(self._store, scope_for=lambda principal: _SQL_SCOPE,
                                    authorize_dataset=self._sessions.authorize,
                                    on_submit=self._on_submit)
        return service.submit(principal, prepared, run_metadata=metadata)

    def _submit_csv_prediction(self, principal, project, training, trained, config, model_id,
                               dataset_id):
        """296b (agreed 2026-11-09): a CSV model predicts the rows without a class
        of its own training dataset, as in the Voilà GUI; the CSV prediction
        path is pinned to that dataset's identity. The configuration is the
        training run's own (complete, with the roles and data source identity
        the CSV prediction path checks), switched to prediction only. The model
        file's embedded configuration is not used here: after training it names
        the single chosen algorithm, which is not a valid run configuration."""
        from copy import deepcopy
        from JBGRunComposition import compose_csv_run
        csv = training.spec.get("csv") or {}
        if not csv.get("artifact_id") or not csv.get("configuration_id"):
            raise RunRequestRefused("the model is not available")
        if dataset_id and dataset_id != trained.get("dataset_id"):
            # 297f (decision 4 changed 2026-11-10): another CSV dataset of the project.
            return self._submit_csv_prediction_on(principal, project, trained, config, csv,
                                                  model_id, dataset_id)
        runtime = deepcopy(config)
        runtime.mode.train, runtime.mode.predict = False, True
        runtime.mode.mispredicted = False
        runtime.mode.use_metas = False
        runtime.mode.calculate_feature_importance = False
        runtime.mode.calculate_dark_numbers = False
        runtime.io.model_path, runtime.io.model_name = config.io.model_path, config.io.model_name
        metadata = {"project_id": project.project_id, "kind": "prediction", "model_id": model_id,
                    "dataset_id": trained.get("dataset_id"),
                    "configuration_name": trained.get("configuration_name"),
                    "configuration_version": trained.get("configuration_version")}
        repository, scope = project_csv_storage(project.project_id, root=self._csv_root)
        composed = compose_csv_run(repository, scope, csv["configuration_id"], _QuietLogger())
        prepared = composed.preparation.prepare(
            DatasetReference(DataSourceKind.CSV, csv["artifact_id"]),
            RunConfiguration(DataSourceKind.CSV, runtime))
        service = DurableRunService(self._store, scope_for=lambda principal: scope,
                                    on_submit=self._on_submit)
        return service.submit(principal, prepared, configuration_id=csv["configuration_id"],
                              run_metadata=metadata)

    def _submit_csv_prediction_on(self, principal, project, trained, config, csv, model_id,
                                  dataset_id):
        """297f: a CSV model predicts another CSV dataset of its project.

        The dataset must be a prediction or mixed dataset whose columns fit the
        model (``check_csv_prediction_columns``); the class column may be
        missing (then every row is predicted, as for SQL), otherwise its rows
        without a class are predicted. The dataset gets its own CSV registration
        with the model's roles, the run reads its rows, and the worker checks
        the model against its training dataset (``project_run.model_source``,
        set as ``Config.model_source``, 297e).
        """
        from copy import deepcopy
        from JBGDatasetRoles import IncompatibleDataset
        from JBGProjectStorage import check_csv_prediction_columns
        from JBGRunComposition import compose_csv_run
        from Storage.CSVModelConfiguration import (load_csv_model_configuration,
                                                   save_csv_model_configuration)
        owner = principal.principal_key
        dataset = self._dataset(owner, project.project_id, dataset_id)
        if dataset.role.value not in ("prediction", "mixed") or not dataset.registration_id:
            raise RunRequestRefused("choose a prediction or mixed dataset")
        repository, scope = project_csv_storage(project.project_id, root=self._csv_root)
        model_roles = load_csv_model_configuration(repository, scope, csv["configuration_id"])
        artifact_id = dataset.registration_id
        csv_format = csv_upload_format(repository, scope, artifact_id)
        try:
            check_csv_prediction_columns(
                repository, scope, artifact_id, csv_format, id_column=model_roles.id_column,
                class_column=model_roles.class_column,
                numeric_columns=model_roles.data_columns, text_columns=model_roles.text_columns)
        except IncompatibleDataset:
            raise RunDoesNotMatchDataset("the dataset does not fit the model") from None
        except ValueError as error:
            if str(error) == "nothing to predict":
                raise NothingToPredict("every row of the dataset has a class") from None
            raise RunDoesNotMatchDataset("the dataset does not fit the model") from None
        registered = save_csv_model_configuration(
            repository, scope, artifact_id, id_column=model_roles.id_column,
            class_column=model_roles.class_column, data_columns=model_roles.data_columns,
            text_columns=model_roles.text_columns, csv_format=csv_format)
        trained_source = config.data_source
        runtime = deepcopy(config)
        runtime.mode.train, runtime.mode.predict = False, True
        runtime.mode.mispredicted = False
        runtime.mode.use_metas = False
        runtime.mode.calculate_feature_importance = False
        runtime.mode.calculate_dark_numbers = False
        runtime.data_source = type(trained_source)(
            kind="csv", dataset_id=registered.dataset_id, artifact_id=artifact_id,
            display_name=dataset.name[:200], **csv_format)
        runtime.data_source.validate()
        metadata = {"project_id": project.project_id, "kind": "prediction", "model_id": model_id,
                    "dataset_id": dataset.dataset_id,
                    "configuration_name": trained.get("configuration_name"),
                    "configuration_version": trained.get("configuration_version"),
                    "model_source": {"dataset_id": trained_source.dataset_id,
                                     "artifact_id": trained_source.artifact_id}}
        composed = compose_csv_run(repository, scope, registered.configuration_id, _QuietLogger())
        prepared = composed.preparation.prepare(
            DatasetReference(DataSourceKind.CSV, artifact_id),
            RunConfiguration(DataSourceKind.CSV, runtime))
        service = DurableRunService(self._store, scope_for=lambda principal: scope,
                                    on_submit=self._on_submit)
        return service.submit(principal, prepared, configuration_id=registered.configuration_id,
                              run_metadata=metadata)

    def _compose_checked_sql(self, principal, config):
        """Open the run's DataLayer with the user's own login and check, with one
        metadata query, that the ID column is a whole-number column (298i);
        then compose the run on that same layer."""
        from JBGRunComposition import compose_sql_run_on_layer
        logger = _QuietLogger()
        # PermissionError("SQL sign-in required") without a sign-in.
        layer = self._sessions.data_layer_for(principal, config, logger)
        try:
            types = layer.get_table_columns(config.connection.data_catalog,
                                            config.connection.data_table)
        except Exception:
            types = None                    # the run's own checks report an unreachable table
        if isinstance(types, dict):
            kind = types.get(config.connection.id_column)
            if kind is not None and str(kind).lower() not in INTEGER_SQL_TYPES:
                raise IdColumnNotInteger(
                    "the ID column is not a whole-number column (int, bigint, smallint, tinyint)")
        return compose_sql_run_on_layer(layer, logger)

    def _submit_sql(self, principal, settings, dataset, model_folder, model_id, metadata,
                    roles):
        from JBGRunSubmission import _SQL_SCOPE, validate_saved_sql_run_config
        config = _load_config(settings)
        roles.apply_to(config)
        self._sql_server.apply_to(config)
        config.connection.data_catalog = dataset.catalog
        config.connection.data_table = dataset.table
        _read_all_rows_unless_limited(config)                    # 297b
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
        composed = self._compose_checked_sql(principal, config)
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
