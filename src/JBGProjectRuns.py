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

288a covers CSV projects; SQL projects follow in 288b. The CSV path builds the
run exactly as Voilà does at training start (``prepare_csv_gui_training_config``),
which is also where the configuration's column roles are checked against the
dataset.
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
                 model_root: Path = DEFAULT_MODEL_ROOT) -> None:
        self._store = store
        self._projects = projects
        self._datasets = datasets
        self._configurations = configurations
        self._on_submit = on_submit
        self._csv_root = Path(csv_root)
        self._model_root = Path(model_root)

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
        if project.source is not DataSourceKind.CSV:
            raise RunRequestRefused("runs for SQL projects are not available yet")
        configuration = self._configurations.stored_configuration(
            owner, project_id, configuration_id, version)
        settings = configuration.settings
        if not settings["mode"].get("train"):
            raise RunRequestRefused(
                "prediction-only runs need a saved model and are not available yet")
        dataset = self._dataset(owner, project_id, dataset_id)
        if dataset.role is DatasetRole.PREDICTION:
            raise RunRequestRefused("a training run needs a training or mixed dataset")

        repository, scope = project_csv_storage(project.project_id, root=self._csv_root)
        artifact_id = dataset.registration_id
        csv_format = csv_upload_format(repository, scope, artifact_id)
        base = _load_config(dict(settings, data_source=dict(
            settings["data_source"], kind="csv", artifact_id=artifact_id,
            dataset_id=_upload_dataset_id(repository, scope, artifact_id), **csv_format)))
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
            raise RunDoesNotMatchDataset("the configuration's column roles do not fit "
                                         "the dataset") from None

        # Decision 3: one model file per training run, outside Voilà's folders.
        model_id = str(uuid4())
        model_folder = self._model_root / project.project_id
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
                              run_metadata={"project_id": project.project_id,
                                            "configuration_id": configuration.configuration_id,
                                            "configuration_version": configuration.version,
                                            "dataset_id": dataset.dataset_id,
                                            "model_id": model_id})


def _upload_dataset_id(repository, scope, artifact_id: str) -> str:
    from Storage.ArtifactContract import ArtifactKind
    for ref in repository.list_artifacts(scope):
        if ref.artifact_id == artifact_id and ref.kind == ArtifactKind.DATASET:
            return ref.dataset_id
    raise FileNotFoundError("registered CSV not found")
