"""Owner-bound dataset service for the HTTP API (revision 286b).

Datasets belong to a project and inherit its source type (enforced by the 285k
repository). SQL datasets name a catalog and table; CSV datasets are uploaded
into the project's own storage (JBGProjectStorage) and referenced by an opaque
artifact ID. Nothing here reads SQL data or deletes datasets; validation (the
"Validate dataset" button) and deletion follow in later revisions, so new
datasets stay "unverified".
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, BinaryIO

from JBGApplicationContracts import DataSourceKind
from JBGDatasets import Dataset, DatasetRepository, DatasetRole, DuplicateDatasetName
from JBGProjectService import ProjectService
from JBGProjectStorage import (DEFAULT_CSV_ROOT, MAX_UPLOAD_BYTES, detect_delimiter,
                               project_csv_storage, register_csv_upload)


class DatasetUnavailable(LookupError):
    """Unknown dataset, or a dataset of another user's project."""


class DatasetConflict(ValueError):
    """The project already has a dataset with that name."""


class UploadTooLarge(ValueError):
    """The uploaded file exceeds the 25 MB limit."""


class WrongSourceType(ValueError):
    """SQL data for a CSV project, or the other way round."""


def dataset_response(dataset: Dataset, *, upload=None) -> dict[str, Any]:
    response = {"dataset_id": dataset.dataset_id, "project_id": dataset.project_id,
                "name": dataset.name, "source": dataset.source.value,
                "role": dataset.role.value, "active_training": dataset.active_training,
                "validation_state": dataset.validation_state.value,
                "created_at": dataset.created_at.isoformat(),
                "updated_at": dataset.updated_at.isoformat()}
    if dataset.source is DataSourceKind.SQL:
        response.update(catalog=dataset.catalog, table=dataset.table)
    if upload is not None:
        # Structure of the stored file, for the GUI's preview step.
        response["upload"] = {"rows": upload.rows, "columns": list(upload.columns),
                              "delimiter": upload.delimiter, "encoding": upload.encoding}
    return response


def _role(value: str) -> DatasetRole:
    try:
        return DatasetRole(value)
    except ValueError:
        raise ValueError("role must be training, prediction or mixed") from None


class DatasetService:
    def __init__(self, repository: DatasetRepository, projects: ProjectService, *,
                 csv_root: Path = DEFAULT_CSV_ROOT, compatibility=None) -> None:
        if not isinstance(repository, DatasetRepository):
            raise TypeError("DatasetRepository required")
        if not isinstance(projects, ProjectService):
            raise TypeError("ProjectService required")
        self._repository = repository
        self._projects = projects
        self._csv_root = Path(csv_root)
        # 293a: check(principal, project_id, candidate) for prediction datasets.
        self._compatibility = compatibility

    def _project(self, owner: str, project_id: str, source: DataSourceKind | None = None):
        project = self._projects.require(owner, project_id)
        if source is not None and project.source is not source:
            raise WrongSourceType(f"this is a {project.source.value.upper()} project")
        return project

    def list(self, owner: str, project_id: str) -> list[dict[str, Any]]:
        self._project(owner, project_id)
        return [dataset_response(d) for d in self._repository.list_for_project(owner, project_id)]

    def get(self, owner: str, project_id: str, dataset_id: str) -> dict[str, Any]:
        self._project(owner, project_id)
        try:
            return dataset_response(self._repository.get_owned(owner, project_id, dataset_id))
        except KeyError:
            raise DatasetUnavailable("dataset unavailable") from None

    def _check_prediction(self, principal, project_id, role, candidate) -> None:
        """293a: a prediction dataset must fit the active training dataset."""
        if role is DatasetRole.PREDICTION and self._compatibility is not None \
                and principal is not None:
            self._compatibility(principal, project_id, candidate)

    def create_sql(self, owner: str, project_id: str, *, name: str, role: str,
                   catalog: str, table: str, principal=None) -> dict[str, Any]:
        from types import SimpleNamespace
        self._project(owner, project_id, DataSourceKind.SQL)
        role_value = _role(role)
        if role_value is DatasetRole.PREDICTION:
            # Names are checked by the repository before anything is read.
            from JBGDatasets import _plain_sql_names
            _plain_sql_names(catalog, table)
        self._check_prediction(principal, project_id, role_value, SimpleNamespace(
            source=DataSourceKind.SQL, catalog=catalog, table=table, registration_id=None))
        try:
            dataset = self._repository.create(owner, project_id, name, _role(role),
                                              catalog=catalog, table=table)
        except DuplicateDatasetName:
            raise DatasetConflict("dataset name already exists") from None
        return dataset_response(dataset)

    def upload_csv(self, owner: str, project_id: str, *, name: str, role: str,
                   stream: BinaryIO, filename: str | None = None,
                   delimiter: str | None = None,
                   encoding: str = "utf-8-sig", principal=None) -> dict[str, Any]:
        """Store an uploaded CSV in the project's storage and register it."""
        project = self._project(owner, project_id, DataSourceKind.CSV)
        role = _role(role)
        repository, scope = project_csv_storage(project.project_id, root=self._csv_root)
        with tempfile.TemporaryDirectory(prefix="jbg_upload_") as folder:
            path = Path(folder) / "upload.csv"
            with path.open("wb") as target:
                # Copy at most limit + 1 bytes, so a huge upload is never read whole.
                copied = _copy_limited(stream, target, MAX_UPLOAD_BYTES + 1)
            if copied > MAX_UPLOAD_BYTES:
                raise UploadTooLarge("CSV file exceeds 25 MB")
            if delimiter is None:
                with path.open("rb") as reader:
                    delimiter = detect_delimiter(reader.read(64 * 1024), encoding)
            upload = register_csv_upload(repository, scope, path, delimiter=delimiter,
                                         encoding=encoding, original_name=filename)
        try:
            from types import SimpleNamespace
            self._check_prediction(principal, project_id, role, SimpleNamespace(
                source=DataSourceKind.CSV, catalog=None, table=None,
                registration_id=upload.artifact_id))
            dataset = self._repository.create(owner, project_id, name, role,
                                              registration_id=upload.artifact_id)
        except BaseException as error:
            # Do not leave an unreferenced file behind (e.g. duplicate name).
            from Storage.CSVDatasetCatalog import erase_csv_dataset
            try:
                erase_csv_dataset(repository, scope, upload.artifact_id)
            except Exception:
                pass
            if isinstance(error, DuplicateDatasetName):
                raise DatasetConflict("dataset name already exists") from None
            raise
        return dataset_response(dataset, upload=upload)

    def activate(self, owner: str, project_id: str, dataset_id: str) -> dict[str, Any]:
        self._project(owner, project_id)
        try:
            return dataset_response(self._repository.activate_training(owner, project_id,
                                                                        dataset_id))
        except KeyError:
            raise DatasetUnavailable("dataset unavailable") from None


def _copy_limited(source: BinaryIO, target: BinaryIO, limit: int) -> int:
    copied = 0
    while copied < limit:
        chunk = source.read(min(1024 * 1024, limit - copied))
        if not chunk:
            break
        target.write(chunk)
        copied += len(chunk)
    return copied
