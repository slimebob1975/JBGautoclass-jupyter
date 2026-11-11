"""CSV storage for API projects, keyed by stable project ID (revision 286b).

Agreed 2026-11-04: API projects keep their CSV data under the local installation
identity and the project's own UUIDv4, so renaming a project changes nothing.
Voila's storage, keyed by a hash of the project name, shares the physical root
but never the scope, so neither side sees the other's data.

An upload is checked for structure only (headers, consistent rows, at least one
data row, delimiter, encoding), using the same inspection and storage format as
Voila's registration. Column roles are chosen afterwards (agreed Dataset flow)
and checked with the configuration and dataset validation.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_ROOT = ROOT / "output" / "csv_artifacts"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_DELIMITERS = ",;\t|"
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def project_csv_storage(project_id: str, *, root: Path = DEFAULT_CSV_ROOT):
    """Repository and scope for one API project (scope = installation + project ID)."""
    from Storage.FileArtifactRepository import FileArtifactRepository
    from Storage.LocalCSVScope import local_tenant_id
    try:
        if UUID(project_id).version != 4:
            raise ValueError
    except (TypeError, ValueError, AttributeError):
        raise ValueError("invalid project identifier") from None
    root = Path(root)
    tenant = local_tenant_id(root)
    return FileArtifactRepository(root / "objects"), StorageScope(str(tenant), project_id)


def detect_delimiter(sample: bytes, encoding: str = "utf-8-sig") -> str:
    """The same detection as Voila's CSV panel; comma when undecidable."""
    try:
        text = sample.decode(encoding, errors="replace")
        return csv.Sniffer().sniff(text, delimiters=_DELIMITERS).delimiter
    except csv.Error:
        return ","


@dataclass(frozen=True)
class UploadedCSV:
    artifact_id: str
    rows: int
    columns: tuple[str, ...]
    delimiter: str
    encoding: str


def register_csv_upload(repository, scope: StorageScope, path: Path, *, delimiter: str,
                        encoding: str, original_name: str | None = None) -> UploadedCSV:
    """Check structure and store the file as a dataset artifact in ``scope``."""
    from CSVColumnSchema import inspect_csv_columns
    if not isinstance(scope, StorageScope):
        raise ValueError("Invalid dataset storage scope")
    if delimiter not in _DELIMITERS:
        raise ValueError("Invalid CSV delimiter")
    if encoding not in _ENCODINGS:
        raise ValueError("Invalid CSV encoding")
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        raise ValueError("CSV input is unavailable") from None
    if not 0 < size <= MAX_UPLOAD_BYTES:
        raise ValueError("CSV input is empty or exceeds the size limit")
    schema = inspect_csv_columns(path, delimiter=delimiter, encoding=encoding)
    dataset_id = str(uuid4())
    reference = ArtifactReference(scope, str(uuid4()), ArtifactKind.DATASET, "text/csv",
                                  dataset_id=dataset_id)
    try:
        with path.open("rb") as reader:
            repository.save_artifact(scope, reference, reader)
    except OSError:
        raise ValueError("Unable to persist CSV dataset") from None
    # Display name (as in Voilà's registration; never a client path) and, from
    # 288a, the CSV format a run needs. Readers ignore keys they do not know.
    from Storage.CSVDatasetCatalog import clean_csv_display_name
    meta = {"version": 1, "dataset_id": dataset_id,
            "filename": clean_csv_display_name(original_name or "upload.csv"),
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "delimiter": delimiter, "encoding": encoding}
    manifest = ArtifactReference(scope, str(uuid4()), ArtifactKind.MANIFEST,
                                 "application/json", dataset_id=dataset_id)
    repository.save_artifact(scope, manifest, BytesIO(json.dumps(meta).encode("utf-8")))
    return UploadedCSV(reference.artifact_id, schema.rows, tuple(schema.columns),
                       delimiter, encoding)


def csv_upload_format(repository, scope: StorageScope, artifact_id: str) -> dict:
    """The stored CSV format of an uploaded dataset (288a).

    Uploads from 286b recorded no format; for them it is detected from the file
    exactly as at upload time (delimiter sniffing, utf-8-sig).
    """
    refs = list(repository.list_artifacts(scope))
    upload = next((ref for ref in refs if ref.artifact_id == artifact_id
                   and ref.kind == ArtifactKind.DATASET), None)
    if upload is None:
        raise FileNotFoundError("registered CSV not found")
    for ref in refs:
        if ref.kind != ArtifactKind.MANIFEST or ref.dataset_id != upload.dataset_id \
                or ref.run_id is not None:
            continue
        try:
            with repository.open_artifact(scope, ref.artifact_id) as stream:
                record = json.loads(stream.read(4096))
        except (ValueError, OSError, UnicodeError):
            continue
        # Other manifests of the same dataset (e.g. a run's saved roles record)
        # carry no format: only an upload manifest with text values counts.
        if not isinstance(record, dict):
            continue
        delimiter, encoding = record.get("delimiter"), record.get("encoding")
        if (isinstance(delimiter, str) and delimiter in _DELIMITERS
                and isinstance(encoding, str) and encoding in _ENCODINGS):
            return {"delimiter": delimiter, "quotechar": '"', "encoding": encoding}
    with repository.open_artifact(scope, artifact_id) as stream:
        sample = stream.read(64 * 1024)
    return {"delimiter": detect_delimiter(sample, "utf-8-sig"), "quotechar": '"',
            "encoding": "utf-8-sig"}


def check_csv_roles(repository, scope: StorageScope, artifact_id: str, csv_format: dict, *,
                    id_column: str, class_column: str, feature_columns) -> None:
    """Check column roles against the stored CSV, as Voilà does at registration.

    Raises ValueError with Voilà's fixed messages (no values) when a role is
    missing, overlapping or unsuitable (e.g. a non-unique ID column).
    """
    import shutil
    import tempfile
    from CSVColumnSchema import inspect_csv_columns, validate_csv_column_roles
    with tempfile.TemporaryDirectory(prefix="jbg_roles_") as folder:
        path = Path(folder) / "data.csv"
        with repository.open_artifact(scope, artifact_id) as source, path.open("wb") as target:
            shutil.copyfileobj(source, target)
        schema = inspect_csv_columns(path, delimiter=csv_format["delimiter"],
                                     encoding=csv_format["encoding"])
    validate_csv_column_roles(schema, id_column=id_column, class_column=class_column,
                              feature_columns=tuple(feature_columns))


def check_csv_prediction_columns(repository, scope: StorageScope, artifact_id: str,
                                 csv_format: dict, *, id_column: str, class_column: str,
                                 numeric_columns, text_columns) -> bool:
    """297f: does a separate CSV prediction dataset fit a model's roles?

    The ID and feature columns must exist, the ID column must be filled in and
    unique, and the model's numeric features must be numbers here; the class
    column may be missing. Raises JBGDatasetRoles.IncompatibleDataset (column
    names only, no values) or ValueError("nothing to predict") when every row
    already has a class. Returns whether the file has the class column.
    """
    import shutil
    import tempfile
    import pandas as pd
    from CSVColumnSchema import inspect_csv_columns
    from JBGDatasetRoles import IncompatibleDataset
    numeric_columns, text_columns = tuple(numeric_columns), tuple(text_columns)
    with tempfile.TemporaryDirectory(prefix="jbg_predict_") as folder:
        path = Path(folder) / "data.csv"
        with repository.open_artifact(scope, artifact_id) as source, path.open("wb") as target:
            shutil.copyfileobj(source, target)
        schema = inspect_csv_columns(path, delimiter=csv_format["delimiter"],
                                     encoding=csv_format["encoding"])
        missing = [name for name in (id_column, *numeric_columns, *text_columns)
                   if name not in schema.columns]
        different = [name for name in numeric_columns
                     if name in schema.columns and name in schema.text_feature_candidates]
        if missing or different:
            raise IncompatibleDataset(missing, different)
        if id_column not in schema.unique_id_candidates:
            raise IncompatibleDataset((), (id_column,))
        has_class = class_column in schema.columns
        if has_class:
            values = pd.read_csv(path, sep=csv_format["delimiter"],
                                 quotechar=csv_format.get("quotechar", '"'),
                                 encoding=csv_format["encoding"], dtype=object,
                                 keep_default_na=False, usecols=[class_column])[class_column]
            stripped = values.astype("string").str.strip()
            unlabelled = (values.isna() | stripped.eq("").fillna(True)
                          | stripped.str.casefold().eq("null").fillna(True))
            if not unlabelled.any():
                raise ValueError("nothing to predict")
    return has_class
