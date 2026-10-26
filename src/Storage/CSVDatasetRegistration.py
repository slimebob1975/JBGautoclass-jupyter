"""Explicit, SQL-free registration of validated CSV datasets as artifacts.

The caller must supply a trusted, authorized StorageScope. This module does not
provide user authentication, an HTTP endpoint, or classifier execution.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4

from CSVColumnSchema import inspect_csv_columns, validate_csv_column_roles
from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope


@dataclass(frozen=True)
class RegisteredCSVDataset:
    dataset_id: str
    reference: ArtifactReference
    rows: int
    columns: int


def register_csv_dataset(repository, scope: StorageScope, path, *, id_column,
                         class_column, feature_columns, delimiter=",",
                         encoding="utf-8-sig", original_name=None, max_bytes=25 * 1024 * 1024):
    """Validate CSV roles and persist original bytes, returning opaque IDs only."""
    if not isinstance(scope, StorageScope):
        raise ValueError("Invalid dataset storage scope")
    if not isinstance(max_bytes, int) or not 0 < max_bytes <= 25 * 1024 * 1024:
        raise ValueError("Invalid CSV size limit")
    try:
        source = Path(path)
        size = source.stat().st_size
        if not source.is_file() or not 0 < size <= max_bytes:
            raise ValueError
    except (OSError, ValueError, TypeError):
        raise ValueError("CSV input is unavailable or exceeds the size limit") from None
    schema = inspect_csv_columns(source, delimiter=delimiter, encoding=encoding)
    validate_csv_column_roles(schema, id_column=id_column, class_column=class_column,
                              feature_columns=feature_columns)
    dataset_id = str(uuid4())
    ref = ArtifactReference(scope, str(uuid4()), ArtifactKind.DATASET,
                            "text/csv", dataset_id=dataset_id)
    try:
        with source.open("rb") as reader:
            repository.save_artifact(scope, ref, reader)
    except OSError:
        raise ValueError("Unable to persist CSV dataset") from None
    # Optional display metadata is a separate immutable manifest; the CSV artifact
    # remains valid if metadata publication fails. Never store client paths.
    if original_name is not None:
        from Storage.CSVDatasetCatalog import clean_csv_display_name
        name = clean_csv_display_name(original_name)
        meta = {"version": 1, "dataset_id": dataset_id, "filename": name,
                "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        manifest = ArtifactReference(scope, str(uuid4()), ArtifactKind.MANIFEST,
                                     "application/json", dataset_id=dataset_id)
        repository.save_artifact(scope, manifest,
                                 BytesIO(json.dumps(meta).encode("utf-8")))
    return RegisteredCSVDataset(dataset_id, ref, schema.rows, len(schema.columns))
