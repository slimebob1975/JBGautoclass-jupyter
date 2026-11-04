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
    if original_name:
        # Display metadata only, as in Voila's registration; never a client path.
        from Storage.CSVDatasetCatalog import clean_csv_display_name
        meta = {"version": 1, "dataset_id": dataset_id,
                "filename": clean_csv_display_name(original_name),
                "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        manifest = ArtifactReference(scope, str(uuid4()), ArtifactKind.MANIFEST,
                                     "application/json", dataset_id=dataset_id)
        repository.save_artifact(scope, manifest, BytesIO(json.dumps(meta).encode("utf-8")))
    return UploadedCSV(reference.artifact_id, schema.rows, tuple(schema.columns),
                       delimiter, encoding)
