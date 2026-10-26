"""Scoped, SQL-free retrieval of previously registered CSV dataset artifacts.

StorageScope must be supplied by a trusted caller; it is not authentication.
"""
from pathlib import Path
from datetime import datetime
import json
import re

from Storage.ArtifactContract import ArtifactKind, StorageScope


MAX_CSV_BYTES = 25 * 1024 * 1024


def list_csv_datasets(repository, scope: StorageScope):
    """Return opaque (dataset_id, artifact_id) pairs for this scope only."""
    if not isinstance(scope, StorageScope):
        raise ValueError("Invalid dataset scope")
    return tuple((ref.dataset_id, ref.artifact_id)
                 for ref in repository.list_artifacts(scope)
                 if ref.kind == ArtifactKind.DATASET and ref.dataset_id is not None
                 and ref.content_type == "text/csv")


def restore_csv_dataset(repository, scope: StorageScope, artifact_id: str, destination):
    """Copy a scoped dataset to a private staging file, bounded by size.

    The destination must be supplied by trusted application code, never users.
    Do not expose its path or the dataset's original bytes in error messages.
    """
    if not isinstance(scope, StorageScope):
        raise ValueError("Invalid dataset scope")
    ref = repository.get_reference(scope, artifact_id)
    if ref.kind != ArtifactKind.DATASET or ref.content_type != "text/csv" or not ref.dataset_id:
        raise ValueError("Artifact is not a registered CSV dataset")
    target = Path(destination)
    if target.is_symlink() or target.exists():
        raise ValueError("Unsafe dataset staging destination")
    try:
        with repository.open_artifact(scope, artifact_id) as source:
            with target.open("xb") as writer:
                total = 0
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_CSV_BYTES:
                        raise ValueError("Registered CSV exceeds size limit")
                    writer.write(chunk)
        return ref
    except Exception:
        target.unlink(missing_ok=True)
        raise


def clean_csv_display_name(value):
    """Accept only a modest, path-free CSV basename for the local catalog."""
    if not isinstance(value, str):
        raise ValueError("Invalid CSV display name")
    name = value.replace("\\", "/").split("/")[-1].strip()
    if (not name.lower().endswith(".csv") or len(name) > 120
            or not re.fullmatch(r"[\w .()\-]+", name, flags=re.UNICODE)
            or name in (".", "..")):
        raise ValueError("Invalid CSV display name")
    return name


def csv_dataset_display_name(repository, scope: StorageScope, dataset_id: str):
    """Return a safe basename for a scoped dataset; never a server path."""
    from Storage.ArtifactContract import ArtifactKind
    if not any(ref.dataset_id == dataset_id and ref.kind == ArtifactKind.DATASET
               and ref.content_type == "text/csv" for ref in repository.list_artifacts(scope)):
        raise FileNotFoundError("Registered CSV not found")
    for ref in repository.list_artifacts(scope):
        if ref.dataset_id != dataset_id or ref.kind != ArtifactKind.MANIFEST or ref.run_id is not None:
            continue
        try:
            with repository.open_artifact(scope, ref.artifact_id) as stream:
                record = json.loads(stream.read(2048))
            if isinstance(record, dict) and record.get("dataset_id") == dataset_id:
                return clean_csv_display_name(record.get("filename"))
        except (ValueError, TypeError, OSError, UnicodeError, KeyError):
            continue
    return "Registered CSV"


def list_csv_dataset_labels(repository, scope: StorageScope):
    """List (label, artifact_id), with legacy registrations handled safely."""
    entries = list_csv_datasets(repository, scope)
    metadata = {}
    for ref in repository.list_artifacts(scope):
        if ref.kind != ArtifactKind.MANIFEST or ref.content_type != "application/json" or not ref.dataset_id:
            continue
        try:
            with repository.open_artifact(scope, ref.artifact_id) as stream:
                raw = stream.read(2048)
            record = json.loads(raw)
            if not isinstance(record, dict) or record.get("dataset_id") != ref.dataset_id:
                continue
            filename = clean_csv_display_name(record.get("filename"))
            stamp = datetime.fromisoformat(record.get("uploaded_at"))
            if stamp.tzinfo is None:
                continue
            metadata[ref.dataset_id] = (filename, stamp.strftime("%Y-%m-%d %H:%M UTC"))
        except (ValueError, TypeError, OSError, UnicodeError, KeyError):
            continue
    return tuple((f"{metadata[id][0]} · {metadata[id][1]}" if id in metadata else "Dataset " + id, artifact)
                 for id, artifact in entries)


def erase_csv_dataset(repository, scope: StorageScope, artifact_id: str):
    """Permanently erase a scoped CSV dataset and its display manifests.

    Fail closed when other artifacts refer to this dataset (for example runs).
    Callers must request an explicit user confirmation and authorize scope.
    """
    if not isinstance(scope, StorageScope):
        raise ValueError("Invalid dataset scope")
    dataset = repository.get_reference(scope, artifact_id)
    if dataset.kind != ArtifactKind.DATASET or dataset.content_type != "text/csv" or not dataset.dataset_id:
        raise ValueError("Not a registered CSV dataset")
    refs = repository.list_artifacts(scope)
    related = [ref for ref in refs if ref.dataset_id == dataset.dataset_id and ref.artifact_id != artifact_id]
    # Only display manifests may be erased with the dataset. Preserve run data.
    if any(ref.kind != ArtifactKind.MANIFEST or ref.run_id is not None for ref in related):
        raise ValueError("Dataset has related results and cannot be erased")
    # Remove the dataset first; a partially completed erase is never selectable.
    repository.delete_artifact(scope, artifact_id)
    for ref in related:
        repository.delete_artifact(scope, ref.artifact_id)
