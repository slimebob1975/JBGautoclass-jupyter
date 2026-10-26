"""Planning-only, SQL-independent CSV run context; does not enable ML execution."""
from dataclasses import dataclass
from uuid import UUID
from Storage.ArtifactContract import ArtifactKind, StorageKind, StorageScope


@dataclass(frozen=True)
class CSVRunContext:
    scope: StorageScope
    dataset_id: str
    artifact_id: str
    id_column: str
    class_column: str
    feature_columns: tuple[str, ...]
    storage_kind: StorageKind = StorageKind.FILES


def prepare_csv_run(repository, scope, artifact_id, *, id_column, class_column, feature_columns):
    """Resolve an authorized-by-caller dataset and validate roles (no data access)."""
    if not isinstance(scope, StorageScope):
        raise ValueError("Invalid CSV run scope")
    ref = repository.get_reference(scope, artifact_id)
    if ref.kind != ArtifactKind.DATASET or ref.content_type != "text/csv" or ref.dataset_id is None:
        raise ValueError("Invalid CSV run dataset")
    roles = (id_column, class_column, *feature_columns)
    if (not isinstance(id_column, str) or not isinstance(class_column, str)
            or not isinstance(feature_columns, (tuple, list)) or not feature_columns
            or any(not isinstance(r, str) or not r.strip() for r in roles)
            or len(set(roles)) != len(roles)):
        raise ValueError("Invalid CSV run column roles")
    return CSVRunContext(scope, ref.dataset_id, ref.artifact_id,
                         id_column, class_column, tuple(feature_columns))
