"""Backend-neutral, web-addressable storage contracts (revision 186).

This module defines identifiers and boundaries, not authentication or storage.
No files, SQL objects, or HTTP routes are created here.
"""

from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO, Mapping, Protocol
from uuid import UUID


class StorageKind(str, Enum):
    SQL_SERVER = "sql_server"
    FILES = "files"


class ArtifactKind(str, Enum):
    DATASET = "dataset"
    PREDICTIONS = "predictions"
    SPOT_CHECK = "spot_check"
    METRICS = "metrics"
    MODEL = "model"
    MANIFEST = "manifest"


def _opaque_id(value: str, label: str) -> str:
    """Require URL-safe opaque UUIDs; never accept filesystem paths as IDs."""
    try:
        parsed = UUID(value)
        if str(parsed) != value or parsed.version != 4:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"Invalid {label} identifier") from None
    return value


@dataclass(frozen=True)
class StorageScope:
    """Trusted tenant/project context, supplied by an authorization boundary."""
    tenant_id: str
    project_id: str

    def __post_init__(self):
        _opaque_id(self.tenant_id, "tenant")
        _opaque_id(self.project_id, "project")


@dataclass(frozen=True)
class ArtifactReference:
    """Public metadata: never embed absolute paths, credentials or row contents."""
    scope: StorageScope
    artifact_id: str
    kind: ArtifactKind
    content_type: str
    run_id: str | None = None
    dataset_id: str | None = None

    def __post_init__(self):
        _opaque_id(self.artifact_id, "artifact")
        if not isinstance(self.kind, ArtifactKind):
            raise ValueError("Invalid artifact kind")
        if not isinstance(self.content_type, str) or self.content_type not in (
            "text/csv", "application/json", "application/octet-stream"
        ):
            raise ValueError("Unsupported artifact content type")
        if self.run_id is not None:
            _opaque_id(self.run_id, "run")
        if self.dataset_id is not None:
            _opaque_id(self.dataset_id, "dataset")


class ArtifactRepository(Protocol):
    """Implementations must enforce scope and permissions on every operation.

    The scope is not proof of identity. The future service must authenticate the
    caller and authorize its scope before any repository operation. File paths
    and SQL table names remain private to implementations.
    """

    def list_artifacts(self, scope: StorageScope, *, run_id: str | None = None) -> tuple[ArtifactReference, ...]: ...

    def get_reference(self, scope: StorageScope, artifact_id: str) -> ArtifactReference: ...

    def open_artifact(self, scope: StorageScope, artifact_id: str) -> BinaryIO: ...

    def save_artifact(self, scope: StorageScope, reference: ArtifactReference, content: BinaryIO) -> None: ...


class RunRepository(Protocol):
    """List/retrieve runs by logical identifier, never by server path."""

    def list_runs(self, scope: StorageScope) -> tuple[str, ...]: ...

    def get_run_manifest(self, scope: StorageScope, run_id: str) -> Mapping[str, object]: ...
