"""GUI-independent application boundary contracts (revision 280v).

These value objects do not start jobs or access a data source. The application
adapter introduced in later revisions will translate them into the existing
SQL/CSV config and AutomaticClassifier/TaskRunner calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DataSourceKind(str, Enum):
    SQL = "sql"
    CSV = "csv"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class DatasetReference:
    """Opaque identity, always scoped to the source that owns the dataset.

    ``key`` is deliberately not assumed to be a filesystem path, SQL table,
    registration ID or artifact ID. Adapters interpret it in their own scope.
    """

    source: DataSourceKind
    key: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, DataSourceKind):
            raise TypeError("source must be a DataSourceKind")
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("dataset key must be a non-empty string")


@dataclass(frozen=True)
class RunReference:
    """Opaque identifier returned by a future application job adapter."""

    key: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("run key must be a non-empty string")


@dataclass(frozen=True)
class RunSnapshot:
    """Transport-neutral observation of a job, not an ML result payload."""

    run: RunReference
    status: RunStatus
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunReference):
            raise TypeError("run must be a RunReference")
        if not isinstance(self.status, RunStatus):
            raise TypeError("status must be a RunStatus")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("message must be text or None")
