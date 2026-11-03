"""Durable job records and progress events for application runs (revision 285a).

SQLAlchemy Core over a local SQLite file by default, or the JBG metadata
database on SQL Server (configured in a later revision). The store keeps no
secrets: a job specification must be JSON without passwords, SQL credentials
travel to a worker only through its process pipe, and a finished job keeps
only the projected, JSON-native result.

Status changes are guarded single-statement updates, so several processes
(web server, dispatcher, workers) can share one database safely.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData,
                        String, Table, Text, create_engine, delete, func, insert, select,
                        update)
from sqlalchemy.engine import Engine

from JBGApplicationContracts import DataSourceKind, RunReference, RunStatus
from JBGRunProgress import ProgressEvent

TABLE_PREFIX = "jbg_run_"
TERMINAL_STATES = frozenset({RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED,
                             RunStatus.INTERRUPTED})
_FORBIDDEN_SPEC_KEYS = frozenset({"sql_password", "password"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime | None) -> datetime | None:
    # SQLite returns naive datetimes; values are always written as UTC.
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _metadata(schema: str | None) -> tuple[MetaData, Table, Table]:
    metadata = MetaData(schema=schema)
    jobs = Table(
        f"{TABLE_PREFIX}jobs", metadata,
        Column("run_id", String(36), primary_key=True),
        Column("owner", String(256), nullable=False, index=True),
        Column("source", String(8), nullable=False),
        Column("status", String(24), nullable=False, index=True),
        Column("spec", Text, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("started_at", DateTime(timezone=True)),
        Column("finished_at", DateTime(timezone=True)),
        Column("heartbeat_at", DateTime(timezone=True)),
        Column("worker_id", String(64)),
        Column("cancel_requested", Boolean, nullable=False, default=False),
        Column("result", Text),
        Column("error_type", String(128)),
    )
    events = Table(
        f"{TABLE_PREFIX}events", metadata,
        Column("event_id", Integer, primary_key=True, autoincrement=True),
        Column("run_id", String(36),
               ForeignKey(jobs.c.run_id, ondelete="CASCADE"), nullable=False, index=True),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("message", Text),
        Column("percent", Float),
    )
    return metadata, jobs, events


@dataclass(frozen=True)
class JobRecord:
    run: RunReference
    owner: str
    source: DataSourceKind
    status: RunStatus
    spec: dict
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    heartbeat_at: datetime | None
    worker_id: str | None
    cancel_requested: bool
    result: dict | None
    error_type: str | None


def _check_spec(value: Any, path: str = "spec") -> None:
    """Reject anything that is not plain JSON, and any non-empty password field."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: keys must be strings")
            if key.lower() in _FORBIDDEN_SPEC_KEYS and item not in ("", None):
                raise ValueError("job specification must not contain a password")
            _check_spec(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_spec(item, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{path}: not JSON-native")


class RunJobStore:
    """Durable jobs and bounded progress history, shared between processes."""

    def __init__(self, engine: Engine | str, *, schema: str | None = None,
                 max_events: int = 200) -> None:
        if isinstance(engine, str):
            engine = create_engine(engine, future=True)
        if not isinstance(engine, Engine):
            raise TypeError("SQLAlchemy Engine or database URL required")
        if not isinstance(max_events, int) or isinstance(max_events, bool) or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        self._engine = engine
        self._max_events = max_events
        self._metadata, self._jobs, self._events = _metadata(schema)

    @classmethod
    def sqlite(cls, path, **options) -> "RunJobStore":
        """Local file store, used when no SQL Server metadata database is configured."""
        engine = create_engine(f"sqlite:///{path}", future=True,
                               connect_args={"timeout": 30, "check_same_thread": False})
        return cls(engine, **options)

    def close(self) -> None:
        """Release pooled database connections (on Windows, an open SQLite file
        cannot be deleted). The store reconnects if it is used again."""
        self._engine.dispose()

    def create_tables(self) -> None:
        """Create missing tables. On SQL Server, run this with a deployment account."""
        self._metadata.create_all(self._engine)

    # --- Creating and reading ------------------------------------------------------

    def create(self, owner: str, source: DataSourceKind, spec: dict) -> RunReference:
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("owner required")
        if not isinstance(source, DataSourceKind):
            raise TypeError("source must be a DataSourceKind")
        if not isinstance(spec, dict):
            raise TypeError("spec must be a dict")
        _check_spec(spec)
        run = RunReference(str(uuid4()))
        with self._engine.begin() as connection:
            connection.execute(insert(self._jobs).values(
                run_id=run.key, owner=owner, source=source.value,
                status=RunStatus.PENDING.value, spec=json.dumps(spec, sort_keys=True),
                created_at=_now(), cancel_requested=False))
        return run

    def _row_to_record(self, row) -> JobRecord:
        return JobRecord(
            run=RunReference(row.run_id), owner=row.owner, source=DataSourceKind(row.source),
            status=RunStatus(row.status), spec=json.loads(row.spec),
            created_at=_utc(row.created_at), started_at=_utc(row.started_at),
            finished_at=_utc(row.finished_at), heartbeat_at=_utc(row.heartbeat_at),
            worker_id=row.worker_id, cancel_requested=bool(row.cancel_requested),
            result=json.loads(row.result) if row.result is not None else None,
            error_type=row.error_type)

    def get(self, run: RunReference) -> JobRecord:
        """Trusted, server-side read. Raises KeyError for an unknown run."""
        if not isinstance(run, RunReference):
            raise TypeError("expected RunReference")
        with self._engine.connect() as connection:
            row = connection.execute(
                select(self._jobs).where(self._jobs.c.run_id == run.key)).first()
        if row is None:
            raise KeyError("unknown run reference")
        return self._row_to_record(row)

    def get_owned(self, owner: str, run: RunReference) -> JobRecord:
        """Owner-checked read: unknown and foreign runs both raise KeyError."""
        record = self.get(run)
        if record.owner != owner:
            raise KeyError("unknown run reference")
        return record

    def list_for_owner(self, owner: str, limit: int = 50) -> tuple[JobRecord, ...]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                select(self._jobs).where(self._jobs.c.owner == owner)
                .order_by(self._jobs.c.created_at.desc()).limit(limit)).all()
        return tuple(self._row_to_record(row) for row in rows)

    def pending(self, limit: int = 10) -> tuple[JobRecord, ...]:
        """Oldest pending jobs first, for the dispatcher."""
        with self._engine.connect() as connection:
            rows = connection.execute(
                select(self._jobs).where(self._jobs.c.status == RunStatus.PENDING.value)
                .order_by(self._jobs.c.created_at).limit(limit)).all()
        return tuple(self._row_to_record(row) for row in rows)

    # --- Lifecycle -----------------------------------------------------------------

    def claim(self, run: RunReference, worker_id: str) -> bool:
        """Atomically move a pending job to running for ``worker_id``; True if won."""
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id required")
        now = _now()
        with self._engine.begin() as connection:
            changed = connection.execute(
                update(self._jobs)
                .where(self._jobs.c.run_id == run.key,
                       self._jobs.c.status == RunStatus.PENDING.value)
                .values(status=RunStatus.RUNNING.value, worker_id=worker_id,
                        started_at=now, heartbeat_at=now)).rowcount
        return changed == 1

    def heartbeat(self, run: RunReference, worker_id: str) -> bool:
        """Refresh the heartbeat; False if this worker no longer runs the job."""
        with self._engine.begin() as connection:
            changed = connection.execute(
                update(self._jobs)
                .where(self._jobs.c.run_id == run.key,
                       self._jobs.c.status == RunStatus.RUNNING.value,
                       self._jobs.c.worker_id == worker_id)
                .values(heartbeat_at=_now())).rowcount
        return changed == 1

    def cancel_requested(self, run: RunReference) -> bool:
        with self._engine.connect() as connection:
            value = connection.execute(
                select(self._jobs.c.cancel_requested)
                .where(self._jobs.c.run_id == run.key)).scalar_one_or_none()
        if value is None:
            raise KeyError("unknown run reference")
        return bool(value)

    def request_cancel(self, run: RunReference) -> RunStatus:
        """Pending jobs are cancelled at once; running jobs get a cooperative flag.

        Returns the job's status after the request; finished jobs are unchanged.
        """
        now = _now()
        with self._engine.begin() as connection:
            connection.execute(
                update(self._jobs)
                .where(self._jobs.c.run_id == run.key,
                       self._jobs.c.status == RunStatus.PENDING.value)
                .values(status=RunStatus.CANCELLED.value, cancel_requested=True,
                        finished_at=now))
            connection.execute(
                update(self._jobs)
                .where(self._jobs.c.run_id == run.key,
                       self._jobs.c.status == RunStatus.RUNNING.value)
                .values(cancel_requested=True))
        return self.get(run).status

    def finish(self, run: RunReference, worker_id: str, status: RunStatus, *,
               result: dict | None = None, error_type: str | None = None) -> bool:
        """Record the outcome; only the claiming worker of a running job may finish it."""
        if status not in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED):
            raise ValueError("finish status must be succeeded, failed or cancelled")
        if result is not None:
            if not isinstance(result, dict):
                raise TypeError("result must be a dict")
            _check_spec(result, "result")
        if error_type is not None and (not isinstance(error_type, str) or len(error_type) > 128):
            raise ValueError("error_type must be a short string")
        with self._engine.begin() as connection:
            changed = connection.execute(
                update(self._jobs)
                .where(self._jobs.c.run_id == run.key,
                       self._jobs.c.status == RunStatus.RUNNING.value,
                       self._jobs.c.worker_id == worker_id)
                .values(status=status.value, finished_at=_now(),
                        result=json.dumps(result, sort_keys=True) if result is not None else None,
                        error_type=error_type)).rowcount
        return changed == 1

    def mark_stale_interrupted(self, older_than: timedelta) -> tuple[RunReference, ...]:
        """Mark running jobs whose heartbeat is older than ``older_than`` as interrupted."""
        if not isinstance(older_than, timedelta) or older_than <= timedelta(0):
            raise ValueError("older_than must be a positive timedelta")
        limit = _now() - older_than
        with self._engine.begin() as connection:
            stale = [row.run_id for row in connection.execute(
                select(self._jobs.c.run_id)
                .where(self._jobs.c.status == RunStatus.RUNNING.value,
                       self._jobs.c.heartbeat_at < limit)).all()]
            marked = []
            for run_id in stale:
                changed = connection.execute(
                    update(self._jobs)
                    .where(self._jobs.c.run_id == run_id,
                           self._jobs.c.status == RunStatus.RUNNING.value,
                           self._jobs.c.heartbeat_at < limit)
                    .values(status=RunStatus.INTERRUPTED.value, finished_at=_now())).rowcount
                if changed == 1:
                    marked.append(RunReference(run_id))
        return tuple(marked)

    # --- Progress ------------------------------------------------------------------

    def record_progress(self, run: RunReference, message: str | None = None,
                        percent: float | None = None) -> ProgressEvent:
        if message is not None and not isinstance(message, str):
            raise TypeError("message must be text or None")
        if percent is not None:
            if isinstance(percent, bool) or not isinstance(percent, (int, float)):
                raise TypeError("percent must be numeric or None")
            if not 0 <= percent <= 1:
                raise ValueError("percent must be between 0 and 1")
        now = _now()
        with self._engine.begin() as connection:
            exists = connection.execute(
                select(self._jobs.c.run_id).where(self._jobs.c.run_id == run.key)).first()
            if exists is None:
                raise KeyError("unknown run reference")
            connection.execute(insert(self._events).values(
                run_id=run.key, created_at=now, message=message,
                percent=float(percent) if percent is not None else None))
            count = connection.execute(
                select(func.count()).select_from(self._events)
                .where(self._events.c.run_id == run.key)).scalar_one()
            if count > self._max_events:
                oldest = select(self._events.c.event_id).where(
                    self._events.c.run_id == run.key).order_by(
                    self._events.c.event_id).limit(count - self._max_events)
                ids = [row.event_id for row in connection.execute(oldest).all()]
                connection.execute(delete(self._events).where(self._events.c.event_id.in_(ids)))
        return ProgressEvent(run, now, message, percent)

    def progress(self, run: RunReference) -> tuple[ProgressEvent, ...]:
        self.get(run)  # unknown runs raise KeyError, as in RunProgressService
        with self._engine.connect() as connection:
            rows = connection.execute(
                select(self._events).where(self._events.c.run_id == run.key)
                .order_by(self._events.c.event_id)).all()
        return tuple(ProgressEvent(run, _utc(row.created_at), row.message, row.percent)
                     for row in rows)
