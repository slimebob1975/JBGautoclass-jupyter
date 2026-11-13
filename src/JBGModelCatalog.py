"""Project-owned model artifact catalog (285l); metadata only, no model I/O.

Model binaries remain with the existing persistence layer. An opaque artifact key
is registered only after artifact persistence succeeds. Legacy artifacts are
not imported automatically. 299e: ``delete`` removes a catalog entry only; the
model files are removed by JBGDeletion.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import (Column, DateTime, ForeignKeyConstraint, Index, MetaData,
                        String, Table, UniqueConstraint, create_engine, delete, insert,
                        select, update, inspect)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from JBGApplicationContracts import DataSourceKind, RunStatus


class ModelState(str, Enum):
    REGISTERED = "registered"
    AVAILABLE = "available"
    MISSING = "missing"
    RETIRED = "retired"


def _text(value, field, limit=200):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= limit:
        raise ValueError(f"{field} must contain 1-{limit} characters")
    return value.strip()


def _uuid(value, field):
    try:
        if not isinstance(value, str) or str(UUID(value)) != value.lower():
            raise ValueError()
    except (ValueError, AttributeError, TypeError) as error:
        raise ValueError(f"{field} must be a UUID") from error
    return value


def _utc(value):
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    project_id: str
    dataset_id: str | None
    run_id: str | None
    name: str
    source: DataSourceKind
    artifact_key: str
    compatibility_key: str | None
    state: ModelState
    created_at: datetime
    updated_at: datetime


class ModelCatalog:
    """Record model provenance with project/owner and run/dataset consistency."""

    def __init__(self, engine: Engine | str, *, schema: str | None = None):
        if isinstance(engine, str):
            engine = create_engine(engine, future=True)
        if not isinstance(engine, Engine):
            raise TypeError("SQLAlchemy Engine required")
        self._engine = engine
        self._metadata = MetaData(schema=schema)
        self._projects = Table(
            "jbg_projects", self._metadata,
            Column("project_id", String(36), primary_key=True),
            Column("owner", String(256)), Column("source", String(8)))
        self._datasets = Table(
            "jbg_datasets", self._metadata,
            Column("dataset_id", String(36), primary_key=True),
            Column("project_id", String(36)), Column("source", String(8)))
        self._runs = Table(
            "jbg_run_jobs", self._metadata,
            Column("run_id", String(36), primary_key=True),
            Column("owner", String(256)), Column("source", String(8)),
            Column("status", String(24)))
        self._models = Table(
            "jbg_models", self._metadata,
            Column("model_id", String(36), primary_key=True),
            Column("project_id", String(36), nullable=False),
            Column("dataset_id", String(36)), Column("run_id", String(36)),
            Column("name", String(200), nullable=False),
            Column("name_key", String(200), nullable=False),
            Column("source", String(8), nullable=False),
            Column("artifact_key", String(128), nullable=False),
            Column("compatibility_key", String(128)),
            Column("state", String(16), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            ForeignKeyConstraint(["project_id"], ["jbg_projects.project_id"]),
            ForeignKeyConstraint(["dataset_id"], ["jbg_datasets.dataset_id"]),
            ForeignKeyConstraint(["run_id"], ["jbg_run_jobs.run_id"]),
            UniqueConstraint("project_id", "name_key", name="uq_jbg_model_project_name"),
            UniqueConstraint("project_id", "artifact_key", name="uq_jbg_model_project_artifact"),
            Index("ix_jbg_models_project_updated", "project_id", "updated_at"),
        )

    def create_tables(self):
        inspector = inspect(self._engine)
        for table in ("jbg_projects", "jbg_datasets", "jbg_run_jobs"):
            if not inspector.has_table(table, schema=self._metadata.schema):
                raise RuntimeError(f"initialize {table} before model catalog")
        self._metadata.create_all(self._engine, tables=[self._models])

    def close(self):
        self._engine.dispose()

    def _owned_project(self, conn, owner, project_id):
        row = conn.execute(select(self._projects.c.source).where(
            self._projects.c.project_id == project_id,
            self._projects.c.owner == owner)).first()
        if row is None:
            raise KeyError("project unavailable")
        return DataSourceKind(row.source)

    @staticmethod
    def _record(row):
        return ModelEntry(row.model_id, row.project_id, row.dataset_id,
                          row.run_id, row.name, DataSourceKind(row.source),
                          row.artifact_key, row.compatibility_key,
                          ModelState(row.state), _utc(row.created_at),
                          _utc(row.updated_at))

    def register(self, owner, project_id, name, artifact_key, *, dataset_id=None,
                 run_id=None, compatibility_key=None):
        owner, name = _text(owner, "owner", 256), _text(name, "model name")
        project_id = _uuid(project_id, "project ID")
        artifact_key = _uuid(artifact_key, "artifact key")
        if dataset_id is not None:
            dataset_id = _uuid(dataset_id, "dataset ID")
        if run_id is not None:
            run_id = _uuid(run_id, "run ID")
        if compatibility_key is not None:
            compatibility_key = _text(compatibility_key, "compatibility key", 128)
        now, model_id = datetime.now(timezone.utc), str(uuid4())
        with self._engine.begin() as conn:
            source = self._owned_project(conn, owner, project_id)
            if dataset_id is not None:
                row = conn.execute(select(self._datasets).where(
                    self._datasets.c.dataset_id == dataset_id,
                    self._datasets.c.project_id == project_id)).first()
                if row is None or row.source != source.value:
                    raise ValueError("dataset is not part of this project/source")
            if run_id is not None:
                row = conn.execute(select(self._runs).where(
                    self._runs.c.run_id == run_id,
                    self._runs.c.owner == owner)).first()
                if row is None or row.source != source.value or row.status != RunStatus.SUCCEEDED.value:
                    raise ValueError("run must be successful and owned by the project owner/source")
            try:
                conn.execute(insert(self._models).values(
                    model_id=model_id, project_id=project_id,
                    dataset_id=dataset_id, run_id=run_id, name=name,
                    name_key=name.casefold(), source=source.value,
                    artifact_key=artifact_key, compatibility_key=compatibility_key,
                    state=ModelState.REGISTERED.value,
                    created_at=now, updated_at=now))
            except IntegrityError as error:
                raise ValueError("model name or artifact already registered") from error
        return self.get_owned(owner, project_id, model_id)

    def get_owned(self, owner, project_id, model_id):
        owner = _text(owner, "owner", 256)
        with self._engine.connect() as conn:
            self._owned_project(conn, owner, project_id)
            row = conn.execute(select(self._models).where(
                self._models.c.project_id == project_id,
                self._models.c.model_id == model_id)).first()
        if row is None:
            raise KeyError("model unavailable")
        return self._record(row)

    def list_for_project(self, owner, project_id, *, limit=100):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be 1-1000")
        with self._engine.connect() as conn:
            self._owned_project(conn, _text(owner, "owner", 256), project_id)
            rows = conn.execute(select(self._models).where(
                self._models.c.project_id == project_id).order_by(
                self._models.c.updated_at.desc(), self._models.c.model_id).limit(limit)).all()
        return tuple(self._record(row) for row in rows)

    def set_state(self, owner, project_id, model_id, state: ModelState):
        """Only metadata changes; caller must verify artifacts separately."""
        if not isinstance(state, ModelState):
            raise TypeError("state must be ModelState")
        with self._engine.begin() as conn:
            self._owned_project(conn, _text(owner, "owner", 256), project_id)
            changed = conn.execute(update(self._models).where(
                self._models.c.project_id == project_id,
                self._models.c.model_id == model_id).values(
                state=state.value, updated_at=datetime.now(timezone.utc))).rowcount
            if not changed:
                raise KeyError("model unavailable")
        return self.get_owned(owner, project_id, model_id)

    def delete(self, owner, project_id, model_id):
        """299e: remove one catalog entry of the owner's project (metadata only)."""
        with self._engine.begin() as conn:
            self._owned_project(conn, _text(owner, "owner", 256), project_id)
            removed = conn.execute(delete(self._models).where(
                self._models.c.project_id == project_id,
                self._models.c.model_id == model_id)).rowcount
        if not removed:
            raise KeyError("model unavailable")
