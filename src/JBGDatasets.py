"""Project-scoped dataset registrations (285k), without source I/O.

SQL locators are database catalog/table names; CSV locators are opaque IDs from
an existing CSV registration, never paths. This metadata repository does not
read rows, connect to SQL, verify source existence, or delete source artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import (Boolean, Column, DateTime, ForeignKeyConstraint, Index,
                        MetaData, String, Table, UniqueConstraint, create_engine,
                        insert, select, update)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from JBGApplicationContracts import DataSourceKind


class DatasetRole(str, Enum):
    TRAINING = "training"
    PREDICTION = "prediction"
    MIXED = "mixed"


class DatasetValidationState(str, Enum):
    UNVERIFIED = "unverified"
    VALID = "valid"
    INVALID = "invalid"


class DuplicateDatasetName(ValueError):
    """Dataset names are unique per project, ignoring letter case."""


def _text(value, name, size=200):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= size:
        raise ValueError(f"{name} must contain 1-{size} characters")
    return value.strip()


def _utc(value):
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _locator(source, *, catalog=None, table=None, registration_id=None):
    if source is DataSourceKind.SQL:
        if registration_id is not None:
            raise ValueError("SQL dataset cannot reference a CSV registration")
        catalog, table = _text(catalog, "SQL catalog"), _text(table, "SQL table")
        if catalog.casefold() in {"n/a", "none", "null"} or table.casefold() in {"n/a", "none", "null"}:
            raise ValueError("SQL locator must not be a placeholder")
        return catalog, table, None
    if source is DataSourceKind.CSV:
        if catalog is not None or table is not None:
            raise ValueError("CSV dataset cannot reference a SQL table")
        value = _text(registration_id, "CSV registration ID", 64)
        try:
            parsed = UUID(value)
            if parsed.version != 4:
                raise ValueError("not a UUIDv4")
        except (ValueError, AttributeError) as error:
            raise ValueError("CSV registration ID must be an opaque UUIDv4") from error
        return None, None, value
    raise TypeError("unknown data source kind")


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    project_id: str
    name: str
    source: DataSourceKind
    role: DatasetRole
    catalog: str | None
    table: str | None
    registration_id: str | None
    active_training: bool
    validation_state: DatasetValidationState
    created_at: datetime
    updated_at: datetime


class DatasetRepository:
    """Registration metadata only. Ownership checked on every operation.

    Exactly zero or one dataset per project may be active for training; a
    prediction-only dataset cannot be activated. Switching is atomic.
    """

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
            Column("owner", String(256), nullable=False),
            Column("source", String(8), nullable=False),
        )
        self._datasets = Table(
            "jbg_datasets", self._metadata,
            Column("dataset_id", String(36), primary_key=True),
            Column("project_id", String(36), nullable=False),
            Column("name", String(200), nullable=False),
            Column("name_key", String(200), nullable=False),
            Column("source", String(8), nullable=False),
            Column("role", String(12), nullable=False),
            Column("catalog", String(200)),
            Column("table_name", String(200)),
            Column("registration_id", String(64)),
            Column("active_training", Boolean, nullable=False, default=False),
            Column("validation_state", String(16), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            ForeignKeyConstraint(["project_id"], ["jbg_projects.project_id"]),
            UniqueConstraint("project_id", "name_key", name="uq_jbg_dataset_project_name"),
            Index("ix_jbg_dataset_project", "project_id", "updated_at"),
        )
        # Database-level safeguard against concurrent activation requests.
        Index("uq_jbg_active_training_project", self._datasets.c.project_id,
              unique=True,
              sqlite_where=self._datasets.c.active_training.is_(True),
              mssql_where=self._datasets.c.active_training == True)  # noqa: E712

    def create_tables(self):
        from sqlalchemy import inspect
        if not inspect(self._engine).has_table("jbg_projects", schema=self._metadata.schema):
            raise RuntimeError("initialize ProjectRepository before datasets")
        self._metadata.create_all(self._engine, tables=[self._datasets])

    def close(self):
        self._engine.dispose()

    def _owned_source(self, connection, owner, project_id):
        row = connection.execute(select(self._projects.c.source).where(
            self._projects.c.owner == owner,
            self._projects.c.project_id == project_id)).first()
        if row is None:
            raise KeyError("project unavailable")
        return DataSourceKind(row.source)

    @staticmethod
    def _record(row):
        return Dataset(row.dataset_id, row.project_id, row.name,
                       DataSourceKind(row.source), DatasetRole(row.role),
                       row.catalog, row.table_name, row.registration_id,
                       bool(row.active_training), DatasetValidationState(row.validation_state),
                       _utc(row.created_at), _utc(row.updated_at))

    def _owned_row(self, connection, owner, project_id, dataset_id):
        self._owned_source(connection, owner, project_id)
        row = connection.execute(select(self._datasets).where(
            self._datasets.c.project_id == project_id,
            self._datasets.c.dataset_id == dataset_id)).first()
        if row is None:
            raise KeyError("dataset unavailable")
        return row

    def create(self, owner, project_id, name, role: DatasetRole, *,
               catalog=None, table=None, registration_id=None):
        owner, name = _text(owner, "owner", 256), _text(name, "dataset name")
        if not isinstance(role, DatasetRole):
            raise TypeError("role must be DatasetRole")
        now, dataset_id = datetime.now(timezone.utc), str(uuid4())
        try:
            with self._engine.begin() as connection:
                source = self._owned_source(connection, owner, project_id)
                catalog, table, registration_id = _locator(
                    source, catalog=catalog, table=table, registration_id=registration_id)
                connection.execute(insert(self._datasets).values(
                    dataset_id=dataset_id, project_id=project_id, name=name,
                    name_key=name.casefold(), source=source.value, role=role.value,
                    catalog=catalog, table_name=table, registration_id=registration_id,
                    active_training=False, validation_state=DatasetValidationState.UNVERIFIED.value,
                    created_at=now, updated_at=now))
        except IntegrityError as error:
            raise DuplicateDatasetName("dataset name already exists") from error
        return self.get_owned(owner, project_id, dataset_id)

    def get_owned(self, owner, project_id, dataset_id):
        owner = _text(owner, "owner", 256)
        with self._engine.connect() as connection:
            return self._record(self._owned_row(connection, owner, project_id, dataset_id))

    def list_for_project(self, owner, project_id, *, limit=100):
        owner = _text(owner, "owner", 256)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be 1-1000")
        with self._engine.connect() as connection:
            self._owned_source(connection, owner, project_id)
            rows = connection.execute(select(self._datasets).where(
                self._datasets.c.project_id == project_id).order_by(
                self._datasets.c.updated_at.desc(), self._datasets.c.dataset_id).limit(limit)).all()
        return tuple(self._record(row) for row in rows)

    def activate_training(self, owner, project_id, dataset_id):
        owner = _text(owner, "owner", 256)
        with self._engine.begin() as connection:
            row = self._owned_row(connection, owner, project_id, dataset_id)
            if DatasetRole(row.role) is DatasetRole.PREDICTION:
                raise ValueError("prediction-only dataset cannot be active for training")
            now = datetime.now(timezone.utc)
            connection.execute(update(self._datasets).where(
                self._datasets.c.project_id == project_id).values(active_training=False))
            connection.execute(update(self._datasets).where(
                self._datasets.c.dataset_id == dataset_id).values(
                active_training=True, updated_at=now))
        return self.get_owned(owner, project_id, dataset_id)

    def set_validation(self, owner, project_id, dataset_id, state: DatasetValidationState):
        """Record a result obtained by a separate validation service.

        Modifying or re-registering source metadata must not inherit this state.
        """
        if not isinstance(state, DatasetValidationState):
            raise TypeError("state must be DatasetValidationState")
        owner = _text(owner, "owner", 256)
        with self._engine.begin() as connection:
            self._owned_row(connection, owner, project_id, dataset_id)
            connection.execute(update(self._datasets).where(
                self._datasets.c.dataset_id == dataset_id).values(
                validation_state=state.value, updated_at=datetime.now(timezone.utc)))
        return self.get_owned(owner, project_id, dataset_id)
