"""Persistent, owner-scoped project metadata (285i).

Projects own future dataset/configuration records, but this repository deliberately
never deletes source data, runs, models, or other artifacts. Cascading deletion and
configuration persistence will be introduced only with explicit services.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (Column, DateTime, Index, MetaData, String, Table,
                        Text, UniqueConstraint, create_engine, insert, select, update)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from JBGApplicationContracts import DataSourceKind


class DuplicateProjectName(ValueError):
    """The owner already has a project with this name (case insensitive)."""


def _now():
    return datetime.now(timezone.utc)


def _utc(value):
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _name(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 200:
        raise ValueError("project name must contain 1-200 characters")
    return value.strip()


def _owner(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("project owner required")
    return value.strip()


def _description(value):
    if not isinstance(value, str) or len(value) > 4000:
        raise ValueError("project description must be text of at most 4000 characters")
    return value


@dataclass(frozen=True)
class Project:
    project_id: str
    owner: str
    name: str
    description: str
    source: DataSourceKind
    created_at: datetime
    updated_at: datetime


class ProjectRepository:
    """Owner-scoped CRUD metadata except deletion, which needs cascade policy.

    Use the same metadata database and schema as durable jobs, not the data-source
    SQL Server. All lookups are owner-filtered and unknown/foreign IDs are identical.
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
            Column("name", String(200), nullable=False),
            Column("name_key", String(200), nullable=False),
            Column("description", Text, nullable=False),
            Column("source", String(8), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("owner", "name_key", name="uq_jbg_project_owner_name"),
            Index("ix_jbg_projects_owner_updated", "owner", "updated_at"),
        )

    def create_tables(self):
        self._metadata.create_all(self._engine)

    def close(self):
        self._engine.dispose()

    @staticmethod
    def _record(row):
        return Project(row.project_id, row.owner, row.name, row.description,
                       DataSourceKind(row.source), _utc(row.created_at),
                       _utc(row.updated_at))

    def create(self, owner: str, name: str, source: DataSourceKind,
               description: str = "") -> Project:
        owner, name, description = _owner(owner), _name(name), _description(description)
        if not isinstance(source, DataSourceKind):
            raise TypeError("source must be DataSourceKind")
        project_id, now = str(uuid4()), _now()
        try:
            with self._engine.begin() as connection:
                connection.execute(insert(self._projects).values(
                    project_id=project_id, owner=owner, name=name,
                    name_key=name.casefold(), description=description,
                    source=source.value, created_at=now, updated_at=now))
        except IntegrityError as error:
            raise DuplicateProjectName("project name already exists") from error
        return self.get_owned(owner, project_id)

    def get_owned(self, owner: str, project_id: str) -> Project:
        owner = _owner(owner)
        with self._engine.connect() as connection:
            row = connection.execute(select(self._projects).where(
                self._projects.c.owner == owner,
                self._projects.c.project_id == project_id)).first()
        if row is None:
            raise KeyError("project unavailable")
        return self._record(row)

    def list_for_owner(self, owner: str, *, limit: int = 100) -> tuple[Project, ...]:
        owner = _owner(owner)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise ValueError("limit must be 1-1000")
        with self._engine.connect() as connection:
            rows = connection.execute(select(self._projects).where(
                self._projects.c.owner == owner).order_by(
                self._projects.c.updated_at.desc(), self._projects.c.project_id).limit(limit)).all()
        return tuple(self._record(row) for row in rows)

    def update(self, owner: str, project_id: str, *, name: str | None = None,
               description: str | None = None) -> Project:
        owner = _owner(owner)
        if name is None and description is None:
            return self.get_owned(owner, project_id)
        values = {"updated_at": _now()}
        if name is not None:
            values.update(name=_name(name), name_key=_name(name).casefold())
        if description is not None:
            values["description"] = _description(description)
        try:
            with self._engine.begin() as connection:
                changed = connection.execute(update(self._projects).where(
                    self._projects.c.owner == owner,
                    self._projects.c.project_id == project_id).values(**values)).rowcount
        except IntegrityError as error:
            raise DuplicateProjectName("project name already exists") from error
        if not changed:
            raise KeyError("project unavailable")
        return self.get_owned(owner, project_id)
