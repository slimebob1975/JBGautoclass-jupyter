"""Owner-bound project service for the HTTP API (revision 286a).

Thin layer over ProjectRepository (285i): turns records into JSON responses and
repository errors into the API's error rules. A project's source type is fixed
at creation; its stable ID (not its name) identifies it everywhere, so it can
be renamed freely. Deletion is deliberately absent until a cascade policy
exists (BACKLOG).
"""
from __future__ import annotations

from typing import Any

from JBGApplicationContracts import DataSourceKind
from JBGProjects import DuplicateProjectName, Project, ProjectRepository


class ProjectUnavailable(LookupError):
    """Unknown project, or a project of another user: indistinguishable."""


class ProjectConflict(ValueError):
    """The owner already has a project with that name."""


def project_response(project: Project) -> dict[str, Any]:
    # The owner is never returned: it is the caller, and a principal key is
    # an internal identifier.
    return {"project_id": project.project_id, "name": project.name,
            "description": project.description, "source": project.source.value,
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat()}


class ProjectService:
    def __init__(self, repository: ProjectRepository) -> None:
        if not isinstance(repository, ProjectRepository):
            raise TypeError("ProjectRepository required")
        self._repository = repository

    def list(self, owner: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return [project_response(project)
                for project in self._repository.list_for_owner(owner, limit=limit)]

    def create(self, owner: str, *, name: str, source: str,
               description: str = "") -> dict[str, Any]:
        try:
            kind = DataSourceKind(source)
        except ValueError:
            raise ValueError("source must be 'sql' or 'csv'") from None
        try:
            project = self._repository.create(owner, name, kind, description)
        except DuplicateProjectName:
            raise ProjectConflict("project name already exists") from None
        return project_response(project)

    def get(self, owner: str, project_id: str) -> dict[str, Any]:
        return project_response(self.require(owner, project_id))

    def require(self, owner: str, project_id: str) -> Project:
        """The owner's project, for other services (datasets, configurations)."""
        if not isinstance(project_id, str) or not project_id:
            raise ProjectUnavailable("project unavailable")
        try:
            return self._repository.get_owned(owner, project_id)
        except KeyError:
            raise ProjectUnavailable("project unavailable") from None

    def update(self, owner: str, project_id: str, *, name: str | None = None,
               description: str | None = None) -> dict[str, Any]:
        self.require(owner, project_id)
        try:
            project = self._repository.update(owner, project_id, name=name,
                                              description=description)
        except DuplicateProjectName:
            raise ProjectConflict("project name already exists") from None
        except KeyError:
            raise ProjectUnavailable("project unavailable") from None
        return project_response(project)
