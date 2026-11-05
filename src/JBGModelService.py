"""Owner-bound model catalog reads for the HTTP API (revision 288c).

Lists the models that project runs have registered (285l catalog). Responses
never contain file paths or artifact keys; a model is identified by its
catalog ID and points to the run and dataset it came from.
"""
from __future__ import annotations

from typing import Any

from JBGModelCatalog import ModelCatalog, ModelEntry
from JBGProjectService import ProjectService


class ModelUnavailable(LookupError):
    """Unknown model, or a model of another user's project."""


def model_response(entry: ModelEntry) -> dict[str, Any]:
    return {"model_id": entry.model_id, "project_id": entry.project_id, "name": entry.name,
            "source": entry.source.value, "state": entry.state.value,
            "dataset_id": entry.dataset_id, "run_id": entry.run_id,
            "created_at": entry.created_at.isoformat(), "updated_at": entry.updated_at.isoformat()}


class ModelService:
    def __init__(self, catalog: ModelCatalog, projects: ProjectService) -> None:
        if not isinstance(catalog, ModelCatalog):
            raise TypeError("ModelCatalog required")
        if not isinstance(projects, ProjectService):
            raise TypeError("ProjectService required")
        self._catalog = catalog
        self._projects = projects

    def list(self, owner: str, project_id: str) -> list[dict[str, Any]]:
        self._projects.require(owner, project_id)
        return [model_response(entry) for entry in self._catalog.list_for_project(owner, project_id)]

    def get(self, owner: str, project_id: str, model_id: str) -> dict[str, Any]:
        self._projects.require(owner, project_id)
        try:
            return model_response(self._catalog.get_owned(owner, project_id, model_id))
        except KeyError:
            raise ModelUnavailable("model unavailable") from None
