"""Owner-bound run service on the durable job store (revision 285c).

The durable counterpart of RunAPIPresentationService + PrincipalRunAccessService:
the same authorization on submit and the same response shapes for reads, but
jobs are stored in RunJobStore and executed by worker processes (dispatcher),
so they survive a server restart.

Responses never contain progress messages, engine error messages, data frames
or credentials. Unknown runs and other users' runs are indistinguishable.
"""
from __future__ import annotations

from typing import Any, Callable

from JBGApplicationContracts import DataSourceKind, RunReference, RunStatus
from JBGIdentity import AuthenticatedPrincipal
from JBGRunJobStore import RunJobStore
from JBGRunPreparationService import PreparedRun, RunConfiguration
from JBGRunWorker import build_job_spec
from Storage.ArtifactContract import StorageScope


class DurableRunService:
    """Submit, observe and cancel stored runs on behalf of their owners."""

    def __init__(self, store: RunJobStore, *,
                 scope_for: Callable[[AuthenticatedPrincipal], StorageScope],
                 authorize_dataset: Callable[[AuthenticatedPrincipal, PreparedRun], None]
                 | None = None,
                 on_submit: Callable[[], None] | None = None) -> None:
        if not isinstance(store, RunJobStore):
            raise TypeError("RunJobStore required")
        if not callable(scope_for):
            raise TypeError("trusted scope resolver required")
        if authorize_dataset is not None and not callable(authorize_dataset):
            raise TypeError("dataset authorizer must be callable")
        if on_submit is not None and not callable(on_submit):
            raise TypeError("on_submit must be callable")
        self._store = store
        self._scope_for = scope_for
        self._authorize_dataset = authorize_dataset
        self._on_submit = on_submit

    # --- Submitting (same authorization as PrincipalRunAccessService) ------------------

    def _authorize(self, principal: AuthenticatedPrincipal, prepared: PreparedRun) -> StorageScope:
        if not isinstance(principal, AuthenticatedPrincipal):
            raise PermissionError("authenticated principal required")
        if not isinstance(prepared, PreparedRun):
            raise TypeError("prepared run required")
        scope = self._scope_for(principal)
        if not isinstance(scope, StorageScope):
            raise PermissionError("authorized storage scope unavailable")
        if prepared.dataset.source is DataSourceKind.CSV:
            if getattr(prepared.resolved_dataset, "scope", None) != scope:
                raise PermissionError("dataset unavailable")
        if prepared.dataset.source is DataSourceKind.SQL and self._authorize_dataset is None:
            raise PermissionError("SQL dataset authorization required")
        if self._authorize_dataset is not None:
            self._authorize_dataset(principal, prepared)
        return scope

    def submit(self, principal: AuthenticatedPrincipal, prepared: PreparedRun, *,
               configuration_id: str | None = None,
               run_metadata: dict | None = None) -> dict[str, Any]:
        """Authorize, store and queue a prepared run.

        CSV runs name their saved column-role configuration (``configuration_id``,
        trusted, server-side). The stored specification is derived from the
        prepared run itself, so it cannot disagree with what was authorized.
        """
        scope = self._authorize(principal, prepared)
        source = prepared.dataset.source
        configuration = prepared.configuration
        if not isinstance(configuration, RunConfiguration) or configuration.source is not source:
            raise TypeError("prepared run must carry a RunConfiguration for its source")
        config = configuration.payload
        if source is DataSourceKind.CSV:
            spec = build_job_spec(source, config, scope=scope, artifact_id=prepared.dataset.key,
                                  configuration_id=configuration_id)
        else:
            spec = build_job_spec(source, config)
        if run_metadata is not None:
            # 288a: which project, configuration version, dataset and model a
            # project run uses. Plain JSON; the store refuses secrets.
            spec["project_run"] = dict(run_metadata)
        run = self._store.create(principal.principal_key, source, spec)
        if self._on_submit is not None:
            self._on_submit()
        return self._snapshot(self._store.get(run))

    # --- Reading and cancelling (owner key, as RunAPIPresentationService) -----------------

    def _owned(self, principal: str, run: RunReference):
        if not isinstance(principal, str) or not principal.strip():
            raise PermissionError("authentication required")
        if not isinstance(run, RunReference):
            raise PermissionError("run unavailable")
        try:
            return self._store.get_owned(principal, run)
        except KeyError:
            raise PermissionError("run unavailable") from None

    @staticmethod
    def _snapshot(record) -> dict[str, Any]:
        return {"run_id": record.run.key, "status": record.status.value,
                "has_error": record.status is RunStatus.FAILED}

    def snapshot(self, principal: str, run: RunReference) -> dict[str, Any]:
        return self._snapshot(self._owned(principal, run))

    def progress(self, principal: str, run: RunReference) -> list[dict[str, Any]]:
        self._owned(principal, run)
        # Messages are not returned: they may name pipelines, columns or tables.
        return [{"run_id": run.key, "timestamp": event.timestamp.isoformat(),
                 "percent": event.percent} for event in self._store.progress(run)]

    def result(self, principal: str, run: RunReference) -> dict[str, Any]:
        record = self._owned(principal, run)
        return {"run_id": run.key, "available": record.status is RunStatus.SUCCEEDED}

    def result_details(self, principal: str, run: RunReference) -> dict[str, Any]:
        record = self._owned(principal, run)
        if record.status is not RunStatus.SUCCEEDED or record.result is None:
            return {"run_id": run.key, "available": False, "summary": None}
        return {"run_id": run.key, "available": True, "summary": record.result}

    def overview(self, principal: str, run: RunReference) -> dict[str, Any]:
        """Safe, versioned GUI overview; authorization precedes progress reads."""
        from JBGRunOverview import project_run_overview
        record = self._owned(principal, run)
        return project_run_overview(run.key, record.status, self._store.progress(run),
                                    record.result, error_type=record.error_type)

    def cancel(self, principal: str, run: RunReference) -> dict[str, Any]:
        """Queued jobs stop at once; a running job stops at its next checkpoint."""
        self._owned(principal, run)
        self._store.request_cancel(run)
        return self._snapshot(self._store.get(run))

    def list_project_runs(self, principal: str, project_id: str,
                          limit: int = 20) -> list[dict[str, Any]]:
        """294a: the owner's runs of one project (newest first), for the Run tab."""
        if not isinstance(principal, str) or not principal.strip():
            raise PermissionError("authentication required")
        runs = []
        for record in self._store.list_for_owner(principal, 200):
            meta = record.spec.get("project_run") if isinstance(record.spec, dict) else None
            if not isinstance(meta, dict) or meta.get("project_id") != project_id:
                continue
            runs.append(dict(self._snapshot(record),
                             created_at=record.created_at.isoformat(),
                             finished_at=record.finished_at.isoformat() if record.finished_at else None,
                             configuration_name=meta.get("configuration_name"),
                             configuration_version=meta.get("configuration_version"),
                             kind=meta.get("kind", "training")))           # 295c
            if len(runs) >= limit:
                break
        return runs

    def list_runs(self, principal: str, limit: int = 50) -> list[dict[str, Any]]:
        if not isinstance(principal, str) or not principal.strip():
            raise PermissionError("authentication required")
        return [self._snapshot(record) for record in self._store.list_for_owner(principal, limit)]
