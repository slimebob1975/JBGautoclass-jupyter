"""Fail-closed, transport-neutral read boundary for in-memory ML jobs (281g).

The caller supplies an authenticated principal. No principal is inferred from a
run ID, request parameter, prepared run, or dataset reference. This is not an
HTTP authentication implementation or a multi-worker persistence layer.
"""
from __future__ import annotations

from threading import RLock
from typing import Any

from JBGApplicationContracts import RunReference, RunStatus
from JBGRunJobService import RunJobService
from JBGRunPreparationService import PreparedRun
from JBGRunResultProjection import project_classifier_result


class RunAPIPresentationService:
    """Owns job IDs created through this facade and exposes safe observations.

    Deliberately does not serialize model results or engine exception messages.
    Raw objects remain accessible exclusively through the internal job service.
    """

    def __init__(self, jobs: RunJobService) -> None:
        if not isinstance(jobs, RunJobService):
            raise TypeError("jobs must be RunJobService")
        self._jobs = jobs
        self._owners: dict[RunReference, str] = {}
        self._lock = RLock()

    @staticmethod
    def _principal(principal: str) -> str:
        if not isinstance(principal, str) or not principal.strip():
            raise PermissionError("authentication required")
        return principal

    def create(self, principal: str, prepared: PreparedRun) -> dict[str, Any]:
        owner = self._principal(principal)
        # Must be called only after dataset authorization by the composition root.
        # PreparedRun itself does not confer authorization.
        with self._lock:
            run = self._jobs.create(prepared)
            self._owners[run] = owner
            return self._snapshot(run)

    def _authorized(self, principal: str, run: RunReference) -> None:
        owner = self._principal(principal)
        if not isinstance(run, RunReference) or self._owners.get(run) != owner:
            # No distinction between unknown and someone else's run.
            raise PermissionError("run unavailable")

    def _snapshot(self, run: RunReference) -> dict[str, Any]:
        snapshot = self._jobs.snapshot(run)
        # Never return snapshot.message: it may include paths, SQL or secrets.
        return {"run_id": run.key, "status": snapshot.status.value,
                "has_error": snapshot.status is RunStatus.FAILED}

    def snapshot(self, principal: str, run: RunReference) -> dict[str, Any]:
        with self._lock:
            self._authorized(principal, run)
            return self._snapshot(run)

    def execute(self, principal: str, run: RunReference) -> dict[str, Any]:
        # Authorization is checked before triggering the engine. The underlying
        # RunJobService enforces the one-shot execution state transition.
        with self._lock:
            self._authorized(principal, run)
        try:
            self._jobs.execute(run)
        except Exception:
            # The original exception remains in RunJobService.error(run).
            # It must not be serialized into an API response.
            return self.snapshot(principal, run)
        return self.snapshot(principal, run)

    def progress(self, principal: str, run: RunReference) -> list[dict[str, Any]]:
        with self._lock:
            self._authorized(principal, run)
            events = self._jobs.progress_history(run)
        return [{"run_id": run.key, "timestamp": event.timestamp.isoformat(),
                 "percent": event.percent} for event in events]

    def result(self, principal: str, run: RunReference) -> dict[str, Any]:
        with self._lock:
            self._authorized(principal, run)
            snapshot = self._jobs.snapshot(run)
        # A future explicit, allowlisted result DTO can be added separately.
        return {"run_id": run.key, "available": snapshot.status is RunStatus.SUCCEEDED}

    def result_details(self, principal: str, run: RunReference) -> dict[str, Any]:
        """Allowlisted JSON result; owner check precedes internal result access."""
        with self._lock:
            self._authorized(principal, run)
            snapshot = self._jobs.snapshot(run)
            if snapshot.status is not RunStatus.SUCCEEDED:
                return {"run_id": run.key, "available": False, "summary": None}
            raw = self._jobs.result(run)
        return {"run_id": run.key, "available": True,
                "summary": project_classifier_result(raw)}
