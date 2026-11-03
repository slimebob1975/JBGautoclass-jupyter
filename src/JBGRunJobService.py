"""Synchronous in-memory job lifecycle around the existing run execution service.

A job is created before execution so failures remain observable by RunReference.
This adapter does not schedule work, persist results or implement cancellation.

Revision 284a: a job may carry its own execution service, composed for one
storage scope and configuration (see JBGRunComposition). Without one, the
service passed to the constructor is used, as before.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any
from uuid import uuid4

from JBGApplicationContracts import RunReference, RunSnapshot, RunStatus
from JBGRunExecutionService import RunExecutionService
from JBGRunPreparationService import PreparedRun
from JBGRunProgress import RunProgressService, _ACTIVE_RUN


@dataclass
class _Job:
    prepared: PreparedRun
    execution: RunExecutionService
    status: RunStatus = RunStatus.PENDING
    result: Any = None
    error: Exception | None = None


class RunJobService:
    """Local synchronous job registry; not a durable or multi-worker store."""

    def __init__(self, execution: RunExecutionService, *,
                 progress: RunProgressService | None = None) -> None:
        if not isinstance(execution, RunExecutionService):
            raise TypeError("execution must be RunExecutionService")
        self._execution = execution
        self._progress = progress if progress is not None else RunProgressService()
        self._jobs: dict[RunReference, _Job] = {}
        self._lock = RLock()

    def create(self, prepared: PreparedRun, *,
               execution: RunExecutionService | None = None) -> RunReference:
        """Register a job; ``execution`` must come from a trusted composition root."""
        if not isinstance(prepared, PreparedRun):
            raise TypeError("expected PreparedRun")
        if execution is None:
            execution = self._execution
        elif not isinstance(execution, RunExecutionService):
            raise TypeError("execution must be RunExecutionService")
        reference = RunReference(str(uuid4()))
        with self._lock:
            self._progress.register(reference)
            self._jobs[reference] = _Job(prepared=prepared, execution=execution)
        return reference

    def snapshot(self, reference: RunReference) -> RunSnapshot:
        with self._lock:
            job = self._get_job(reference)
            message = str(job.error) if job.error is not None else None
            return RunSnapshot(reference, job.status, message)

    def execute(self, reference: RunReference) -> Any:
        """Run exactly once. Engine errors remain available and are re-raised."""
        with self._lock:
            job = self._get_job(reference)
            if job.status is not RunStatus.PENDING:
                raise ValueError("run has already started")
            job.status = RunStatus.RUNNING
            prepared = job.prepared
            execution = job.execution
        try:
            token = _ACTIVE_RUN.set(reference)
            try:
                result = execution.execute(prepared)
            finally:
                _ACTIVE_RUN.reset(token)
        except Exception as error:
            with self._lock:
                job.error = error
                job.status = RunStatus.FAILED
            raise
        with self._lock:
            job.result = result
            job.status = RunStatus.SUCCEEDED
        return result

    def progress_history(self, reference: RunReference):
        with self._lock:
            self._get_job(reference)
        return self._progress.history(reference)

    def latest_progress(self, reference: RunReference):
        with self._lock:
            self._get_job(reference)
        return self._progress.latest(reference)

    def result(self, reference: RunReference) -> Any:
        with self._lock:
            job = self._get_job(reference)
            if job.status is not RunStatus.SUCCEEDED:
                raise ValueError("run has not succeeded")
            return job.result

    def error(self, reference: RunReference) -> Exception | None:
        with self._lock:
            job = self._get_job(reference)
            return job.error

    def _get_job(self, reference: RunReference) -> _Job:
        if not isinstance(reference, RunReference):
            raise TypeError("expected RunReference")
        try:
            return self._jobs[reference]
        except KeyError:
            raise KeyError("unknown run reference") from None
