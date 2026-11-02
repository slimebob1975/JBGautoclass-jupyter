"""Synchronous in-memory job lifecycle around the existing run execution service.

A job is created before execution so failures remain observable by RunReference.
This adapter does not schedule work, persist results or implement cancellation.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any
from uuid import uuid4

from JBGApplicationContracts import RunReference, RunSnapshot, RunStatus
from JBGRunExecutionService import RunExecutionService
from JBGRunPreparationService import PreparedRun


@dataclass
class _Job:
    prepared: PreparedRun
    status: RunStatus = RunStatus.PENDING
    result: Any = None
    error: Exception | None = None


class RunJobService:
    """Local synchronous job registry; not a durable or multi-worker store."""

    def __init__(self, execution: RunExecutionService) -> None:
        if not isinstance(execution, RunExecutionService):
            raise TypeError("execution must be RunExecutionService")
        self._execution = execution
        self._jobs: dict[RunReference, _Job] = {}
        self._lock = RLock()

    def create(self, prepared: PreparedRun) -> RunReference:
        if not isinstance(prepared, PreparedRun):
            raise TypeError("expected PreparedRun")
        reference = RunReference(str(uuid4()))
        with self._lock:
            self._jobs[reference] = _Job(prepared=prepared)
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
        try:
            result = self._execution.execute(prepared)
        except Exception as error:
            with self._lock:
                job.error = error
                job.status = RunStatus.FAILED
            raise
        with self._lock:
            job.result = result
            job.status = RunStatus.SUCCEEDED
        return result

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
