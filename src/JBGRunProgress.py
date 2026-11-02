"""Bounded, transport-neutral observations of existing classifier progress (281e).

The logger adapter forwards every call to the original logger. Progress values
come only from the ML engine; job lifecycle events carry no invented percentage.
"""
from __future__ import annotations

from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from JBGApplicationContracts import RunReference

_ACTIVE_RUN: ContextVar[RunReference | None] = ContextVar("jbg_active_progress_run", default=None)


@dataclass(frozen=True)
class ProgressEvent:
    run: RunReference
    timestamp: datetime
    message: str | None
    percent: float | None = None


class RunProgressService:
    """In-memory per-job event history; not a persistent or shared job store."""

    def __init__(self, max_events: int = 200):
        if not isinstance(max_events, int) or isinstance(max_events, bool) or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        self._max_events = max_events
        self._events: dict[RunReference, deque[ProgressEvent]] = {}
        self._lock = RLock()

    def register(self, run: RunReference) -> None:
        if not isinstance(run, RunReference):
            raise TypeError("expected RunReference")
        with self._lock:
            if run in self._events:
                raise ValueError("run already registered")
            self._events[run] = deque(maxlen=self._max_events)

    def record(self, run: RunReference, message: str | None = None,
               percent: float | None = None) -> ProgressEvent:
        if not isinstance(run, RunReference):
            raise TypeError("expected RunReference")
        if message is not None and not isinstance(message, str):
            raise TypeError("message must be text or None")
        if percent is not None:
            if isinstance(percent, bool) or not isinstance(percent, (int, float)):
                raise TypeError("percent must be numeric or None")
            if not 0 <= percent <= 1:
                raise ValueError("percent must be between 0 and 1")
        event = ProgressEvent(run, datetime.now(timezone.utc), message, percent)
        with self._lock:
            if run not in self._events:
                raise KeyError("unknown run reference")
            self._events[run].append(event)
        return event

    def history(self, run: RunReference) -> tuple[ProgressEvent, ...]:
        if not isinstance(run, RunReference):
            raise TypeError("expected RunReference")
        with self._lock:
            if run not in self._events:
                raise KeyError("unknown run reference")
            return tuple(self._events[run])

    def latest(self, run: RunReference) -> ProgressEvent | None:
        events = self.history(run)
        return events[-1] if events else None


class ProgressObservingLogger:
    """Optional delegate for a runtime logger; never changes its progress calls.

    Wrap the logger supplied to SQLClassifierRuntime/RegisteredCSVClassifierRuntime
    with this proxy to capture genuine print_progress messages during job execution.
    Calls outside an active RunJobService.execute are just forwarded.
    """

    def __init__(self, logger: Any, progress: RunProgressService):
        if not isinstance(progress, RunProgressService):
            raise TypeError("expected RunProgressService")
        self._logger = logger
        self._progress = progress

    def print_progress(self, message=None, percent=None):
        result = self._logger.print_progress(message=message, percent=percent)
        run = _ACTIVE_RUN.get()
        if run is not None:
            self._progress.record(run, message, percent)
        return result

    def __getattr__(self, name: str):
        return getattr(self._logger, name)
