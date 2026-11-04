"""285m: safe, versioned run overview for the FastAPI GUI.

The engine currently persists only an allowlisted classifier summary. Do not
interpret log text as metrics or expose raw progress messages. Progress phase
is a coarse UI estimate, not a claim about the active ML task.
"""
from __future__ import annotations

from math import isfinite
from typing import Any

from JBGApplicationContracts import RunStatus


def _percent(events) -> float | None:
    for event in reversed(tuple(events)):
        value = event.percent
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if isfinite(value) and 0 <= value <= 1:
                return value
    return None


def project_run_overview(run_id: str, status: RunStatus, events,
                         result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Produce JSON-native, allowlisted fields from already-authorized records."""
    if not isinstance(status, RunStatus):
        raise TypeError("status must be RunStatus")
    percent = _percent(events)
    if status is RunStatus.SUCCEEDED:
        phase = "succeeded"
        percent = 1.0
    elif status is RunStatus.FAILED:
        phase = "failed"
    elif status is RunStatus.CANCELLED:
        phase = "cancelled"
    elif status is RunStatus.INTERRUPTED:
        phase = "interrupted"
    elif status is RunStatus.PENDING:
        phase = "queued"
    elif status is RunStatus.AWAITING_CREDENTIALS:
        phase = "awaiting_credentials"
    else:
        phase = "preparing" if percent is None or percent == 0 else (
            "finalizing" if percent >= 0.9 else "executing")

    summary = None
    if status is RunStatus.SUCCEEDED and isinstance(result, dict):
        # Deliberately copy only reviewed numeric fields, never arbitrary keys.
        summary = {"kind": "classification"} if result.get("kind") == "classification" else None
        if summary is not None:
            for field in ("class_count", "mispredicted_count"):
                value = result.get(field)
                summary[field] = value if (type(value) is int and value >= 0) else None

    return {"schema_version": 1, "run_id": run_id, "status": status.value,
            "stage": {"name": phase, "precision": "estimated" if status is RunStatus.RUNNING
                      else "exact", "percent": percent},
            "results": {"available": status is RunStatus.SUCCEEDED,
                        "summary": summary}}
