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

    # Exact running task markers are emitted by the task runner's optional
    # stage hook. Other progress messages must never be interpreted as tasks.
    from JBGStructuredEvaluation import TASK_STAGES
    if status is RunStatus.RUNNING:
        for event in reversed(tuple(events)):
            message = event.message
            if isinstance(message, str) and message.startswith("jbg_stage:"):
                task = message[len("jbg_stage:"):]
                if task in TASK_STAGES:
                    phase = task
                    break

    summary = None
    if status is RunStatus.SUCCEEDED and isinstance(result, dict):
        # Deliberately copy only reviewed numeric fields, never arbitrary keys.
        summary = {"kind": "classification"} if result.get("kind") == "classification" else None
        if summary is not None:
            for field in ("class_count", "mispredicted_count"):
                value = result.get(field)
                summary[field] = value if (type(value) is int and value >= 0) else None

    evaluation = None
    if status is RunStatus.SUCCEEDED and isinstance(result, dict):
        from JBGStructuredEvaluation import sanitize_stored_evaluation
        evaluation = sanitize_stored_evaluation(result.get("evaluation"))
    results = {"available": status is RunStatus.SUCCEEDED, "summary": summary}
    if evaluation is not None:
        results["evaluation"] = evaluation
    return {"schema_version": 2, "run_id": run_id, "status": status.value,
            "stage": {"name": phase, "precision": "exact" if status is not RunStatus.RUNNING
                      or phase in TASK_STAGES
                      else "estimated", "percent": percent},
            "results": results}
