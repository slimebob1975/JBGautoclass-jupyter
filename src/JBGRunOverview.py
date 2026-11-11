"""285m: safe, versioned run overview for the FastAPI GUI.

The engine currently persists only an allowlisted classifier summary. Do not
interpret log text as metrics or expose raw progress messages. Progress phase
is a coarse UI estimate, not a claim about the active ML task.

297a: ``stage.inner`` ({kind, done, total}) while a running run has an inner
progress bar, for example the algorithm combinations of a spot check.
"""
from __future__ import annotations

from math import isfinite
from typing import Any

from JBGApplicationContracts import RunStatus


# 297a: the engine's inner progress bars (``start_inline_progress``), by the
# description it gives them, as the kinds the interface may show. Others are
# not recorded.
INNER_PROGRESS = {
    "Checking & Training models": "models",
    "Fetching data": "fetch_data",
    "Validation progress": "validation",
    "Feature importance": "feature_importance",
    "Shadow fits": "shadow_fits",
}
INNER_KINDS = frozenset(INNER_PROGRESS.values())


def _inner(events) -> dict[str, Any] | None:
    """The inner progress, if it was recorded after the newest stage marker."""
    for event in reversed(tuple(events)):
        message = event.message
        if not isinstance(message, str):
            continue
        if message.startswith("jbg_stage:"):
            return None                      # a later stage: the inner bar is over
        if message.startswith("jbg_inner:"):
            parts = message[len("jbg_inner:"):].split(":")
            if len(parts) != 3 or parts[0] not in INNER_KINDS:
                return None
            if not (parts[1].isdigit() and parts[2].isdigit()):
                return None
            done, total = int(parts[1]), int(parts[2])
            if total < 1 or done > total:
                return None
            return {"kind": parts[0], "done": done, "total": total}
    return None


def _percent(events) -> float | None:
    for event in reversed(tuple(events)):
        value = event.percent
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if isfinite(value) and 0 <= value <= 1:
                return value
    return None


# 297e: why a run failed, as a fixed code the interface turns into a text. The
# stored error_type is a class name chosen by the worker or dispatcher (never
# message text); unknown ones are "other".
FAILURES = {
    "NoModelTrained": "no_model",                # 297b
    "SystemExit": "engine_stopped",              # the engine's abort_cleanly()
    "WorkerStartFailed": "worker_start",
    "PermissionError": "sign_in",
    "ConfigException": "model_mismatch",         # e.g. the model does not fit the data
}


def failure_code(status: RunStatus, error_type: str | None) -> str | None:
    if status is RunStatus.INTERRUPTED:
        return "interrupted"
    if status is not RunStatus.FAILED:
        return None
    return FAILURES.get(error_type, "other")


def project_run_overview(run_id: str, status: RunStatus, events,
                         result: dict[str, Any] | None = None, *,
                         error_type: str | None = None) -> dict[str, Any]:
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
    if status is RunStatus.SUCCEEDED and isinstance(result, dict):
        from JBGStructuredFeatureImportance import sanitize_stored_feature_importance
        importance = sanitize_stored_feature_importance(result.get("feature_importance"))
        if importance is not None:
            results["feature_importance"] = importance
    stage = {"name": phase, "precision": "exact" if status is not RunStatus.RUNNING
             or phase in TASK_STAGES else "estimated", "percent": percent}
    if status is RunStatus.RUNNING:
        inner = _inner(events)
        if inner is not None:
            stage["inner"] = inner                       # 297a: only while running
    overview = {"schema_version": 2, "run_id": run_id, "status": status.value,
                "stage": stage, "results": results}
    failure = failure_code(status, error_type)
    if failure is not None:
        overview["failure"] = failure                   # 297e: failed or interrupted only
    return overview
