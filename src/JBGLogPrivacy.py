"""Privacy-safe defaults for persistent runtime logs.

Rich notebook/Voila output may still show operator-requested diagnostics. Persistent
logs omit SQL text, row-level data, source-schema metadata, prediction payloads and
local filesystem paths unless the operator explicitly opts in with
``JBG_LOG_SENSITIVE_DATA=1`` in a trusted local environment.
"""
from __future__ import annotations

import os
import re


_TRUTHY = {"1", "true", "yes", "on"}
_SENSITIVE_ENV = "JBG_LOG_SENSITIVE_DATA"
_TRACEBACK_PATH_RE = re.compile(r'(?P<prefix>\bFile\s+["\'])(?P<path>[^"\']+)(?P<suffix>["\'])')
_WINDOWS_PATH_RE = re.compile(r'(?i)(?<![A-Za-z0-9_])(?:[A-Z]:\\|\\\\)[^;\r\n]+')
_POSIX_PATH_RE = re.compile(r'(?<![A-Za-z0-9_:/])/(?![A-Za-z][A-Za-z0-9:-]*\s*>)(?:[^\s;\r\n]+)')


def sensitive_log_data_enabled(value: str | None = None) -> bool:
    """Return whether sensitive runtime details may enter persistent logs."""
    if value is None:
        value = os.environ.get(_SENSITIVE_ENV, "")
    return str(value).strip().lower() in _TRUTHY


def safe_local_path(path) -> str:
    """Return a local path only when sensitive logging is explicitly enabled."""
    if sensitive_log_data_enabled():
        return str(path)
    return "<local-path>"


def redact_traceback_paths(text: str) -> str:
    """Hide quoted Python traceback filenames while preserving traceback structure."""
    value = str(text)
    if sensitive_log_data_enabled():
        return value
    return _TRACEBACK_PATH_RE.sub(
        lambda match: f"{match.group('prefix')}<local-path>{match.group('suffix')}",
        value,
    )


def redact_local_paths(text: str) -> str:
    """Best-effort removal of absolute local paths from persistent log text."""
    value = str(text)
    if sensitive_log_data_enabled():
        return value
    value = _TRACEBACK_PATH_RE.sub(
        lambda match: f"{match.group('prefix')}<local-path>{match.group('suffix')}",
        value,
    )
    value = _WINDOWS_PATH_RE.sub("<local-path>", value)
    value = _POSIX_PATH_RE.sub("<local-path>", value)
    return value


def safe_query_log_message(query_type: str, query: str) -> str:
    """Format a persistent-log SQL message without leaking SQL by default."""
    if sensitive_log_data_enabled():
        return f"Query for {query_type}: {query}"
    return (
        f"Query for {query_type}: [SQL omitted from persistent log; "
        f"set {_SENSITIVE_ENV}=1 only for trusted local diagnostics]"
    )


def safe_prediction_query_log_message(row_count: int, query: str) -> str:
    """Describe prediction output while protecting retrieval SQL in persistent logs."""
    if sensitive_log_data_enabled():
        return f"Added {row_count} rows to prediction table. Get them with SQL query: {query}"
    return (
        f"Added {row_count} rows to prediction table. Retrieval SQL omitted from persistent log; "
        f"set {_SENSITIVE_ENV}=1 only for trusted local diagnostics."
    )


def safe_prediction_values_log_message(predictions) -> str:
    """Summarize a row-level prediction payload without persisting IDs/classes."""
    if sensitive_log_data_enabled():
        return f"Predictions: {predictions}"
    try:
        entries = len(predictions)
    except Exception:
        entries = "unknown"
    return (
        "Predictions: [row-level prediction payload omitted from persistent log; "
        f"entries={entries}; set {_SENSITIVE_ENV}=1 only for trusted local diagnostics]"
    )


def safe_reclassification_log_message(index, new_class) -> str:
    """Describe a reclassification action without persisting row IDs or class labels."""
    if sensitive_log_data_enabled():
        return f"Changing data row {index} to {new_class}:"
    return (
        "Changing data row [row identifier and class omitted from persistent log; "
        f"set {_SENSITIVE_ENV}=1 only for trusted local diagnostics]:"
    )


def safe_code_log_message(key: str, code) -> str:
    """Prevent embedded download payloads and obvious local paths entering logs by default."""
    text = str(code)
    if sensitive_log_data_enabled():
        return f"{key}: {text}"
    lowered = text.lower()
    if "base64," in lowered and ("data:" in lowered or "<a " in lowered):
        return f"{key}: [embedded download payload omitted from persistent log]"
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", text):
        return f"{key}: {text}"
    if (re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith(("/", "\\\\"))
            or "\\" in text or "/" in text):
        return f"{key}: <local-path>"
    return f"{key}: {text}"


def redacted_table_log_message(matrix, category: str = "table") -> str:
    """Return a shape-only placeholder for a data/schema-derived table snapshot."""
    try:
        rows, columns = matrix.shape
        shape = f"{rows}x{columns}"
    except Exception:
        shape = "unknown"
    return (
        f"[{category} omitted from persistent log; shape={shape}; "
        f"set {_SENSITIVE_ENV}=1 only for trusted local diagnostics]"
    )
