"""Durable, non-executable candidate feature-search profiles.

A nested-CV result is never eligible for automatic production reuse.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROFILE_SCHEMA = 1


def save_candidate(store, result, friendly_name=""):
    """Persist provenance and a recommendation without elevating validation status."""
    if result.get("status") != "completed":
        raise ValueError("Only completed investigations may be saved")
    recommendation = result.get("recommendation", {})
    if recommendation.get("validation_status") != "candidate":
        raise ValueError("Unexpected recommendation validation status")
    provenance = result.get("provenance", {})
    # Search exports use `study_provenance` in the `provenance` field.
    context = provenance.get("context", {})
    required = ("project", "source_tables", "class_column", "id_column")
    if not all(context.get(field) for field in required):
        raise ValueError("Source provenance is incomplete; profile cannot be safely saved")
    selected = recommendation.get("selected_columns", [])
    available = result.get("original_columns", [])
    if not selected or not set(selected).issubset(set(available)):
        raise ValueError("Recommendation columns are inconsistent with study inputs")
    record = {
        "profile_schema": PROFILE_SCHEMA,
        "profile_id": str(uuid4()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "friendly_name": str(friendly_name).strip()[:100],
        "validation_status": "candidate",
        "reuse_eligible": False,
        "provenance": provenance,
        "original_columns": available,
        "recommendation": recommendation,
        "investigative_model": result.get("investigative_model"),
        "nested_validation": result.get("nested_validation"),
        "options": result.get("options"),
        "text_options": result.get("text_options"),
        "signatures": result.get("signatures"),
        "evaluation": result.get("evaluation"),
        "target": result.get("target"),
        "output_directory": result.get("output_directory"),
        "reason": "Nested CV validates the search procedure, not the exact final subset. Independent validation and semantic compatibility review are required for reuse.",
    }
    directory = Path(store)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (record["profile_id"] + ".json")
    # Exclusive creation prevents clobbering older profile revisions.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, default=str)
    return path


def load_candidates(store):
    records = []
    for path in sorted(Path(store).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("profile_schema") == PROFILE_SCHEMA and record.get("validation_status") == "candidate":
                records.append(record)
        except (OSError, ValueError, TypeError):
            continue
    return records


def compatible_for_review(record, *, project, tables, class_column, id_column, columns):
    """Strict identity/column check, *not* authorization to reuse a profile."""
    context = record.get("provenance", {}).get("context", {})
    schema = record.get("provenance", {}).get("original_column_schema", [])
    return (record.get("profile_schema") == PROFILE_SCHEMA
            and context.get("project") == project
            and context.get("source_tables") == list(tables)
            and context.get("class_column") == class_column
            and context.get("id_column") == id_column
            and set(record.get("recommendation", {}).get("selected_columns", [])) <= set(columns)
            and {entry.get("name") for entry in schema} == set(record.get("original_columns", [])))


def eligible_for_reuse(record):
    """Fail closed until a separate independent validation workflow exists."""
    return False


def review_compatibility(record, *, project, tables, class_column, id_column, columns, column_schema=None):
    """Report mismatches without granting reuse permission. Fail closed on missing metadata."""
    problems = []
    if record.get("profile_schema") != PROFILE_SCHEMA:
        problems.append("Profile schema is unsupported")
    context = record.get("provenance", {}).get("context", {})
    for key, expected in (("project", project), ("source_tables", list(tables)),
                          ("class_column", class_column), ("id_column", id_column)):
        if not expected or context.get(key) != expected:
            problems.append(f"Source {key} differs or is unknown")
    original = record.get("original_columns", [])
    selected = record.get("recommendation", {}).get("selected_columns", [])
    if not original or not selected or not set(selected).issubset(set(original)):
        problems.append("Stored column selection is incomplete")
    if not set(selected).issubset(set(columns)):
        problems.append("Recommended columns are not all available")
    stored_schema = record.get("provenance", {}).get("original_column_schema", [])
    if not stored_schema or {item.get("name") for item in stored_schema} != set(original):
        problems.append("Stored column schema is incomplete")
    if column_schema is not None:
        actual = {item["name"]: item.get("dtype") for item in column_schema}
        for entry in stored_schema:
            if entry.get("name") in selected and actual.get(entry.get("name")) != entry.get("dtype"):
                problems.append("Selected column dtype differs: " + str(entry.get("name")))
    if record.get("validation_status") != "candidate":
        problems.append("Unrecognized profile validation status")
    return problems


def find_candidate(store, profile_id):
    """Return a stored candidate by its identifier (never trust a path from the GUI)."""
    if not isinstance(profile_id, str) or len(profile_id) != 36:
        return None
    for record in load_candidates(store):
        if record.get("profile_id") == profile_id:
            return record
    return None


def profile_display_name(record):
    """Human-readable label; the immutable UUID remains the dropdown value."""
    context = record.get("provenance", {}).get("context", {})
    project = str(context.get("project") or "Unnamed project").strip()
    tables = context.get("source_tables") or []
    source = ", ".join(str(table) for table in tables) if tables else "Unknown source"
    model = (record.get("investigative_model") or {}).get("model_type") or "Unknown model"
    friendly = str(record.get("friendly_name") or "").strip()
    timestamp = str(record.get("created_utc") or "")[:16].replace("T", " ")
    suffix = " | ".join(part for part in (model, friendly, timestamp) if part)
    return f"{project} | {source} | {suffix} (candidate)"


def delete_candidate(store, profile_id):
    """Delete exactly one UUID-named candidate; never interpret GUI input as a path."""
    from uuid import UUID
    try:
        key = str(UUID(profile_id))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Invalid recommendation identifier")
    if key != profile_id:
        raise ValueError("Invalid recommendation identifier")
    directory = Path(store)
    path = directory / (key + ".json")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Selected recommendation is no longer available")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as ex:
        raise ValueError("Saved recommendation is unreadable") from ex
    if (record.get("profile_schema") != PROFILE_SCHEMA or
            record.get("profile_id") != key or
            record.get("validation_status") != "candidate"):
        raise ValueError("Selected file does not match the candidate recommendation")
    path.unlink()
    # The audit contains only an identifier and event time, never dataset values or credentials.
    audit = directory / "deletions.jsonl"
    with audit.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "candidate_deleted", "profile_id": key,
                                 "deleted_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    return record
