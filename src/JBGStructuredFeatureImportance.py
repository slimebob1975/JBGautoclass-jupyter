"""Allowlisted permutation-importance result for owner-authorized run overviews.

Only aggregate, original-column importance is stored. No rows, class labels,
permutation draws, fitted estimators, or arbitrary feature metadata are persisted.
"""
from __future__ import annotations

from math import isfinite

MAX_FEATURES = 512
MAX_NAME_LENGTH = 128
MAX_ROWS = 2**53
MAX_ABS_SCORE = 1e12


def _number(value):
    # Reject numpy objects and bools at the storage boundary; explicitly
    # convert trusted computation outputs before calling this validator.
    if type(value) not in (int, float) or not isfinite(value):
        return None
    number = float(value)
    return number if abs(number) <= MAX_ABS_SCORE else None


def _feature_name(value):
    if not isinstance(value, str) or not value or len(value) > MAX_NAME_LENGTH:
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return None
    return value


def sanitize_stored_feature_importance(value):
    """Return only a bounded, JSON-safe DTO, or None for malformed input."""
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int \
            or value["schema_version"] != 1:
        return None
    baseline = _number(value.get("baseline_score"))
    rows, repeats, total = (value.get(key) for key in
                            ("holdout_rows", "repeats", "total_features"))
    if baseline is None or type(rows) is not int or not 1 <= rows <= MAX_ROWS \
            or type(repeats) is not int or not 2 <= repeats <= 30 \
            or type(total) is not int or not 1 <= total <= MAX_FEATURES:
        return None
    entries = value.get("features")
    if not isinstance(entries, list) or len(entries) != total:
        return None
    cleaned, names = [], set()
    for rank, item in enumerate(entries, 1):
        if not isinstance(item, dict) or type(item.get("rank")) is not int \
                or item["rank"] != rank:
            return None
        name = _feature_name(item.get("name"))
        mean = _number(item.get("mean_score_decrease"))
        std = _number(item.get("std_across_repeats"))
        if name is None or name in names or mean is None or std is None or std < 0:
            return None
        names.add(name)
        cleaned.append({"rank": rank, "name": name,
                        "mean_score_decrease": mean, "std_across_repeats": std})
    return {"schema_version": 1, "method": "permutation",
            "baseline_score": baseline, "holdout_rows": rows,
            "repeats": repeats, "total_features": total, "features": cleaned}


def feature_importance_from_report(report):
    """Adapt an existing computed FeatureImportanceReport, without recomputation."""
    try:
        table = report.table
        if not 1 <= len(table) <= MAX_FEATURES:
            return None
        entries = []
        for row in table.itertuples(index=False, name=None):
            # Expected order is Rank, Feature, Mean score decrease,
            # Std. across repeats, followed by per-repeat values (discarded).
            rank, name, mean, std = row[:4]
            entries.append({"rank": int(rank), "name": name,
                            "mean_score_decrease": float(mean),
                            "std_across_repeats": float(std)})
        return sanitize_stored_feature_importance({
            "schema_version": 1, "baseline_score": float(report.baseline_score),
            "holdout_rows": int(report.rows), "repeats": int(report.repeats),
            "total_features": len(entries), "features": entries,
        })
    except (AttributeError, TypeError, ValueError, OverflowError, IndexError):
        return None
