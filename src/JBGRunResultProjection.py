"""Explicit safe JSON projection for AutomaticClassifier.run() results (281l).

Do not serialize data frames, labels, estimator objects, paths or arbitrary
engine dictionaries. Extend this allowlist only with reviewed result contracts.
"""
from __future__ import annotations

from typing import Any


def project_classifier_result(result: Any) -> dict[str, Any]:
    """Return a JSON-native summary of the documented classifier result.

    AutomaticClassifier.post_run returns a dictionary containing
    ``mispredicted`` (a DataFrame or None) and optionally ``unique_classes``.
    An early exit may return False. Unknown shapes are not inspected.
    """
    summary: dict[str, Any] = {"schema_version": 1, "kind": "unavailable"}
    if not isinstance(result, dict) or "mispredicted" not in result:
        return summary
    mispredicted = result["mispredicted"]
    if mispredicted is None:
        return {"schema_version": 1, "kind": "classification",
                "mispredicted_count": None, "class_count": None}
    # Only accept a real pandas DataFrame, not arbitrary objects with a
    # user-defined shape/len protocol. Neither data nor index is serialized.
    try:
        from pandas import DataFrame
    except ImportError:
        return summary
    if not isinstance(mispredicted, DataFrame):
        return summary
    count = int(mispredicted.shape[0])
    classes = result.get("unique_classes")
    # Labels can contain sensitive data: expose only a count for well-known
    # sequences, not their values. Strings are not class collections.
    class_count = len(classes) if isinstance(classes, (list, tuple, set, frozenset)) else None
    return {"schema_version": 1, "kind": "classification",
            "mispredicted_count": count, "class_count": class_count}
