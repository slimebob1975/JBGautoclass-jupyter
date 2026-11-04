"""285m2: Allowlisted, label-free evaluation DTO from the existing logger contract.

Capture only aggregate counts and scores. Never serialize class names, per-row
predictions, arbitrary engine values or raw estimator objects.
"""
from __future__ import annotations

from math import isfinite

from JBGScoring import evaluation_diagnostics

TASK_STAGES = frozenset({
    "load_model", "fetch_data", "load_dataset", "separate_dataset",
    "convert_to_numbers", "train_model", "evaluate_model",
    "feature_importance", "retrain_model", "display_mispredicted",
    "make_predictions", "send_completetion_email",
})


def _score(value, lower=0.0, upper=1.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if isfinite(value) and lower <= value <= upper else None


def evaluation_from_report(*, accuracy_score, confusion_matrix, class_labels,
                           classification_matrix, sample_rates=None):
    """Return an allowlisted DTO, or None when inputs are not trustworthy."""
    try:
        import numpy as np
        matrix = np.asarray(confusion_matrix)
        if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
                or not 1 <= matrix.shape[0] <= 32 or matrix.dtype.kind not in "iu"
                or np.any(matrix < 0) or int(matrix.sum()) > 2**53):
            return None
        if not isinstance(class_labels, (list, tuple)) or len(class_labels) != matrix.shape[0]:
            return None
        accuracy = _score(accuracy_score)
        if accuracy is None:
            return None
        report = classification_matrix if isinstance(classification_matrix, dict) else {}
        macro = report.get("macro avg", {})
        weighted = report.get("weighted avg", {})
        if not isinstance(macro, dict) or not isinstance(weighted, dict):
            return None
        diagnostics = evaluation_diagnostics(matrix)
        return {
            "schema_version": 1,
            "accuracy": accuracy,
            "balanced_accuracy": _score(diagnostics.get("Balanced accuracy for evaluation data")),
            "mcc": _score(diagnostics.get("Matthews correlation coefficient (MCC) for evaluation data"), -1, 1),
            "precision_macro": _score(macro.get("precision")),
            "recall_macro": _score(macro.get("recall")),
            "f1_macro": _score(macro.get("f1-score")),
            "precision_weighted": _score(weighted.get("precision")),
            "recall_weighted": _score(weighted.get("recall")),
            "f1_weighted": _score(weighted.get("f1-score")),
            # Class labels and support per class are deliberately omitted.
            "confusion_matrix": matrix.tolist(),
        }
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        return None


def sanitize_stored_evaluation(value):
    """Never echo arbitrary keys or unbounded matrices from stored results."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return None
    fields = ("accuracy", "balanced_accuracy", "mcc", "precision_macro",
              "recall_macro", "f1_macro", "precision_weighted",
              "recall_weighted", "f1_weighted")
    result = {"schema_version": 1}
    for name in fields:
        result[name] = _score(value.get(name), -1, 1) if name == "mcc" else _score(value.get(name))
    matrix = value.get("confusion_matrix")
    if not isinstance(matrix, list) or not 1 <= len(matrix) <= 32:
        return None
    n = len(matrix)
    if not all(isinstance(row, list) and len(row) == n and all(
            type(item) is int and 0 <= item <= 2**53 for item in row) for row in matrix):
        return None
    if sum(sum(row) for row in matrix) > 2**53:
        return None
    result["confusion_matrix"] = [row[:] for row in matrix]
    return result
