"""Target-class diagnostics from existing training-only CV fits."""
from collections import Counter
import hashlib
import json

import numpy as np
from sklearn.metrics import get_scorer


CV_TARGET_COLUMNS = [
    "CV target", "CV target precision", "CV target recall", "CV target F1",
    "CV predicted positives", "CV target support", "CV validation rows",
    "CV diagnostic status",
]
COUNT_KEYS = ("tp", "fp", "fn", "predicted", "support", "rows")


def partition_signature(ids, labels):
    """Log a stable ID/label signature without exposing individual row IDs."""
    rows = sorted((int(key), str(label)) for key, label in zip(ids, labels))
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_cv_target(y, requested=""):
    """Use the explicit target, otherwise the least frequent training label."""
    counts = Counter(np.asarray(y).ravel().tolist())
    if requested:
        matches = [label for label in counts if str(label) == str(requested)]
        if len(matches) == 1:
            return matches[0], ""
        return None, f"Configured target {requested!r} is absent or ambiguous in training labels"
    if not counts:
        return None, "No training labels"
    # Break equal-support ties independently of input/fetch order.
    target = min(counts, key=lambda label: (counts[label], type(label).__name__, str(label)))
    return target, ""


class TargetCVScorer:
    """Keep the selection score authoritative; diagnostics never require another fit."""
    def __init__(self, scorer, target, target_error=""):
        self.scorer = get_scorer(scorer) if isinstance(scorer, str) else scorer
        self.target = target
        self.target_error = target_error

    def __call__(self, estimator, X, y):
        # Selection errors propagate unchanged. Only added diagnostics are optional.
        score = self.scorer(estimator, X, y)
        result = {"score": score, "target_available": 0.0,
                  **{f"target_{key}": np.nan for key in COUNT_KEYS}}
        if self.target_error or self.target is None:
            return result
        try:
            truth = np.asarray(y).ravel() == self.target
            predictions = np.asarray(estimator.predict(X)).ravel()
            if predictions.size != truth.size:
                return result
            predicted = predictions == self.target
            result.update(
                target_tp=float(np.sum(truth & predicted)),
                target_fp=float(np.sum(~truth & predicted)),
                target_fn=float(np.sum(truth & ~predicted)),
                target_predicted=float(np.sum(predicted)),
                target_support=float(np.sum(truth)), target_rows=float(truth.size),
                target_available=1.0,
            )
        except Exception:
            # A diagnostic-only prediction failure must not discard a valid candidate.
            pass
        return result


def summarize_target_cv(results, target, target_error=""):
    """Pool held-out-fold counts, rather than averaging fold ratios."""
    summary = dict.fromkeys(CV_TARGET_COLUMNS, np.nan)
    summary["CV target"] = "" if target is None else str(target)
    summary["CV diagnostic status"] = target_error or "Unavailable: target diagnostics failed or missing"
    available = np.asarray(results.get("test_target_available", []))
    if target_error or available.size == 0 or not np.all(available == 1):
        return summary
    totals = {key: np.sum(results[f"test_target_{key}"]) for key in COUNT_KEYS}
    if not all(np.isfinite(value) for value in totals.values()):
        return summary
    tp, fp, fn = (totals[key] for key in ("tp", "fp", "fn"))
    summary.update({
        "CV target precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "CV target recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "CV target F1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "CV predicted positives": int(totals["predicted"]),
        "CV target support": int(totals["support"]),
        "CV validation rows": int(totals["rows"]),
        "CV diagnostic status": "OK: pooled held-out CV folds",
    })
    return summary


def annotate_grid_target_cv(results, folds, target, target_error=""):
    """Add the same pooled diagnostics to every existing GridSearch candidate."""
    for index in results.index:
        fold_results = {}
        for key in (*COUNT_KEYS, "available"):
            columns = [f"split{fold}_test_target_{key}" for fold in range(folds)]
            if all(column in results for column in columns):
                fold_results[f"test_target_{key}"] = results.loc[index, columns].to_numpy(dtype=float)
        summary = summarize_target_cv(fold_results, target, target_error)
        for column, value in summary.items():
            if column not in results:
                results[column] = "" if column in ("CV target", "CV diagnostic status") else np.nan
            results.at[index, column] = value
