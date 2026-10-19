"""Development-only probability threshold diagnostics and person-overlap audit."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score

THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15,
              0.20, 0.25, 0.30, 0.40, 0.50)

def build_threshold_table(labels, probabilities, positive_label, thresholds=THRESHOLDS):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape:
        raise ValueError("Labels and probabilities must be aligned one-dimensional arrays")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and within [0,1]")
    classes = np.unique(labels)
    if len(classes) != 2 or positive_label not in classes:
        raise ValueError("Binary labels including positive_label are required")
    actual = labels == positive_label
    rows = []
    for threshold in thresholds:
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be strictly between 0 and 1")
        predicted = probabilities >= threshold
        tp = int(np.sum(predicted & actual))
        fp = int(np.sum(predicted & ~actual))
        fn = int(np.sum(~predicted & actual))
        tn = int(np.sum(~predicted & ~actual))
        rows.append({"threshold": threshold, "positive_label": positive_label,
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "predicted_positive": tp + fp,
                     "precision": precision_score(actual, predicted, zero_division=0),
                     "recall": recall_score(actual, predicted, zero_division=0),
                     "f1": f1_score(actual, predicted, zero_division=0),
                     "mcc": matthews_corrcoef(actual, predicted)})
    return pd.DataFrame(rows)

def identity_overlap(train, holdout, column="iafId"):
    if not isinstance(train, pd.DataFrame) or not isinstance(holdout, pd.DataFrame):
        return None
    if column not in train or column not in holdout:
        return None
    a = set(train[column].dropna())
    b = set(holdout[column].dropna())
    return len(a & b), len(a), len(b)
