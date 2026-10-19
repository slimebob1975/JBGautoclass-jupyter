"""Exploratory, development-only OOF model benchmarking (revision 150)."""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             matthews_corrcoef, precision_score, recall_score, f1_score)


def summarize_binary_oof(y, probabilities, classes, threshold=0.1, bins=10):
    """Scores describe the supplied held-out-fold probabilities, not unseen deployment data."""
    y = np.asarray(y)
    probs = np.asarray(probabilities, dtype=float)
    classes = list(classes)
    if len(classes) != 2 or len(set(y)) != 2:
        raise ValueError("Binary benchmark requires two present classes")
    if probs.ndim != 2 or probs.shape != (len(y), 2):
        raise ValueError("Expected one two-class probability row per label")
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("Probabilities must be finite and within [0,1]")
    positive = min(classes, key=lambda label: (np.count_nonzero(y == label), str(label)))
    scores = probs[:, classes.index(positive)]
    actual = (y == positive).astype(int)
    predicted = (scores >= threshold).astype(int)
    brier = float(brier_score_loss(actual, scores))
    # Equal-width expected calibration error, with empty bins omitted.
    edges = np.linspace(0., 1., bins + 1)
    bucket = np.minimum(np.searchsorted(edges, scores, side="right") - 1, bins - 1)
    ece = 0.
    for i in range(bins):
        select = bucket == i
        if select.any():
            ece += select.mean() * abs(float(actual[select].mean() - scores[select].mean()))
    return dict(positive_class=str(positive), rows=len(y), positives=int(actual.sum()),
                prevalence=float(actual.mean()), threshold=float(threshold),
                predicted_positives=int(predicted.sum()),
                precision=float(precision_score(actual, predicted, zero_division=0)),
                recall=float(recall_score(actual, predicted, zero_division=0)),
                f1=float(f1_score(actual, predicted, zero_division=0)),
                mcc=float(matthews_corrcoef(actual, predicted)),
                average_precision=float(average_precision_score(actual, scores)),
                roc_auc=float(roc_auc_score(actual, scores)),
                brier=brier, ece_10_bins=float(ece))
