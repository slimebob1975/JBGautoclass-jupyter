"""Scoring explanations and descriptive diagnostics for single-label classification."""

import numpy as np


SCORING_DISPLAY_NAMES = {
    "f1_micro": "F1 Micro",
    "f1_macro": "F1 Macro",
    "f1_weighted": "F1 Weighted",
}

_SCORING_HELP = {
    "accuracy": (
        "Accuracy: fraction of correct predictions. Frequent classes dominate; high accuracy "
        "can hide poor minority-class recall. Consider F1 Macro or MCC for imbalanced classes."
    ),
    "f1_micro": (
        "F1 Micro: pools counts across all classes. In single-label classification over all "
        "classes it equals accuracy, so frequent classes dominate. Consider F1 Macro or MCC "
        "when minority-class performance matters."
    ),
    "f1_macro": (
        "F1 Macro: unweighted mean of each class's F1. Every class contributes equally, "
        "including rare classes; F1 combines precision and recall."
    ),
    "f1_weighted": (
        "F1 Weighted: mean of each class's F1 weighted by its observed support. Frequent "
        "classes contribute more, so poor minority-class performance can remain hidden."
    ),
    "balanced_accuracy": (
        "Balanced Accuracy: unweighted mean recall across classes. Each class contributes "
        "equally, regardless of its frequency; precision is not included."
    ),
    "balanced_accuracy_adjusted": (
        "Balanced Accuracy (Adjusted): mean class recall adjusted so chance-level performance "
        "is 0 and perfect performance is 1."
    ),
    "mcc": (
        "Matthews Correlation Coefficient (MCC): uses the full confusion matrix, including "
        "both missed positives and false alarms. Range -1 to 1; 1 is perfect, 0 indicates "
        "no measured correlation, and -1 indicates complete binary disagreement."
    ),
}


def score_metric_help(metric_name: str | None) -> str:
    explanation = _SCORING_HELP.get(metric_name, "The selected metric measures classification performance.")
    return "Used to rank candidates and tune parameters by cross-validation. " + explanation


def evaluation_diagnostics(confusion_matrix) -> dict[str, float]:
    """Describe the held-out predictions without changing model selection.

    The baseline is the largest *evaluation* class share: the retrospective accuracy
    of always choosing that evaluation-majority label, not a fitted dummy competitor.
    Rows are true labels, columns are predicted labels. Zero-support rows (predicted-only
    labels) are excluded from mean recall, matching sklearn balanced_accuracy_score.
    MCC is computed directly from counts to avoid rebuilding full label arrays in loggers.
    """
    counts = np.asarray(confusion_matrix, dtype=float)
    true_support = counts.sum(axis=1)
    predicted_support = counts.sum(axis=0)
    total = float(counts.sum())
    if total == 0:
        return {}
    observed = true_support > 0
    balanced_accuracy = float(np.mean(np.diag(counts)[observed] / true_support[observed]))
    covariance = float(np.trace(counts) * total - np.dot(true_support, predicted_support))
    true_variance = float(total ** 2 - np.dot(true_support, true_support))
    predicted_variance = float(total ** 2 - np.dot(predicted_support, predicted_support))
    denominator = np.sqrt(max(0.0, true_variance) * max(0.0, predicted_variance))
    mcc = float(covariance / denominator) if denominator > 0 else 0.0
    return {
        "Majority-class baseline accuracy (evaluation majority)": float(true_support.max() / total),
        "Balanced accuracy for evaluation data": balanced_accuracy,
        "Matthews correlation coefficient (MCC) for evaluation data": mcc,
    }
