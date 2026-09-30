import warnings

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef

from JBGScoring import evaluation_diagnostics, score_metric_help


BASELINE = "Majority-class baseline accuracy (evaluation majority)"
BALANCED = "Balanced accuracy for evaluation data"
MCC = "Matthews correlation coefficient (MCC) for evaluation data"


@pytest.mark.parametrize("truth,predictions", [
    (["0"] * 99 + ["1"], ["0"] * 100),
    (["a", "b", "c", "d"], ["a", "b", "c", "d"]),
    (["a", "a", "a", "b", "b", "c"], ["b", "a", "a", "c", "b", "c"]),
    ([0, 0, 1, 1], [0, 2, 1, 2]),  # a predicted-only label creates a zero-support row
    (["only"] * 5, ["only"] * 5),
    ([0, 0, 1, 1], [1, 1, 0, 0]),
])
def test_diagnostics_match_sklearn_for_binary_multiclass_and_constant_predictions(truth, predictions):
    matrix = confusion_matrix(truth, predictions)
    actual = evaluation_diagnostics(matrix)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected_balanced = balanced_accuracy_score(truth, predictions)
    assert actual[BALANCED] == pytest.approx(expected_balanced)
    assert actual[MCC] == pytest.approx(matthews_corrcoef(truth, predictions))
    _, support = np.unique(truth, return_counts=True)
    assert actual[BASELINE] == pytest.approx(support.max() / len(truth))
    np.testing.assert_array_equal(matrix, confusion_matrix(truth, predictions))


def test_real_creditcard_evaluation_exposes_majority_baseline_and_weak_recall():
    actual = evaluation_diagnostics(np.array([[4781, 162], [73, 11]]))
    assert actual[BASELINE] == pytest.approx(4943 / 5027)
    assert actual[BALANCED] == pytest.approx((4781 / 4943 + 11 / 84) / 2)
    assert actual[BASELINE] > (4781 + 11) / 5027
    assert 0.5 < actual[BALANCED] < 0.56
    assert 0.0 < actual[MCC] < 0.1


def test_no_evaluation_rows_produce_no_diagnostics():
    assert evaluation_diagnostics(np.zeros((2, 2))) == {}


@pytest.mark.parametrize("metric,explanation", [
    ("f1_micro", "equals accuracy"),
    ("f1_macro", "Every class contributes equally"),
    ("f1_weighted", "Frequent classes contribute more"),
    ("balanced_accuracy", "precision is not included"),
    ("mcc", "full confusion matrix"),
    ("auc", "cross-validation"),
    (None, "cross-validation"),
])
def test_metric_help_explains_selection_and_averaging(metric, explanation):
    assert explanation in score_metric_help(metric)


@pytest.mark.parametrize("module_name", ["JBGLogger", "JBGStreamedLogger"])
def test_both_loggers_include_diagnostics_with_existing_report_signature(module_name):
    import importlib

    module = importlib.import_module(module_name)
    logger = object.__new__(module.JBGLogger)
    logger._enable_quiet = True
    outputs = {}
    logger.print_progress = lambda **kwargs: None
    logger.display_matrix = lambda title, matrix: outputs.update({title: matrix})
    matrix = np.array([[4781, 162], [73, 11]])
    logger.print_prediction_report(
        accuracy_score=(4781 + 11) / 5027,
        confusion_matrix=matrix,
        class_labels=["0", "1"],
        classification_matrix={"0": {"recall": 4781 / 4943}, "1": {"recall": 11 / 84}},
        sample_rates=None,
    )
    evaluation = outputs["Evaluation information"].iloc[:, 0]
    assert float(evaluation[BASELINE]) == pytest.approx(4943 / 5027)
    assert float(evaluation[BALANCED]) == pytest.approx((4781 / 4943 + 11 / 84) / 2)
    assert float(evaluation[MCC]) == pytest.approx(evaluation_diagnostics(matrix)[MCC])
    assert float(evaluation["Accuracy score for evaluation data"]) == pytest.approx((4781 + 11) / 5027)
    np.testing.assert_array_equal(outputs["Confusion matrix for evaluation data"].to_numpy(), matrix)
