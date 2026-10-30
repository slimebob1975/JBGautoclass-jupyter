"""Target CV counts and stable splits without extra fits or holdout scoring."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_validate

from Helpers import build_model_performance_matrix
from JBGCVReporting import (CV_TARGET_COLUMNS, TargetCVScorer, resolve_cv_target,
                           summarize_target_cv, partition_signature)
from JBGHandler import DatasetHandler
from test_sampling_search import data, handler


def test_explicit_numeric_and_string_targets_and_stable_minority_tie():
    assert resolve_cv_target([0, 1, 1], "0") == (0, "")
    assert resolve_cv_target(["Korrekt", "Inkorrekt", "Korrekt"], "Inkorrekt") == ("Inkorrekt", "")
    assert resolve_cv_target(["b", "a"]) == resolve_cv_target(["a", "b"]) == ("a", "")
    assert resolve_cv_target(["a", "b", "b"])[0] == "a"
    target, error = resolve_cv_target(["a", "b"], "missing")
    assert target is None and "absent" in error


def test_pooled_counts_match_fold_predictions_without_extra_fits(monkeypatch):
    from JBGMeta import Reduction, Algorithm, Preprocess, Oversampling, Undersampling
    dh, mh = data(), handler(over=("NOG",), under=("NUG",))
    dh.Y_train = dh.Y_train.map({0: "Korrekt", 1: "Inkorrekt"})
    mh.handler.config.mode.dark_number_target = "Inkorrekt"
    pipe = mh.get_pipeline(Reduction.NOR, None, Algorithm.LRN, LogisticRegression(max_iter=300),
        Preprocess.STA, Preprocess.STA.call_preprocess(), Oversampling.NOG, Undersampling.NUG, 5)
    cv, expected = StratifiedKFold(3, shuffle=True, random_state=1), np.empty(len(dh.Y_train), dtype=object)
    for train, test in cv.split(dh.X_train, dh.Y_train):
        expected[test] = clone(pipe).fit(dh.X_train.iloc[train], dh.Y_train.iloc[train]).predict(dh.X_train.iloc[test])
    calls, original = [], LogisticRegression.fit
    def tracked(self, X, y, *args, **kwargs):
        calls.append(len(y))
        return original(self, X, y, *args, **kwargs)
    monkeypatch.setattr(LogisticRegression, "fit", tracked)
    scores = mh.get_cross_val_score(pipe, dh, cv, Algorithm.LRN)
    assert len(calls) == 3
    report = mh._last_cv_diagnostics[1]
    p, r, f, _ = precision_recall_fscore_support(dh.Y_train, expected, labels=["Inkorrekt"], zero_division=0)
    np.testing.assert_allclose([report[k] for k in CV_TARGET_COLUMNS[1:4]], [p[0], r[0], f[0]])
    assert report["CV target"] == "Inkorrekt"
    assert report["CV predicted positives"] == np.sum(expected == "Inkorrekt")
    assert report["CV target support"] == np.sum(dh.Y_train == "Inkorrekt")
    assert report["CV validation rows"] == len(dh.Y_train)
    reference = cross_validate(pipe, dh.X_train, dh.Y_train, cv=cv, scoring="matthews_corrcoef")
    np.testing.assert_array_equal(scores, reference["test_score"])


def test_no_positive_predictions_report_zero_metrics_and_original_support():
    X, y = np.zeros((12, 1)), np.array([0] * 9 + [1] * 3)
    result = cross_validate(DummyClassifier(strategy="most_frequent"), X, y,
        scoring=TargetCVScorer("matthews_corrcoef", 1), cv=StratifiedKFold(3))
    report = summarize_target_cv(result, 1)
    assert [report[k] for k in CV_TARGET_COLUMNS[1:5]] == [0.0, 0.0, 0.0, 0]
    assert report["CV target support"] == 3 and report["CV validation rows"] == 12


def test_diagnostic_failure_preserves_score_but_selection_failure_propagates():
    class Broken:
        def predict(self, X):
            raise ValueError("diagnostic predict failed")
    result = TargetCVScorer(lambda *args: 0.7, 1)(Broken(), None, [0, 1])
    assert result["score"] == 0.7 and result["target_available"] == 0
    error = ValueError("selection failed")
    def fail(*args):
        raise error
    with pytest.raises(ValueError) as caught:
        TargetCVScorer(fail, 1)(Broken(), None, [0, 1])
    assert caught.value is error


@pytest.mark.filterwarnings("ignore:lbfgs failed to converge.*:sklearn.exceptions.ConvergenceWarning")
def test_neural_nystroem_runtime_and_target_diagnostics_share_the_same_fits():
    from test_nystroem_runtime import pipeline, data as nystroem_data
    from JBGNystroemRuntime import prepare_runtime_cv
    X, y, cv = nystroem_data()
    original = pipeline()
    profiled, scorer, enabled = prepare_runtime_cv(original, TargetCVScorer("roc_auc", 1), "MLPC")
    assert enabled
    result = cross_validate(profiled, X, y, scoring=scorer, cv=cv, n_jobs=2, error_score="raise")
    reference = cross_validate(original, X, y, scoring="roc_auc", cv=cv, n_jobs=2, error_score="raise")
    np.testing.assert_array_equal(result["test_score"], reference["test_score"])
    assert np.all(result["test_runtime_available"] == 1)
    assert np.all(result["test_target_available"] == 1)
    assert summarize_target_cv(result, 1)["CV validation rows"] == len(y)


def test_invalid_target_and_failed_pair_do_not_reuse_diagnostics(tmp_path):
    mh = handler(over=("NOG", "SME"), under=("NUG",))
    mh.handler.config.mode.dark_number_target = "absent"
    model = mh.spot_check_machine_learning_models(data(minority=2), str(tmp_path / "cv.csv"), k=2)
    table = build_model_performance_matrix(mh.handler.logger.print_test_performance.call_args.args[0])
    assert model.pipeline is not None
    good, bad = table.loc[table.Oversampling.eq("NOG")].iloc[0], table.loc[table.Oversampling.eq("SME")].iloc[0]
    assert "absent" in good["CV diagnostic status"] and good.Exception == ""
    assert np.isnan(good["CV target precision"])
    assert "at least two minority" in bad.Exception and np.isnan(bad["CV target recall"])


def test_every_sampling_pair_reports_original_validation_support(tmp_path):
    mh, dh = handler(), data()
    mh.spot_check_machine_learning_models(dh, str(tmp_path / "cv.csv"), k=3)
    table = build_model_performance_matrix(mh.handler.logger.print_test_performance.call_args.args[0])
    assert len(table) == 4 and table["CV diagnostic status"].str.startswith("OK").all()
    assert table["CV validation rows"].eq(len(dh.Y_train)).all()
    assert table["CV target support"].eq(sum(dh.Y_train == 1)).all()


def test_grid_reports_every_candidate_without_extra_fits_and_exports(monkeypatch, tmp_path):
    from test_final_search import handler as final_handler, data as final_data, pipe
    from sklearn.ensemble import ExtraTreesClassifier
    mh, (X, y) = final_handler(), final_data()
    calls, original = [], ExtraTreesClassifier.fit
    def tracked(self, X, y, *args, **kwargs):
        calls.append(len(y))
        return original(self, X, y, *args, **kwargs)
    monkeypatch.setattr(ExtraTreesClassifier, "fit", tracked)
    _, results = mh.model_parameter_grid_search(pipe(n_estimators=7), {"ETC__max_depth": [None, 3]}, 3, X, y)
    assert len(calls) == 7
    assert results["CV validation rows"].eq(len(y)).all()
    assert results["CV target support"].eq(sum(y == 1)).all()
    assert results["CV diagnostic status"].str.startswith("OK").all()
    assert "split0_test_target_tp" in results
    mh._export_final_search_results(results, tmp_path / "cv.csv")
    saved = pd.read_csv(tmp_path / "cv_grid_search.csv", sep=";", decimal=",")
    np.testing.assert_allclose(saved["CV target recall"], results["CV target recall"])
    assert len(calls) == 7


def test_optional_target_report_failure_keeps_search_summary_and_baseline_flags(monkeypatch):
    import JBGHandler
    from test_final_search import handler as final_handler, data as final_data, pipe
    mh, (X, y) = final_handler(), final_data()
    def fail(*args):
        raise RuntimeError("optional reporting failure")
    monkeypatch.setattr(JBGHandler, "annotate_grid_target_cv", fail)
    model, results = mh.model_parameter_grid_search(pipe(n_estimators=7), {"ETC__max_depth": [None, 3]}, 3, X, y)
    assert model is not None and results.is_baseline.sum() == 1 and results.is_selected.sum() == 1
    assert mh._last_grid_search_summary["best_cv_mean"] == results.loc[results.is_selected, "mean_test_score"].iloc[0]
    assert "Unavailable" in mh._last_grid_search_summary["target_cv"]["CV diagnostic status"]


def partitioned(frame, train=True):
    config = SimpleNamespace(get_id_column_name=lambda: "row_key",
        get_class_column_name=lambda: "label", should_train=lambda: train,
        get_test_size=lambda: 0.2, should_calculate_feature_importance=lambda: False,
        mode=SimpleNamespace(holdout_split_mode="stratified_rows", grouped_cross_validation=False))
    dh = DatasetHandler(SimpleNamespace(config=config, logger=Mock()))
    ordered = dh.shuffle_dataset(frame)
    dh.dataset, dh.keys = dh.split_keys_from_dataset(ordered, "row_key")
    dh.separate_dataset()
    if train:
        dh.split_dataset_for_training_and_validation()
    return dh


def test_partition_and_cv_membership_independent_of_fetch_order_and_global_rng():
    frame = pd.DataFrame({"row_key": np.arange(120) + 1000, "value": np.arange(120),
                          "label": ["Inkorrekt"] * 20 + ["Korrekt"] * 95 + [None, "", " ", None, ""]})
    first = partitioned(frame.sample(frac=1, random_state=3))
    np.random.rand(40)
    second = partitioned(frame.sample(frac=1, random_state=29))
    for name in ("X_train", "X_validation", "Y_train", "Y_validation", "X_prediction"):
        a, b = getattr(first, name), getattr(second, name)
        (pd.testing.assert_frame_equal if isinstance(a, pd.DataFrame) else pd.testing.assert_series_equal)(a, b)
    assert len(first.X_prediction) == 5 and len(first.X) == 115
    assert first.unpredicted_keys.tolist() == first.X_prediction.index.tolist()
    cv = StratifiedKFold(3, shuffle=True, random_state=1)
    folds = lambda dh: [dh.X_train.index[test].tolist() for _, test in cv.split(dh.X_train, dh.Y_train)]
    assert folds(first) == folds(second)
    assert partition_signature(first.X_validation.index, first.Y_validation) == partition_signature(second.X_validation.index, second.Y_validation)
    assert "ID/label SHA256=" in first.handler.logger.print_privacy_safe.call_args.args[0]


def test_signature_detects_changed_membership_or_labels():
    original = partition_signature([11, 12], ["a", "b"])
    assert original == partition_signature([12, 11], ["b", "a"])
    assert original != partition_signature([11, 13], ["a", "b"])
    assert original != partition_signature([11, 12], ["b", "b"])


def test_prediction_only_rows_and_duplicate_id_rejection():
    frame = pd.DataFrame({"row_key": [3, 1, 2], "value": [30, 10, 20], "label": [None, "", " "]})
    first = partitioned(frame, train=False)
    assert first.X.empty and first.X_prediction.value.tolist() == [10, 20, 30]
    duplicate = pd.DataFrame({"row_key": [1, 1], "value": [3, 4], "label": ["a", "b"]})
    from JBGExceptions import DatasetException
    with pytest.raises(DatasetException, match="not unique"):
        partitioned(duplicate)
