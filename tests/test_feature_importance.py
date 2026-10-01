"""Original-input permutation, read-only fitting contract and optional task lifecycle."""
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from JBGFeatureImportance import compute_feature_importance, FittedInputClassifier
from JBGTaskRunner import TaskRunner, get_tasks
from JBGTransformers import TextDataToNumbersConverter
from JBGHandler import DatasetHandler
from Helpers import prepare_estimator_input


def numeric_model():
    rng = np.random.RandomState(12)
    X = pd.DataFrame({"Signal": rng.normal(size=250), "Constant": np.zeros(250)})
    y = pd.Series(np.where(X.Signal > 0, "Ja", "Nej"), index=X.index)
    pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())]).fit(X.iloc[:150], y.iloc[:150])
    return pipeline, X.iloc[150:].copy(), y.iloc[150:].copy()


def test_signal_ranks_first_repeats_match_summary_and_state_stays_fixed():
    pipeline, X, y = numeric_model()
    original = X.copy(deep=True)
    model_before = pickle.dumps(pipeline)
    progress = []
    report = compute_feature_importance(pipeline, X, y, make_scorer(matthews_corrcoef), repeats=5,
                                        progress=lambda completed, total: progress.append((completed, total)))
    assert report.table.Feature.tolist() == ["Signal", "Constant"]
    assert report.table.iloc[0]["Mean score decrease"] > 0.5
    assert report.table.iloc[1]["Mean score decrease"] == 0
    values = report.table[[f"Repeat {i}" for i in range(1, 6)]].to_numpy()
    np.testing.assert_allclose(values.mean(axis=1), report.table["Mean score decrease"])
    np.testing.assert_allclose(values.std(axis=1), report.table["Std. across repeats"])
    assert progress == [(i, 11) for i in range(1, 12)]
    pd.testing.assert_frame_equal(X, original)
    assert pickle.dumps(pipeline) == model_before
    assert report.rows == len(y) == 100


def test_deterministic_repeats_and_no_refit(monkeypatch):
    pipeline, X, y = numeric_model()
    monkeypatch.setattr(pipeline, "fit", Mock(side_effect=AssertionError("unexpected refit")))
    first = compute_feature_importance(pipeline, X, y, "balanced_accuracy", repeats=3)
    second = compute_feature_importance(pipeline, X, y, "balanced_accuracy", repeats=3)
    pd.testing.assert_frame_equal(first.table, second.table)
    pipeline.fit.assert_not_called()
    with pytest.raises(RuntimeError, match="fitting is disabled"):
        FittedInputClassifier(pipeline).fit(X, y)


def test_decision_function_scorer_with_nonprobabilistic_classifier():
    _, X, y = numeric_model()
    pipeline = Pipeline([("model", LinearSVC())]).fit(X, y)
    adapter = FittedInputClassifier(pipeline)
    assert not hasattr(adapter, "predict_proba")
    assert hasattr(adapter, "decision_function")
    report = compute_feature_importance(pipeline, X, y, "roc_auc", repeats=2)
    assert report.baseline_score > 0.95


def test_real_text_category_converter_preserves_original_names_and_sparse_input(monkeypatch):
    rng = np.random.RandomState(5)
    y = pd.Series(["Ja", "Nej"] * 60)
    X = pd.DataFrame({"Belopp": rng.uniform(size=120),
                      "Comment": ["red apple" if v == "Ja" else "blue pear" for v in y],
                      "Region": ["north", "south"] * 60})
    converter = TextDataToNumbersConverter(text_columns=["Comment", "Region"],
        category_columns=["Region"], use_categorization=False, stop_words=False, use_encryption=True)
    converter.fit(X.iloc[:80])
    converted = converter.transform(X.iloc[:80])
    from scipy import sparse
    assert sparse.issparse(prepare_estimator_input(converted))
    pipeline = Pipeline([("model", LogisticRegression())]).fit(prepare_estimator_input(converted), y.iloc[:80])
    heldout = X.iloc[80:].copy()
    baseline_predictions = pipeline.predict(prepare_estimator_input(converter.transform(heldout)))
    np.testing.assert_array_equal(FittedInputClassifier(pipeline, converter).predict(heldout), baseline_predictions)
    monkeypatch.setattr(converter, "fit", Mock(side_effect=AssertionError("converter must stay fitted")))
    report = compute_feature_importance(pipeline, heldout, y.iloc[80:], "balanced_accuracy", converter=converter, repeats=2)
    assert set(report.table.Feature) == {"Belopp", "Comment", "Region"}
    assert len(report.table) == 3
    assert report.baseline_score == 1
    converter.fit.assert_not_called()


@pytest.mark.parametrize("bad", [0, 1, 31, True, 2.5])
def test_invalid_repeats_are_rejected(bad):
    pipeline, X, y = numeric_model()
    with pytest.raises(ValueError, match="repeats"):
        compute_feature_importance(pipeline, X, y, "accuracy", repeats=bad)


def test_holdout_alignment_and_duplicate_names_are_checked():
    pipeline, X, y = numeric_model()
    with pytest.raises(ValueError, match="aligned"):
        compute_feature_importance(pipeline, X, y.iloc[::-1], "accuracy")
    X.columns = ["duplicate", "duplicate"]
    with pytest.raises(ValueError, match="unique"):
        compute_feature_importance(pipeline, X, y, "accuracy")


def test_nonfinite_scorer_does_not_produce_a_report():
    pipeline, X, y = numeric_model()
    with pytest.raises(ValueError, match="nonfinite"):
        compute_feature_importance(pipeline, X, y, lambda estimator, features, labels: np.nan)


def task_config(enabled=True, training=True):
    return SimpleNamespace(should_train=lambda: training, should_predict=lambda: not training,
        get_text_column_names=lambda: [], should_calculate_feature_importance=lambda: enabled and training)


def test_analysis_is_opt_in_and_between_evaluation_and_retraining():
    tasks = get_tasks(task_config())
    assert tasks.index("evaluate_model") < tasks.index("feature_importance") < tasks.index("retrain_model")
    for config, suite in [(task_config(False), False), (task_config(True, False), False), (task_config(), True)]:
        assert "feature_importance" not in get_tasks(config, regression_suite=suite)


def make_analysis_task(tmp_path):
    pipeline, X, y = numeric_model()
    runner = TaskRunner.__new__(TaskRunner)
    runner.logger = Mock()
    runner.config = SimpleNamespace(should_calculate_feature_importance=lambda: True,
        get_feature_importance_repeats=lambda: 2,
        get_output_filepath=lambda kind: tmp_path / f"{kind}.csv",
        mode=SimpleNamespace(scoring=SimpleNamespace(full_name="MCC")))
    runner.dh = SimpleNamespace(X_validation_original=X, X_validation=X.copy(), Y_validation=y)
    runner.mh = SimpleNamespace(model=SimpleNamespace(pipeline=pipeline, text_converter=None, get_name=lambda: "LRN"),
        _resolve_scoring_mechanism_for_target=lambda labels: make_scorer(matthews_corrcoef))
    return runner


def test_optional_task_exports_all_repeats_and_details_then_releases_snapshot(tmp_path):
    runner = make_analysis_task(tmp_path)
    pipeline_before = pickle.dumps(runner.mh.model.pipeline)
    assert runner.feature_importance__task() == {}
    summary = pd.read_csv(tmp_path / 'feature_importance.csv', sep=';', decimal=',', index_col=0)
    assert len(summary) == 2
    assert {'Feature', 'Mean score decrease', 'Std. across repeats', 'Repeat 1', 'Repeat 2'} <= set(summary.columns)
    assert (tmp_path / 'feature_importance_details.csv').exists()
    assert runner.dh.X_validation_original is None
    assert pickle.dumps(runner.mh.model.pipeline) == pipeline_before
    runner.logger.end_inline_progress.assert_called_once_with("feature_importance", set_100=True)
    runner.logger.print_warning.assert_not_called()


def test_optional_failure_warns_and_returns_without_aborting_training(tmp_path):
    runner = make_analysis_task(tmp_path)
    runner.mh._resolve_scoring_mechanism_for_target = Mock(side_effect=ValueError("unavailable scorer"))
    assert runner.feature_importance__task() == {}
    assert "unavailable scorer" in runner.logger.print_warning.call_args.args[0]
    runner.logger.end_inline_progress.assert_called_once_with("feature_importance", set_100=False)
    assert runner.dh.X_validation_original is None
    assert not (tmp_path / 'feature_importance.csv').exists()


def test_disabled_task_does_no_analysis_or_output(tmp_path):
    runner = make_analysis_task(tmp_path)
    runner.config.should_calculate_feature_importance = lambda: False
    original = runner.dh.X_validation_original
    assert runner.feature_importance__task() == {}
    assert runner.dh.X_validation_original is original
    assert not runner.logger.mock_calls


@pytest.mark.parametrize("enabled", [False, True])
def test_original_holdout_is_captured_only_when_enabled_and_keeps_label_alignment(enabled):
    dh = DatasetHandler.__new__(DatasetHandler)
    dh.X = pd.DataFrame({"Region": ["north", "south"] * 20, "Belopp": np.arange(40)}, index=np.arange(100, 140))
    dh.Y = pd.Series(["Ja", "Nej"] * 20, index=dh.X.index)
    dh.handler = SimpleNamespace(logger=Mock(), config=SimpleNamespace(
        should_train=lambda: True, get_test_size=lambda: 0.2,
        should_calculate_feature_importance=lambda: enabled,
    ))
    assert dh.split_dataset_for_training_and_validation() is True
    if enabled:
        pd.testing.assert_frame_equal(dh.X_validation_original, dh.X_validation)
        assert dh.X_validation_original.index.equals(dh.Y_validation.index)
        assert dh.X_validation_original is not dh.X_validation
    else:
        assert dh.X_validation_original is None
