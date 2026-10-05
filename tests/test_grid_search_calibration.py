"""Comparable-workload calibration, failure isolation and real search contracts."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

import JBGHandler as handler_module
from JBGHandler import ModelHandler
import JBGTrainingTiming as timing_module
from JBGTrainingTiming import (
    CVTiming, GridSearchEstimate, GridSearchTimingHistory,
    calibrate_grid_search_estimate, format_grid_search_timing_diagnostics,
    grid_search_timing_key,
)


def workload():
    X, y = make_classification(n_samples=60, n_features=5, random_state=17)
    return dict(
        model=Pipeline([("scale", StandardScaler()), ("LRN", LogisticRegression())]),
        search_params={"LRN__C": [0.1, 1]}, scorer="balanced_accuracy",
        X=pd.DataFrame(X), y=pd.Series(y), folds=3, workers=1, cv_workers=1,
        cv_timing=CVTiming(3, 1, 1, 0.1, 3.3),
        source=("server", "catalog", "training_table", "target"),
    )


@pytest.mark.parametrize("ratios,factor", [([4.448], 4.448), ([0.634], 0.634), ([1, 2, 100], 2)])
def test_matching_history_can_raise_or_lower_estimate_using_median(ratios, factor):
    base = GridSearchEstimate(10, 2, 6, 1, 1)
    calibrated = calibrate_grid_search_estimate(base, [
        dict(base_seconds=10, actual_seconds=10 * ratio) for ratio in ratios
    ])
    assert calibrated.seconds == pytest.approx(10 * factor)
    assert calibrated.base_seconds == 10
    assert calibrated.calibration_samples == len(ratios)
    assert calibrated.workers == base.workers and calibrated.cv_fits == base.cv_fits
    assert calibrated.calibration_min == min(ratios)
    assert calibrated.calibration_max == max(ratios)


def test_invalid_samples_do_not_change_base_estimate():
    base = GridSearchEstimate(10, 1, 2, 1, 0)
    invalid = [{}, None, dict(base_seconds=0, actual_seconds=1),
               dict(base_seconds=1, actual_seconds=float("nan")),
               dict(base_seconds=1e-300, actual_seconds=1e300)]
    assert calibrate_grid_search_estimate(base, invalid) is base
    assert calibrate_grid_search_estimate(None, []) is None


@pytest.mark.parametrize("change", [
    "source", "model", "grid", "scorer", "folds", "workers", "rows",
    "columns", "dtypes", "class_counts", "environment", "cv_workers", "timing_cost", "timing_startup",
])
def test_history_is_scoped_to_workload_and_execution_context(change, monkeypatch):
    args = workload()
    original = grid_search_timing_key(**args)
    assert original is not None
    if change == "source":
        args["source"] = ("server", "catalog", "different_table", "target")
    elif change == "model":
        args["model"].set_params(LRN__solver="liblinear")
    elif change == "grid":
        args["search_params"] = {"LRN__C": [1, 10]}
    elif change in ("scorer", "folds", "workers", "cv_workers"):
        args[change] = {"scorer": "accuracy", "folds": 2, "workers": 2, "cv_workers": 2}[change]
    elif change == "rows":
        args["X"], args["y"] = args["X"].iloc[:-1], args["y"].iloc[:-1]
    elif change == "columns":
        args["X"] = args["X"].rename(columns={0: "renamed"})
    elif change == "dtypes":
        args["X"] = args["X"].astype("float32")
    elif change == "class_counts":
        args["y"] = pd.Series([0] * 59 + [1])
    elif change == "timing_cost":
        args["cv_timing"] = CVTiming(3, 1, 32, 0.1, 96.3)
    elif change == "timing_startup":
        args["cv_timing"] = CVTiming(3, 1, 1, 0.1, 100)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", "123")
    assert grid_search_timing_key(**args) != original


def test_signature_ignores_fitted_state_and_row_order_without_fitting_again():
    args = workload()
    original = grid_search_timing_key(**args)
    args["model"].fit(args["X"], args["y"])
    assert grid_search_timing_key(**args) == original
    args["X"], args["y"] = args["X"].iloc[::-1], args["y"].iloc[::-1]
    assert grid_search_timing_key(**args) == original
    assert grid_search_timing_key(**dict(args, source=None)) is None
    assert grid_search_timing_key(**dict(args, y=args["y"].iloc[:-1])) is None


def test_unhashable_custom_objects_only_disable_optional_calibration():
    args = workload()
    args["scorer"] = lambda *args: 1
    assert grid_search_timing_key(**args) is None


def test_dense_and_sparse_inputs_do_not_share_a_timing_profile():
    args = workload()
    args["X"] = args["X"].to_numpy()
    dense_key = grid_search_timing_key(**args)
    args["X"] = csr_matrix(args["X"])
    assert dense_key is not None
    assert grid_search_timing_key(**args) != dense_key


def test_history_survives_reload_and_always_records_uncalibrated_denominator(tmp_path):
    path = tmp_path / "history.json"
    key = grid_search_timing_key(**workload())
    base = GridSearchEstimate(10, 1, 6, 1, 0)
    history = GridSearchTimingHistory(path)
    assert history.record(key, base, 40)
    loaded = GridSearchTimingHistory(path)
    calibrated = calibrate_grid_search_estimate(base, loaded.observations(key))
    assert calibrated.seconds == 40
    assert loaded.record(key, calibrated, 40)
    samples = loaded.observations(key)
    assert [sample["base_seconds"] for sample in samples] == [10, 10]
    assert calibrate_grid_search_estimate(base, samples).seconds == 40
    payload = json.loads(path.read_text())
    assert list(payload) == ["version", "profiles"]
    assert list(payload["profiles"]) == [key]
    assert set(samples[0]) == {"timestamp", "base_seconds", "actual_seconds"}
    assert "training_table" not in path.read_text()
    assert not list(tmp_path.glob("*.tmp"))


def test_history_bounds_age_profile_count_samples_and_discards_invalid_rows(tmp_path, monkeypatch):
    clock = [10_000_000.0]
    monkeypatch.setattr(timing_module.time, "time", lambda: clock[0])
    monkeypatch.setattr(GridSearchTimingHistory, "MAX_PROFILES", 2)
    monkeypatch.setattr(GridSearchTimingHistory, "MAX_SAMPLES", 2)
    history = GridSearchTimingHistory(tmp_path / "history.json")
    base = GridSearchEstimate(1, 1, 2, 1, 0)
    for i in range(3):
        clock[0] += 1
        history.record(str(i) * 40, base, 2)
    assert history.observations("0" * 40) == []
    for i in range(3):
        clock[0] += 1
        history.record("2" * 40, base, i + 1)
    assert len(history.observations("2" * 40)) == 2
    assert not history.record("2" * 40, base, float("nan"))
    assert not history.record("bad-key", base, 2)
    clock[0] += GridSearchTimingHistory.MAX_AGE_SECONDS + 1
    assert history.observations("2" * 40) == []


@pytest.mark.parametrize("payload", ["broken json", '{"version":999}', '{"version":1,"profiles":[]}'])
def test_corrupt_or_unknown_history_is_explicitly_rejected(tmp_path, payload):
    path = tmp_path / "history.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        GridSearchTimingHistory(path).observations("a" * 40)


def handler_for_history(tmp_path):
    handler = ModelHandler(handler=SimpleNamespace(
        STANDARD_DESIRED_N_JOBS=1,
        config=SimpleNamespace(
            debug=False, io=SimpleNamespace(verbose=False), script_path=tmp_path,
            connection=SimpleNamespace(host="server", data_catalog="catalog",
                                       data_table="training_table", class_column="target"),
        ), logger=Mock(),
    ))
    handler._resolve_scoring_mechanism_for_target = lambda y: "balanced_accuracy"
    handler._selected_cv_timing = CVTiming(3, 1, 1, 0.1, 3.3)
    return handler


def test_real_search_records_then_calibrates_without_changing_results_or_fit_count(tmp_path, monkeypatch):
    args = workload()
    handler = handler_for_history(tmp_path)
    fits = []
    original_fit = LogisticRegression.fit

    def fit(model, *args, **kwargs):
        fits.append(1)
        return original_fit(model, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", fit)
    first, first_results = handler.model_parameter_grid_search(
        args["model"], args["search_params"], 3, args["X"], args["y"],
    )
    assert len(fits) == 7  # 2 combinations × 3 folds + exactly one refit.
    handler.handler.logger.reset_mock()
    second, second_results = handler.model_parameter_grid_search(
        args["model"], args["search_params"], 3, args["X"], args["y"],
    )
    assert len(fits) == 14
    np.testing.assert_array_equal(first.predict(args["X"]), second.predict(args["X"]))
    np.testing.assert_array_equal(first_results["mean_test_score"], second_results["mean_test_score"])
    messages = [call.args[0] for call in handler.handler.logger.print_info.call_args_list]
    assert any("from 1 matching completed" in message for message in messages)
    assert any("Grid search measured timing" in message for message in messages)
    assert any("not a confidence interval" in message for message in messages)


def test_failed_search_cannot_teach_history(tmp_path, monkeypatch):
    handler = handler_for_history(tmp_path)
    args = workload()
    monkeypatch.setattr(handler_module.GridSearchCV, "fit", Mock(side_effect=ValueError("failed fit")))
    with pytest.raises(handler_module.ModelException):
        handler.model_parameter_grid_search(args["model"], args["search_params"], 3, args["X"], args["y"])
    assert not (tmp_path / "output" / "grid_search_timing.json").exists()


@pytest.mark.parametrize("failure", ["read", "write"])
def test_optional_history_failure_retains_trained_model(tmp_path, monkeypatch, failure):
    handler = handler_for_history(tmp_path)
    args = workload()
    def fail(*args, **kwargs):
        raise OSError("history unavailable")
    monkeypatch.setattr(GridSearchTimingHistory, "observations" if failure == "read" else "record", fail)
    model, results = handler.model_parameter_grid_search(
        args["model"], args["search_params"], 3, args["X"], args["y"],
    )
    assert len(results) == 2 and hasattr(model.named_steps["LRN"], "classes_")
    assert handler.handler.logger.print_warning.called


def test_measured_diagnostics_separate_refit_without_claiming_startup_is_isolated():
    search = SimpleNamespace(cv_results_={"mean_fit_time": [1, 5], "mean_score_time": [0.2, 0.4]},
                             refit_time_=4)
    text = format_grid_search_timing_diagnostics(search, 20)
    assert "1–5 s/fold" in text and "refit 4 s" in text and "phase 16 s" in text
    assert "not separately measured" in text
    assert format_grid_search_timing_diagnostics(search, 3) is None
    assert format_grid_search_timing_diagnostics(SimpleNamespace(), 10) is None
