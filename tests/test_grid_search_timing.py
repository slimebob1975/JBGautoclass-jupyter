"""Search budgets, selected-candidate provenance and sklearn scoring compatibility."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import JBGHandler as handler_module
from JBGHandler import ModelHandler, _SpotCheckState
from JBGTrainingTiming import (
    CVTiming, GridSearchEstimate, estimate_grid_search, format_approximate_duration,
    format_actual_duration, format_grid_search_comparison,
)


def test_parallel_budget_includes_scoring_refit_and_one_startup_allowance():
    timing = CVTiming(10, 5, 100.0, 10.0, 230.0)
    estimate = estimate_grid_search(timing, 6, 10, 5)
    # 12 concurrent batches, one larger sequential fit and 10 s observed overhead.
    assert estimate.seconds == pytest.approx(12 * 110 + 100 * 10 / 9 + 10)
    assert estimate.cv_fits == 60
    assert estimate.workers == 5
    assert estimate.overhead_seconds == 10


def test_no_unmeasured_speedup_when_search_has_more_workers_than_cv():
    timing = CVTiming(10, 10, 60, 0, 60)
    assert estimate_grid_search(timing, 20, 10, 32).workers == 10
    assert estimate_grid_search(timing, 20, 10, 32).seconds == pytest.approx(1200 + 60 * 10 / 9)


def test_smaller_search_worker_cap_and_partial_batches():
    timing = CVTiming(5, 5, 10, 2, 12)
    estimate = estimate_grid_search(timing, 3, 5, 2)
    assert estimate.workers == 2
    assert estimate.seconds == pytest.approx(8 * 12 + 12.5)


def test_serial_and_reduced_fold_budget():
    timing = CVTiming(2, 1, 5, 1, 12)
    assert estimate_grid_search(timing, 3, 2, 1).seconds == 46


@pytest.mark.parametrize("results,folds,workers,wall", [
    ({}, 2, 1, 1),
    ({"fit_time": [1], "score_time": [1, 1]}, 2, 1, 1),
    ({"fit_time": [1, float('nan')], "score_time": [1, 1]}, 2, 1, 1),
    ({"fit_time": [1, 1], "score_time": [-1, 1]}, 2, 1, 1),
    ({"fit_time": [0, 0], "score_time": [1, 1]}, 2, 1, 1),
    ({"fit_time": [1, 1], "score_time": [1, 1]}, 2, None, 1),
    ({"fit_time": [1, 1], "score_time": [1, 1]}, 2, 1, float('inf')),
])
def test_invalid_telemetry_has_no_estimate(results, folds, workers, wall):
    assert CVTiming.from_results(results, folds, workers, wall) is None


@pytest.mark.parametrize("timing,combinations,folds,workers", [
    (None, 2, 2, 1),
    (CVTiming(2, 1, 1, 0, 2), 2, 3, 1),
    (CVTiming(2, 1, 1, 0, 2), 0, 2, 1),
    (CVTiming(2, 1, 1, 0, 2), 2, 2, 0),
    (CVTiming(2, 1, float('inf'), 0, 2), 2, 2, 1),
])
def test_unavailable_budget_is_explicit(timing, combinations, folds, workers):
    assert estimate_grid_search(timing, combinations, folds, workers) is None


@pytest.mark.parametrize("seconds,display", [
    (0.19, "~<1 s"), (1.01, "~2 s"), (59, "~59 s"),
    (60, "~1 min"), (3599, "~1 h 0 min"), (9300, "~2 h 35 min"),
])
def test_human_duration(seconds, display):
    assert format_approximate_duration(seconds) == display


def make_handler(workers=1):
    handler = ModelHandler(handler=SimpleNamespace(
        STANDARD_DESIRED_N_JOBS=workers,
        config=SimpleNamespace(debug=False, io=SimpleNamespace(verbose=False)),
        logger=Mock(),
    ))
    handler._resolve_scoring_mechanism_for_target = lambda y: "balanced_accuracy"
    return handler


@pytest.mark.parametrize("workers,sparse", [(1, False), (2, False), (1, True)])
def test_real_sklearn_scores_are_identical_and_timings_are_transient(workers, sparse):
    X, y = make_classification(n_samples=80, n_features=5, random_state=42)
    dh = SimpleNamespace(X_train=pd.DataFrame(X), Y_train=pd.Series(y))
    if sparse:
        dh.X_train = dh.X_train.astype(pd.SparseDtype("float64", fill_value=0))
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    pipeline = Pipeline([("scale", StandardScaler(with_mean=not sparse)), ("LRN", LogisticRegression())])
    handler = make_handler(workers)
    actual = handler.get_cross_val_score(pipeline, dh, cv, SimpleNamespace(fit_params={}))
    expected = cross_val_score(pipeline, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=1)
    np.testing.assert_array_equal(actual, expected)
    measured_pipeline, timing = handler._last_cv_timing
    assert measured_pipeline is pipeline
    assert timing.folds == 4 and timing.workers == workers
    assert timing.mean_fit_seconds > 0 and timing.mean_score_seconds >= 0
    assert not hasattr(pipeline.named_steps["LRN"], "classes_")
    assert not hasattr(pipeline, "_jbg_cv_timing")


def test_estimate_tracks_winner_not_last_candidate():
    handler = make_handler()
    state = _SpotCheckState(best_num_components=5, best_rfe_feature_selection=5)
    winner, loser = object(), object()
    winner_timing = CVTiming(2, 1, 10, 1, 22)
    handler._last_cv_timing = (winner, winner_timing)
    handler._consider_spot_check_candidate(state, winner, None, None, None, 0.9, 0.01, 5, 5, "")
    handler._last_cv_timing = (loser, CVTiming(2, 1, 100, 1, 202))
    handler._consider_spot_check_candidate(state, loser, None, None, None, 0.5, 0.01, 5, 5, "")
    assert state.trained_pipeline is winner
    assert state.best_cv_timing is winner_timing
    handler.model = SimpleNamespace()
    handler.handler.config.update_attributes = Mock()
    handler._apply_spot_check_state(state)
    assert handler._selected_cv_timing is winner_timing


def test_missing_current_measurement_cannot_reuse_previous_candidate():
    handler = make_handler()
    handler._last_cv_timing = (object(), CVTiming(2, 1, 10, 1, 22))
    state = _SpotCheckState(best_num_components=5, best_rfe_feature_selection=5)
    handler._update_spot_check_best_state(state, object(), None, None, None, 0.9, 0.01, 5, 5)
    assert state.best_cv_timing is None


def test_actual_worker_retry_count_is_recorded(monkeypatch):
    handler = make_handler(4)
    monkeypatch.setattr(handler_module.psutil, "cpu_count", lambda logical: 4)
    attempts = []

    def operation(*, n_jobs):
        attempts.append(n_jobs)
        if n_jobs > 1:
            raise MemoryError("worker budget")
        return "ok"

    assert handler.execute_n_job(operation, n_jobs_desired=4) == "ok"
    assert attempts == [4, 2, 1]
    assert handler._last_execution_n_jobs == 1


@pytest.mark.parametrize("has_timing", [True, False])
def test_real_grid_search_logs_estimate_before_fit_without_verbose(monkeypatch, has_timing):
    X, y = make_classification(n_samples=60, n_features=5, random_state=17)
    handler = make_handler()
    handler._selected_cv_timing = CVTiming(3, 1, 1, 0.1, 3.3) if has_timing else None
    original_fit = handler_module.GridSearchCV.fit

    def checked_fit(search, *args, **kwargs):
        messages = [call.args[0] for call in handler.handler.logger.print_info.call_args_list]
        expected = "estimated wall-clock time" if has_timing else "time estimate unavailable"
        assert any(expected in message and "6 CV fits + 1 refit" in message for message in messages)
        assert any("estimate assumptions" in message for message in messages) == has_timing
        return original_fit(search, *args, **kwargs)

    monkeypatch.setattr(handler_module.GridSearchCV, "fit", checked_fit)
    fitted, results = handler.model_parameter_grid_search(
        Pipeline([("LRN", LogisticRegression())]), {"LRN__C": [0.1, 1]}, 3,
        pd.DataFrame(X), pd.Series(y),
    )
    assert hasattr(fitted.named_steps["LRN"], "classes_")
    assert len(results) == 2
    assert handler.handler.logger.print_progress.call_count == int(has_timing)


def test_missing_timing_logs_reason_and_fit_count():
    handler = make_handler()
    handler._report_grid_search_estimate(6, 10, 2)
    message = handler.handler.logger.print_info.call_args.args[0]
    assert "unavailable" in message and "60 CV fits + 1 refit" in message
    handler.handler.logger.print_progress.assert_not_called()


def test_comparison_uses_unrounded_seconds_and_signed_deviation():
    estimate = GridSearchEstimate(601.0, 1, 10, 0, 0)
    message = format_grid_search_comparison(estimate, 480.8)
    assert "estimated ~11 min" in message
    assert "actual/estimate 80.0%" in message
    assert "deviation -20.0%" in message
    assert "unrounded" in message
    assert "deviation +20.0%" in format_grid_search_comparison(estimate, 721.2)


@pytest.mark.parametrize("estimate,actual", [
    (None, 1), (GridSearchEstimate(0, 1, 1, 0, 0), 1),
    (GridSearchEstimate(1, 1, 1, 0, 0), float('nan')),
    (GridSearchEstimate(1, 1, 1, 0, 0), -1),
])
def test_invalid_comparisons_are_unavailable(estimate, actual):
    assert format_grid_search_comparison(estimate, actual) is None


@pytest.mark.parametrize("seconds,display", [(0.1234, "0.12 s"), (589.96, "9 min 50 s"), (3661, "1 h 1 min 1 s")])
def test_actual_duration_is_not_labelled_as_estimated(seconds, display):
    assert format_actual_duration(seconds) == display


def test_failed_grid_search_never_logs_successful_percentage(monkeypatch):
    handler = make_handler()
    handler._selected_cv_timing = CVTiming(2, 1, 1, 0.1, 2.2)
    monkeypatch.setattr(handler_module.GridSearchCV, "fit", Mock(side_effect=ValueError("failed fit")))
    with pytest.raises(handler_module.ModelException):
        handler.model_parameter_grid_search(
            Pipeline([("LRN", LogisticRegression())]), {"LRN__C": [1]}, 2,
            pd.DataFrame({"a": [1, 2, 3, 4]}), pd.Series([0, 1, 0, 1]),
        )
    messages = [call.args[0] for call in handler.handler.logger.print_info.call_args_list]
    assert not any("actual/estimate" in message or "Grid search completed" in message for message in messages)
