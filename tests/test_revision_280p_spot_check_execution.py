"""280p: behavioral contracts for the extracted candidate execution."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from JBGSpotCheckExecution import evaluate_spot_check_candidate
from JBGMeta import Reduction
from JBGExceptions import ModelException


def _part(name):
    return SimpleNamespace(name=name)


def _context(reduction=Reduction.PCA, skip_reason=None, cv_results=None):
    owner = SimpleNamespace()
    owner.get_preflight_skip_reason = Mock(return_value=skip_reason)
    owner._get_rfe_search_round_limit = Mock(return_value=5)
    owner._summarize_cv_results = Mock(side_effect=lambda x: (x.mean(), x.std()))
    owner.get_components_from_pipeline = Mock(return_value=3)
    owner.calculate_current_features = Mock(side_effect=lambda score, old, n, high, low: (score, high, low))
    owner._consider_spot_check_candidate = Mock(return_value=("", True))
    owner._build_spot_check_result = Mock(return_value=["result"])
    owner._benchmark_candidate_150 = Mock()
    owner._last_cv_diagnostics = None
    owner.handler = SimpleNamespace(logger=SimpleNamespace(print_warning=Mock(), print_progress=Mock()),
                                    config=SimpleNamespace(mode=SimpleNamespace(model_benchmark_150=False)))
    pipeline = object()
    scores = np.array([0.75, 0.85]) if cv_results is None else cv_results
    owner.create_pipeline_and_cv = Mock(return_value=(pipeline, scores, ""))
    dh = SimpleNamespace(X=np.zeros((8, 4)), X_train=np.zeros((8, 4)), Y_train=np.array([0, 1]*4))
    state = SimpleNamespace(trained_pipeline=None, best_oversampler=None, best_undersampler=None)
    args = (owner, dh, _part("PRE"), object(), reduction, object(), _part("ALG"), object(),
            _part("OVER"), _part("UNDER"), object(), state)
    return owner, state, args


def test_preflight_skip_never_runs_cv():
    owner, state, args = _context(skip_reason="not supported")
    result, succeeded = evaluate_spot_check_candidate(*args)
    assert succeeded is False
    assert result == [["result"]]
    owner.create_pipeline_and_cv.assert_not_called()
    assert owner._build_spot_check_result.call_args.kwargs["failure"] == "SKIPPED: not supported"


def test_successful_candidate_preserves_cv_score_and_row():
    owner, state, args = _context()
    result, succeeded = evaluate_spot_check_candidate(*args)
    assert succeeded is True
    assert result == [["result"]]
    assert owner._consider_spot_check_candidate.call_args.kwargs["cv_score"] == pytest.approx(.8)
    assert owner._consider_spot_check_candidate.call_args.kwargs["cv_stdev"] == pytest.approx(.05)
    assert owner.create_pipeline_and_cv.call_count == 1


def test_selected_candidate_preserves_sampler_references():
    owner, state, args = _context()
    def select(**kwargs):
        state.trained_pipeline = kwargs["pipeline"]
        return "", True
    owner._consider_spot_check_candidate.side_effect = select
    evaluate_spot_check_candidate(*args)
    assert state.best_oversampler is args[8]
    assert state.best_undersampler is args[9]


def test_optional_benchmark_only_for_stable_candidate():
    owner, state, args = _context()
    owner.handler.config.mode.model_benchmark_150 = True
    evaluate_spot_check_candidate(*args)
    owner._benchmark_candidate_150.assert_called_once()
    owner._benchmark_candidate_150.reset_mock()
    owner._consider_spot_check_candidate.return_value = ("failed", False)
    evaluate_spot_check_candidate(*args)
    owner._benchmark_candidate_150.assert_not_called()


def test_model_exception_logs_and_aborts_without_result_row():
    owner, state, args = _context()
    owner.create_pipeline_and_cv.side_effect = ModelException("deliberate")
    result, succeeded = evaluate_spot_check_candidate(*args)
    assert result == []
    assert succeeded is False
    owner.handler.logger.print_warning.assert_called_once()


def test_rfe_progress_still_precedes_cv():
    owner, state, args = _context(reduction=Reduction.RFE)
    owner._get_rfe_search_round_limit.return_value = 1
    events = []
    owner.handler.logger.print_progress.side_effect = lambda **kwargs: events.append("progress")
    owner.create_pipeline_and_cv.side_effect = lambda *a: (events.append("cv") or object(), np.array([.7,.8]), "")
    evaluate_spot_check_candidate(*args)
    assert events[:2] == ["progress", "cv"]
    assert owner.handler.logger.print_warning.call_count >= 1
