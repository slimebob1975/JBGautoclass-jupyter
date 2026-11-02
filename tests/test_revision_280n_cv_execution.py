"""Revision 280n: preserve the CV execution and fallback contract."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import JBGCrossValidationExecution as execution


def make_case(monkeypatch, folds):
    X = object()
    y = np.array([0, 1, 0, 1])
    dh = SimpleNamespace(X_train=X, Y_train=SimpleNamespace(to_numpy=Mock(return_value=y)))
    pipeline = object()
    scorer = object()
    alg = SimpleNamespace(name='LRN')
    logger = SimpleNamespace(print_warning=Mock())
    context = SimpleNamespace(handler=SimpleNamespace(logger=logger),
                              execute_n_job=Mock(return_value={'test_score': np.array([.5, .75])}),
                              _last_execution_n_jobs=99)
    input_data = object()
    prepare_input = Mock(return_value=input_data)
    monkeypatch.setattr(execution, 'prepare_model_input', prepare_input)
    retry = Mock(side_effect=lambda callback, data, **kw: callback(data))
    monkeypatch.setattr(execution, 'call_with_sparse_input_retry', retry)
    profile = Mock(return_value=(pipeline, scorer, False))
    monkeypatch.setattr(execution, 'prepare_runtime_cv', profile)
    return context, dh, pipeline, scorer, alg, logger, input_data, retry, profile


def test_cv_executes_once_with_fold_bound_and_fit_parameters(monkeypatch):
    case = make_case(monkeypatch, [object(), object(), object()])
    context, dh, pipeline, scorer, alg, logger, input_data, retry, profile = case
    params = {'groups': np.array([1, 1, 2, 2])}
    results, profiled, wall = execution.execute_cross_validation(
        context, pipeline, dh, [1, 2, 3], alg, scorer, params,
    )
    assert list(results['test_score']) == [.5, .75]
    assert profiled is False
    assert wall >= 0
    assert context._last_execution_n_jobs is None
    args, kwargs = context.execute_n_job.call_args
    assert args[0] is execution.cross_validate
    assert args[1] is pipeline
    assert args[2] is input_data
    assert kwargs['n_jobs_desired'] == 3
    assert kwargs['params'] is params
    assert kwargs['error_score'] == 'raise'
    assert kwargs['cv'] == [1, 2, 3]
    assert kwargs['scoring'] is scorer
    assert retry.call_args.kwargs['estimator'] is pipeline
    logger.print_warning.assert_not_called()


def test_cv_splitter_uses_get_n_splits(monkeypatch):
    case = make_case(monkeypatch, None)
    context, dh, pipeline, scorer, alg, logger, *_ = case
    folds = SimpleNamespace(get_n_splits=Mock(return_value=4))
    execution.execute_cross_validation(context, pipeline, dh, folds, alg, scorer, {})
    folds.get_n_splits.assert_called_once_with()
    assert context.execute_n_job.call_args.kwargs['n_jobs_desired'] == 4


def test_runtime_profile_fallback_preserves_original_pipeline_and_scorer(monkeypatch):
    case = make_case(monkeypatch, None)
    context, dh, pipeline, scorer, alg, logger, _, _, profile = case
    profile.side_effect = RuntimeError('unsupported profile')
    execution.execute_cross_validation(context, pipeline, dh, [1, 2], alg, scorer, {})
    assert 'unsupported profile' in logger.print_warning.call_args.args[0]
    args = context.execute_n_job.call_args.args
    assert args[1] is pipeline
    assert context.execute_n_job.call_args.kwargs['scoring'] is scorer


def test_profiled_cv_uses_wrapped_pipeline_and_scorer(monkeypatch):
    case = make_case(monkeypatch, None)
    context, dh, pipeline, scorer, alg, logger, _, _, profile = case
    wrapped_pipeline, wrapped_scorer = object(), object()
    profile.return_value = wrapped_pipeline, wrapped_scorer, True
    _, profiled, _ = execution.execute_cross_validation(context, pipeline, dh, [1], alg, scorer, {})
    assert profiled is True
    assert context.execute_n_job.call_args.args[1] is wrapped_pipeline
    assert context.execute_n_job.call_args.kwargs['scoring'] is wrapped_scorer


def test_real_cv_execution_error_propagates(monkeypatch):
    case = make_case(monkeypatch, None)
    context, dh, pipeline, scorer, alg, logger, *_ = case
    context.execute_n_job.side_effect = ValueError('CV failed')
    with pytest.raises(ValueError, match='CV failed'):
        execution.execute_cross_validation(context, pipeline, dh, [1, 2], alg, scorer, {})
