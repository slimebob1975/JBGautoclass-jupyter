"""The runtime investigation must preserve CV scores and model-selection state."""
from types import SimpleNamespace
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

import JBGHandler
import JBGNystroemRuntime as runtime
from JBGHandler import ModelHandler


def pipeline(algorithm='MLPC'):
    mlp = MLPClassifier(hidden_layer_sizes=(6,), solver='lbfgs', max_iter=15, random_state=5)
    bases = [('mlpc', mlp), ('rfcl', RandomForestClassifier(n_estimators=3, random_state=6)),
             ('abc', AdaBoostClassifier(n_estimators=3, random_state=7))]
    classifier = {
        'MLPC': mlp, 'MLP2': mlp,
        'FUTV': VotingClassifier(estimators=bases, voting='soft'),
        'FUTS': StackingClassifier(estimators=bases, final_estimator=LogisticRegression(), cv=3),
    }[algorithm]
    return Pipeline([('MAX', MaxAbsScaler()), ('NYS', Nystroem(
        n_components=12, gamma=0.2, random_state=42)), (algorithm, classifier)])


def data():
    X, y = make_classification(n_samples=90, n_features=6, random_state=42)
    return X, y, StratifiedKFold(3, shuffle=True, random_state=7)


@pytest.mark.filterwarnings('ignore:lbfgs failed to converge.*:sklearn.exceptions.ConvergenceWarning')
@pytest.mark.parametrize('algorithm', ['MLPC', 'MLP2', 'FUTV', 'FUTS'])
@pytest.mark.parametrize('workers,use_sparse', [(1, False), (2, True)])
def test_real_cv_scores_and_parameters_are_preserved(algorithm, workers, use_sparse):
    original = pipeline(algorithm)
    X, y, cv = data()
    if use_sparse:
        X = sparse.csr_matrix(X)
    cv_pipe, scorer, enabled = runtime.prepare_runtime_cv(original, 'roc_auc', algorithm)
    assert enabled and cv_pipe is not original
    assert type(original.named_steps['NYS']) is Nystroem
    assert isinstance(cv_pipe.named_steps['NYS'], runtime.TimedNystroem)
    assert cv_pipe.named_steps['NYS'].get_params() == original.named_steps['NYS'].get_params()
    assert clone(cv_pipe).named_steps['NYS'].get_params() == original.named_steps['NYS'].get_params()
    expected = cross_validate(original, X, y, cv=cv, scoring='roc_auc', n_jobs=workers, error_score='raise')
    actual = cross_validate(cv_pipe, X, y, cv=cv, scoring=scorer, n_jobs=workers, error_score='raise')
    np.testing.assert_array_equal(actual['test_score'], expected['test_score'])
    assert not hasattr(original.named_steps['NYS'], 'components_')
    assert 'estimator' not in actual
    np.testing.assert_array_equal(actual['test_runtime_available'], [1, 1, 1])
    np.testing.assert_array_equal(actual['test_nys_training_rows'], [60, 60, 60])
    np.testing.assert_array_equal(actual['test_nys_output_features'], [12, 12, 12])
    np.testing.assert_array_equal(actual['test_nys_output_bytes'], [60 * 12 * 8] * 3)
    np.testing.assert_array_equal(actual['test_mlp_count'], [1, 1, 1])
    assert np.all(actual['test_mlp_iterations_max'] <= 15)
    assert np.all(actual['test_nys_fit_transform_seconds'] > 0)
    assert np.all(actual['test_nys_fit_transform_seconds'] <= actual['fit_time'])
    assert np.all(actual['test_nys_score_transform_seconds'] >= 0)
    assert np.all(actual['test_native_threads_max'] >= 1)
    rows = runtime.build_runtime_rows(original, actual, workers, 0.123)
    assert all(row['CV wall seconds'] == 0.123 for row in rows)
    summary = runtime.format_runtime_summary(rows)
    assert 'other pipeline fit work' in summary
    assert 'not additive wall time' in summary
    if algorithm == 'FUTS':
        assert rows[0]['Ensemble internal CV'] == 3
        assert 'discarded internal fit iterations are not measured' in summary


@pytest.mark.filterwarnings('ignore:lbfgs failed to converge.*:sklearn.exceptions.ConvergenceWarning')
@pytest.mark.parametrize('algorithm,expected_fits', [('MLPC', 3), ('FUTV', 3), ('FUTS', 12)])
def test_profile_does_not_add_mlp_fits_or_scorer_calls(monkeypatch, algorithm, expected_fits):
    X, y, cv = data()
    original_fit, fits, scores = MLPClassifier.fit, [], []
    def fit(self, X, y, *args, **kwargs):
        fits.append(X.shape[0])
        return original_fit(self, X, y, *args, **kwargs)
    def score(estimator, X, y):
        scores.append(X.shape[0])
        return get_scorer('accuracy')(estimator, X, y)
    monkeypatch.setattr(MLPClassifier, 'fit', fit)
    cv_pipe, scorer, enabled = runtime.prepare_runtime_cv(pipeline(algorithm), score, algorithm)
    cross_validate(cv_pipe, X, y, cv=cv, scoring=scorer, n_jobs=1, error_score='raise')
    assert len(fits) == expected_fits and scores == [30, 30, 30]


def test_other_algorithms_and_reductions_keep_original_cv_objects():
    original, scorer = pipeline(), get_scorer('accuracy')
    for name, pipe in [('LRN', original), ('MLPC', Pipeline(original.steps[:1] + original.steps[2:]))]:
        actual, actual_scorer, enabled = runtime.prepare_runtime_cv(pipe, scorer, name)
        assert actual is pipe and actual_scorer is scorer and not enabled


def test_custom_nystroem_and_pandas_output_are_not_replaced():
    class CustomNystroem(Nystroem):
        pass
    for nys in [CustomNystroem(n_components=12), Nystroem(n_components=12).set_output(transform='pandas')]:
        original = pipeline().set_params(NYS=nys)
        actual, _, enabled = runtime.prepare_runtime_cv(original, 'accuracy', 'MLPC')
        assert actual is original and not enabled
    with config_context(transform_output='pandas'):
        original = pipeline()
        actual, _, enabled = runtime.prepare_runtime_cv(original, 'accuracy', 'MLPC')
        assert actual is original and not enabled
    original = pipeline().set_params(memory='existing-cache')
    actual, _, enabled = runtime.prepare_runtime_cv(original, 'accuracy', 'MLPC')
    assert actual is original and not enabled


@pytest.mark.filterwarnings('ignore:lbfgs failed to converge.*:sklearn.exceptions.ConvergenceWarning')
def test_final_pipeline_roundtrip_contains_plain_nystroem(tmp_path):
    X, y, cv = data()
    original = pipeline()
    profiled, scorer, _ = runtime.prepare_runtime_cv(original, 'accuracy', 'MLPC')
    cross_validate(profiled, X, y, cv=cv, scoring=scorer, n_jobs=1, error_score='raise')
    original.fit(X, y)
    path = tmp_path / 'model.sav'
    joblib.dump(original, path)
    loaded = joblib.load(path)
    assert type(loaded.named_steps['NYS']) is Nystroem
    assert not hasattr(loaded.named_steps['NYS'], 'runtime_fit_transform_seconds_')
    np.testing.assert_array_equal(loaded.predict_proba(X), original.predict_proba(X))


def test_optional_metric_failure_preserves_score_without_a_second_score_call(monkeypatch):
    expected, calls = 0.75, []
    def score(*args):
        calls.append(1)
        return expected
    def broken(*args):
        raise RuntimeError('diagnostics only')
    monkeypatch.setattr(runtime, '_fold_metrics', broken)
    result = runtime.RuntimeScorer(score)(object(), object(), object())
    assert result['score'] == expected and result['runtime_available'] == 0
    assert calls == [1]
    assert all(np.isnan(result[key]) for key in runtime.METRICS)


def test_true_score_error_propagates_unchanged():
    error = TypeError('scorer bug')
    def fail(*args):
        raise error
    with pytest.raises(TypeError) as caught:
        runtime.RuntimeScorer(fail)(object(), object(), object())
    assert caught.value is error


@pytest.mark.filterwarnings('ignore:lbfgs failed to converge.*:sklearn.exceptions.ConvergenceWarning')
def test_handler_retains_original_pipeline_timing_provenance(default_model_handler):
    mh = default_model_handler
    X, y, cv = data()
    dh = SimpleNamespace(X_train=pd.DataFrame(X), Y_train=pd.Series(y))
    original = pipeline()
    result = mh.get_cross_val_score(original, dh, cv, SimpleNamespace(name='MLPC', fit_params={}))
    assert len(result) == 3
    assert mh._last_cv_timing[0] is original
    assert len(mh._nystroem_runtime_rows) == 3
    assert type(original.named_steps['NYS']) is Nystroem


def test_failed_cv_does_not_retry_or_create_success_diagnostics(default_model_handler):
    mh, calls, error = default_model_handler, [], ValueError('model failed')
    def fail(*args, **kwargs):
        calls.append(1)
        raise error
    mh.execute_n_job = fail
    X, y, cv = data()
    with pytest.raises(ValueError) as caught:
        mh.get_cross_val_score(pipeline(), SimpleNamespace(X_train=pd.DataFrame(X), Y_train=pd.Series(y)),
                               cv, SimpleNamespace(name='MLPC', fit_params={}))
    assert caught.value is error and calls == [1]
    assert mh._last_cv_timing is None
    assert not getattr(mh, '_nystroem_runtime_rows', [])


def test_prepare_failure_does_not_repeat_cv(default_model_handler, monkeypatch):
    mh, calls = default_model_handler, []
    def broken(*args):
        raise RuntimeError('preparation failed')
    def cv_call(*args, **kwargs):
        calls.append((args[1], kwargs['scoring']))
        mh._last_execution_n_jobs = 1
        return {'test_score': np.array([0.5, 0.75, 0.6]), 'fit_time': [1, 1, 1], 'score_time': [0.1] * 3}
    monkeypatch.setattr(JBGHandler, 'prepare_runtime_cv', broken)
    mh.execute_n_job = cv_call
    X, y, cv = data()
    original = pipeline()
    mh.get_cross_val_score(original, SimpleNamespace(X_train=pd.DataFrame(X), Y_train=pd.Series(y)),
                           cv, SimpleNamespace(name='MLPC', fit_params={}))
    assert calls == [(original, 'accuracy')]


def test_report_failure_keeps_successful_cv_score(default_model_handler, monkeypatch):
    mh = default_model_handler
    results = {'test_score': np.array([0.5, 0.75, 0.6]), 'fit_time': [1] * 3, 'score_time': [0.1] * 3}
    def cv_call(*args, **kwargs):
        mh._last_execution_n_jobs = 1
        return results
    def fail(*args):
        raise ValueError('diagnostics failed')
    mh.execute_n_job = cv_call
    monkeypatch.setattr(JBGHandler, 'build_runtime_rows', fail)
    X, y, cv = data()
    actual = mh.get_cross_val_score(pipeline(), SimpleNamespace(X_train=pd.DataFrame(X), Y_train=pd.Series(y)),
                                    cv, SimpleNamespace(name='MLPC', fit_params={}))
    np.testing.assert_array_equal(actual, results['test_score'])
    assert mh._last_cv_timing[1] is not None


def test_spot_check_clears_previous_runtime_rows(default_model_handler):
    mh = default_model_handler
    mh._nystroem_runtime_rows = [{'Pipeline': 'old run'}]
    mh.handler.config.get_callable_reductions = lambda *args: []
    mh.handler.config.get_callable_algorithms = lambda **kwargs: []
    mh.handler.config.get_max_iterations = lambda: 15
    mh.handler.config.get_callable_preprocessors = lambda: []
    mh.handler.config.mode = SimpleNamespace(oversampler=None, undersampler=None)
    mh.handler.logger = Mock()
    mh._create_spot_check_kfold = lambda k: StratifiedKFold(k)
    X, y, _ = data()
    dh = SimpleNamespace(X=pd.DataFrame(X), X_train=pd.DataFrame(X))
    from JBGExceptions import ModelException
    with pytest.raises(ModelException, match='No model candidate completed'):
        mh.spot_check_machine_learning_models(dh, 'crossval.csv', k=3)
    assert mh._nystroem_runtime_rows == []


def test_runtime_export_path_and_failures_are_optional(default_model_handler, monkeypatch, tmp_path):
    mh, saved = default_model_handler, []
    mh._nystroem_runtime_rows = [{'Pipeline': 'MAX-NYS-MLPC', 'Fold': 1}]
    monkeypatch.setattr(JBGHandler.Helpers, 'save_matrix_as_csv',
                        lambda frame, path: saved.append((frame, path)), raising=False)
    monkeypatch.setattr(JBGHandler.Helpers, 'create_download_link', lambda *a, **k: 'download', raising=False)
    mh.handler.logger = Mock()
    mh._export_nystroem_runtime(tmp_path / 'crossval_config.csv')
    assert saved[0][1] == str(tmp_path / 'crossval_config_nystroem_runtime.csv')
    assert saved[0][0].iloc[0]['Fold'] == 1
    def broken(*args):
        raise OSError('export failed')
    monkeypatch.setattr(JBGHandler.Helpers, 'save_matrix_as_csv', broken)
    mh._export_nystroem_runtime(tmp_path / 'crossval_config.csv')
    mh.handler.logger.print_warning.assert_called_once()
    mh._nystroem_runtime_rows = []
    mh._export_nystroem_runtime(tmp_path / 'crossval_config.csv')
    mh.handler.logger.print_warning.assert_called_once()
