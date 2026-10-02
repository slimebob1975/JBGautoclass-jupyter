"""Regress the dense-only winner failure at the Dark Number boundary."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.validation import check_array

import JBGHandler
from JBGHandler import PredictionsHandler
from JBGEstimatorInput import call_with_resolved_input, prepare_known_dense_input


class Logger:
    def __init__(self):
        self.warnings = []
        self.info = []
    def print_warning(self, message):
        self.warnings.append(message)
    def print_info(self, message):
        self.info.append(message)


def handler():
    ph = object.__new__(PredictionsHandler)
    mh = SimpleNamespace(execute_n_job=lambda operation, estimator, X, y, **kwargs:
                         operation(estimator, X, y))
    ph.handler = SimpleNamespace(
        logger=Logger(), STANDARD_DESIRED_N_JOBS=1, get_handler=lambda name: mh,
        config=SimpleNamespace(get_dark_number_flip_fraction=lambda: 0.2,
                              should_use_experimental_perturbed_dark_number_fallback=lambda: True),
    )
    ph._auto_n_splits_and_repeats = lambda **kwargs: (2, 1)
    return ph


def dataset():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(160, 3))
    y = pd.Series((values[:, 0] + rng.normal(scale=0.9, size=160) > 0).astype(int))
    return pd.DataFrame(values), y


def fitted_models(X, y):
    def fit(rows):
        pipe = Pipeline([('MAX', MaxAbsScaler()), ('HIST', HistGradientBoostingClassifier(
            max_iter=8, min_samples_leaf=5, random_state=42))])
        return pipe.fit(X.iloc[rows].to_numpy(), y.iloc[rows])
    return [fit(slice(0, 120)), fit(slice(None))]


def run(ph, X, y, models):
    ph.get_dark_numbers(
        X=X, Y=y, models=models, model_names=['Cross', 'Retrained'], target_class='1',
        X_validation=X.iloc[120:], Y_validation=y.iloc[120:],
        X_cv_training=X.iloc[:120], Y_cv_training=y.iloc[:120], type='all',
    )


def test_real_hist_sparse_scopes_and_corrections_match_dense_run():
    X, y = dataset()
    models = fitted_models(X, y)
    baseline, actual = handler(), handler()
    run(baseline, X, y, models)
    run(actual, pd.DataFrame.sparse.from_spmatrix(sparse.csr_matrix(X)), y, models)
    pd.testing.assert_frame_equal(actual.dark_numbers, baseline.dark_numbers)
    pd.testing.assert_frame_equal(actual.dark_numb_conf_matrix, baseline.dark_numb_conf_matrix)
    pd.testing.assert_frame_equal(
        actual.dark_number_fallback_events.drop(columns='execution_seconds', errors='ignore'),
        baseline.dark_number_fallback_events.drop(columns='execution_seconds', errors='ignore'),
    )
    assert len(actual.dark_numbers) == 20  # five formulas in each of four scopes
    assert any('(direct;' in msg for msg in actual.handler.logger.info)
    assert not any('Sparse data was passed' in msg for msg in actual.handler.logger.info)
    retries = [w for w in actual.handler.logger.warnings if 'Confirmed sparse-X rejection' in w]
    assert len(retries) == 2  # once per model, then reuse for proba/full scopes/corrections
    assert 'D_cv_test' in retries[0] and 'D_retrained_full' in retries[1]
    correction_logs = [w for w in actual.handler.logger.warnings if 'correction input' in w]
    assert len(correction_logs) == 2
    assert 'shape=(120, 3)' in correction_logs[0]
    assert 'shape=(160, 3)' in correction_logs[1]


class DenseFixedModel(BaseEstimator):
    def __init__(self, probabilities_only=False):
        self.probabilities_only = probabilities_only
    def predict(self, X):
        X = check_array(X, accept_sparse=self.probabilities_only)
        return (X[:, 0].toarray().ravel() if sparse.issparse(X) else X[:, 0]).astype(int)
    def predict_proba(self, X):
        X = check_array(X, accept_sparse=False)
        return np.tile([0.9, 0.1], (len(X), 1))


@pytest.mark.parametrize('probabilities_only', [False, True])
def test_dense_zero_fp_skips_correction_allocation_and_work(probabilities_only):
    X = pd.DataFrame.sparse.from_spmatrix(sparse.csr_matrix([[0], [1], [0], [1]]))
    y = pd.Series([0, 1, 0, 1])
    ph = handler()
    def unexpected(**kwargs):
        pytest.fail('Zero-FP reports must not estimate corrections')
    ph._calculate_dark_number_corrections = unexpected
    ph.get_dark_numbers(X=X, Y=y, models=[DenseFixedModel(probabilities_only)],
                        model_names=['Cross'], target_class='1', type='all')
    assert set(ph.dark_numbers['corr_source']) == {'not_needed_zero_fp'}
    assert not any('correction input' in w for w in ph.handler.logger.warnings)
    assert len([w for w in ph.handler.logger.warnings if 'Confirmed sparse-X rejection' in w]) == 1


@pytest.mark.parametrize('branch', ['regression', 'perturbed'])
def test_dense_requirement_reaches_both_correction_fallbacks(monkeypatch, branch):
    X, y = dataset()
    models = fitted_models(X, y)
    received = []
    class Factor:
        def __init__(self, **kwargs):
            self.correction_status_ = 'zero_recovery' if branch == 'perturbed' else None
            self.is_valid_for_regression_ = False
        def fit(self, X, y):
            assert not sparse.issparse(X)
            if branch == 'regression':
                raise MemoryError('forced resource fallback')
            return self
    class Regressor:
        invalid_sample_results_ = []
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y):
            received.append(X.copy())
            assert not sparse.issparse(X)
            return self
        def score(self):
            return 2.0
    def perturbed(model, X, y, **kwargs):
        received.append(X.copy())
        assert not sparse.issparse(X)
        return 2.0, 'perturbed_same_model', {'accepted': True}
    monkeypatch.setattr(JBGHandler, 'DarkNumberCorrectionFactorEstimator', Factor)
    monkeypatch.setattr(JBGHandler, 'DarkNumberCorrectionFactorRegressor', Regressor)
    monkeypatch.setattr(JBGHandler, 'estimate_perturbed_same_model_correction', perturbed)
    ph = handler()
    run(ph, pd.DataFrame.sparse.from_spmatrix(sparse.csr_matrix(X)), y, models)
    assert len(received) == 2
    np.testing.assert_array_equal(received[0], X.iloc[:120].to_numpy())
    np.testing.assert_array_equal(received[1], X.to_numpy())
    expected_source = 'regressed_same_model' if branch == 'regression' else 'perturbed_same_model'
    assert set(ph.dark_numbers['corr_source']) == {expected_source}


def test_unrelated_dark_number_typeerror_is_not_densified():
    error, calls = TypeError('unrelated estimator bug'), []
    class Broken:
        def predict(self, X):
            calls.append(X)
            raise error
    ph = handler()
    with pytest.raises(TypeError) as caught:
        ph.get_dark_numbers(X=pd.DataFrame.sparse.from_spmatrix(sparse.eye(4, format='csr')),
                            Y=pd.Series([0, 1, 0, 1]), models=[Broken()])
    assert caught.value is error and len(calls) == 1
    assert sparse.issparse(calls[0]) and ph.handler.logger.warnings == []


def test_dense_requirement_is_specific_to_each_model():
    X = pd.DataFrame.sparse.from_spmatrix(sparse.csr_matrix([[0], [1], [0], [1]]))
    y = pd.Series([0, 1, 0, 1])
    calls = []
    class SparseFixed:
        def predict(self, X):
            calls.append(X)
            assert sparse.issparse(X)
            return X[:, 0].toarray().ravel().astype(int)
        def predict_proba(self, X):
            calls.append(X)
            assert sparse.issparse(X)
            return np.tile([0.9, 0.1], (X.shape[0], 1))
    ph = handler()
    ph.get_dark_numbers(X=X, Y=y, models=[DenseFixedModel(), SparseFixed()],
                        model_names=['Dense', 'Sparse'], target_class='1')
    assert len(calls) == 2
    assert all('pipeline DenseFixedModel' in w for w in ph.handler.logger.warnings)


def test_resolved_input_preserves_sparse_success_and_dense_retry_identity():
    X = sparse.csr_matrix(np.array([[0, 1], [2, 0]], dtype=np.float32))
    log = Logger()
    _, resolved = call_with_resolved_input(lambda value: value, X, logger=log,
                                           context='test', estimator=object())
    assert resolved is X and log.warnings == []
    result, resolved = call_with_resolved_input(lambda value: check_array(value), X,
                                               logger=log, context='test', estimator=object())
    assert result is resolved and resolved.dtype == X.dtype
    np.testing.assert_array_equal(resolved, X.toarray())


def test_known_dense_allocation_logs_before_failure_and_does_not_retry(monkeypatch):
    X = sparse.csr_matrix(np.eye(2, dtype=np.float32))
    log, calls = Logger(), []
    def fail():
        calls.append(1)
        assert '16 bytes' in log.warnings[0]
        raise MemoryError('cannot allocate')
    monkeypatch.setattr(X, 'toarray', fail)
    with pytest.raises(MemoryError, match='cannot allocate'):
        prepare_known_dense_input(X, logger=log, context='correction', estimator=object())
    assert calls == [1]
