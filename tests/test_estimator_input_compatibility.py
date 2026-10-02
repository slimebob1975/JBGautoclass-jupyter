from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.validation import check_array

from JBGEstimatorInput import call_with_sparse_input_retry, is_sparse_input_rejection


class Logger:
    def __init__(self):
        self.warnings = []
    def print_warning(self, message):
        self.warnings.append(message)
    def print_info(self, *args, **kwargs):
        pass
    def print_dragon(self, *args, **kwargs):
        pass
    def anaconda_debug(self, *args, **kwargs):
        pass
    def print_progress(self, *args, **kwargs):
        pass


def data():
    X = sparse.csr_matrix(np.array([[0, 1], [1, 0], [0, 2], [2, 0],
                                    [0, 3], [3, 0], [0, 4], [4, 0]], dtype=np.float32))
    return X, np.array([0, 1] * 4)


def rejected(X, input_name="X"):
    try:
        check_array(X, accept_sparse=False, input_name=input_name)
    except TypeError as error:
        return error
    raise AssertionError("Expected real sklearn sparse rejection")


def test_guard_accepts_real_sklearn_sparse_X_rejection_only():
    X, _ = data()
    error = rejected(X)
    assert is_sparse_input_rejection(error, X)
    assert not is_sparse_input_rejection(error, X.toarray())
    assert not is_sparse_input_rejection(rejected(X, "y"), X)
    assert not is_sparse_input_rejection(TypeError(str(error)), X)
    assert not is_sparse_input_rejection(ValueError(str(error)), X)


@pytest.mark.parametrize("message", ["DataFrame not supported", "invalid model parameter",
                                    "internal arithmetic failure", "Sparse data rejected"])
@pytest.mark.parametrize("use_sparse", [True, False])
def test_unrelated_errors_keep_identity_and_execute_once(message, use_sparse):
    X, _ = data()
    if not use_sparse:
        X = X.toarray()
    logger, calls, error = Logger(), [], TypeError(message)
    def operation(input_X):
        calls.append(input_X)
        raise error
    with pytest.raises(TypeError) as caught:
        call_with_sparse_input_retry(operation, X, logger=logger, context="test", estimator=object())
    assert caught.value is error
    assert calls == [X]
    assert logger.warnings == []


def test_dense_retry_preserves_values_dtype_and_logs_before_conversion():
    X, _ = data()
    logger, calls = Logger(), []
    def operation(input_X):
        calls.append(input_X)
        return check_array(input_X, accept_sparse=False)
    actual = call_with_sparse_input_retry(operation, X, logger=logger, context="CV",
                                         estimator=Pipeline([("MAX", MaxAbsScaler()), ("GNB", GaussianNB())]))
    assert len(calls) == 2 and calls[0] is X
    np.testing.assert_array_equal(actual, X.toarray())
    assert actual.dtype == np.float32
    assert "pipeline MAX-GNB" in logger.warnings[0]
    assert "shape=(8, 2)" in logger.warnings[0]
    assert "64 bytes" in logger.warnings[0]


def test_failure_after_confirmed_retry_keeps_original_failure_and_stops():
    X, _ = data()
    logger, calls, error = Logger(), [], RuntimeError("dense estimator bug")
    def operation(input_X):
        calls.append(input_X)
        if sparse.issparse(input_X):
            check_array(input_X, accept_sparse=False)
        raise error
    with pytest.raises(RuntimeError) as caught:
        call_with_sparse_input_retry(operation, X, logger=logger, context="CV", estimator=object())
    assert caught.value is error
    assert len(calls) == 2
    assert any("one confirmed sparse-X retry" in n for n in error.__notes__)


def test_dense_allocation_failure_is_logged_first_and_never_retried(monkeypatch):
    X, _ = data()
    logger, calls = Logger(), []
    error = MemoryError("dense allocation failed")
    def allocate():
        assert logger.warnings
        raise error
    monkeypatch.setattr(X, "toarray", allocate)
    def operation(input_X):
        calls.append(input_X)
        return check_array(input_X, accept_sparse=False)
    with pytest.raises(MemoryError) as caught:
        call_with_sparse_input_retry(operation, X, logger=logger, context="fit", estimator=object())
    assert caught.value is error
    assert len(calls) == 1


def remote_sparse_rejection(X):
    return check_array(X, accept_sparse=False)


def test_real_joblib_worker_rejection_can_be_confirmed():
    X, _ = data()
    with pytest.raises(TypeError) as caught:
        Parallel(n_jobs=2)(delayed(remote_sparse_rejection)(X) for _ in range(2))
    assert is_sparse_input_rejection(caught.value, X)
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_serialized_worker_traceback_supports_posix_and_windows(separator):
    X, _ = data()
    error = TypeError(str(rejected(X)))
    from joblib.externals.loky.process_executor import _RemoteTraceback
    path = separator.join(["C:", "env", "sklearn", "utils", "validation.py"])
    error.__cause__ = _RemoteTraceback(f'File "{path}", line 626, in _ensure_sparse_format\n    raise TypeError')
    assert is_sparse_input_rejection(error, X)


def handler():
    import JBGHandler
    mh = object.__new__(JBGHandler.ModelHandler)
    logger = Logger()
    mh.handler = SimpleNamespace(logger=logger, STANDARD_DESIRED_N_JOBS=2,
                                 config=SimpleNamespace(debug=False, io=SimpleNamespace(verbose=False),
                                                        get_scoring_mechanism=lambda: "accuracy"))
    return mh, logger


@pytest.mark.parametrize("reduction", ["NOR", "RFE"])
def test_GNB_sparse_preserving_candidates_are_skipped_before_any_probe(reduction):
    from JBGMeta import Algorithm, Preprocess, Reduction
    mh, _ = handler()
    X, _ = data()
    class ForbiddenScaler:
        def fit_transform(self, input_X):
            pytest.fail("Known incompatibility must be skipped before preprocessing")
    assert mh.get_preflight_skip_reason(Preprocess.MAX, ForbiddenScaler(), Reduction[reduction], Algorithm.GNB, X) == "GaussianNB requires dense input"


@pytest.mark.parametrize("reduction", ["PCA", "TSVD", "NYS"])
def test_GNB_dense_reductions_remain_eligible_and_work(reduction):
    from JBGMeta import Algorithm, Preprocess, Reduction
    mh, _ = handler()
    X, y = data()
    assert mh.get_preflight_skip_reason(Preprocess.MAX, MaxAbsScaler(), Reduction[reduction], Algorithm.GNB, X) is None
    transformer = {"PCA": PCA(n_components=1, svd_solver="arpack"),
                   "TSVD": TruncatedSVD(n_components=1, random_state=42),
                   "NYS": Nystroem(n_components=2, random_state=42)}[reduction]
    scores = cross_validate(Pipeline([("MAX", MaxAbsScaler()), (reduction, transformer), ("GNB", GaussianNB())]), X, y,
                            cv=2, error_score="raise")["test_score"]
    assert np.isfinite(scores).all()


def test_dense_GNB_without_reduction_remains_eligible():
    from JBGMeta import Algorithm, Preprocess, Reduction
    mh, _ = handler()
    X, _ = data()
    assert mh.get_preflight_skip_reason(Preprocess.MAX, MaxAbsScaler(), Reduction.NOR, Algorithm.GNB, pd.DataFrame(X.toarray())) is None


@pytest.mark.parametrize("estimator", [GaussianNB(), LogisticRegression(random_state=42)])
def test_real_CV_matches_explicit_dense_or_sparse_input_and_keeps_worker_timing(estimator):
    from JBGMeta import Algorithm
    mh, logger = handler()
    X, y = data()
    frame = pd.DataFrame.sparse.from_spmatrix(X)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    expected_X = X.toarray() if isinstance(estimator, GaussianNB) else X
    expected = cross_validate(estimator, expected_X, y, cv=cv, scoring="accuracy", error_score="raise")["test_score"]
    actual = mh.get_cross_val_score(estimator, SimpleNamespace(X_train=frame, Y_train=pd.Series(y)), cv, Algorithm.GNB)
    np.testing.assert_array_equal(actual, expected)
    assert mh._last_cv_timing[1].workers == 2
    assert len(logger.warnings) == int(isinstance(estimator, GaussianNB))


def test_CV_does_not_repeat_unrelated_error_or_retain_stale_timing():
    from JBGMeta import Algorithm
    mh, logger = handler()
    X, y = data()
    error, calls = TypeError("invalid callback parameter"), []
    mh._last_cv_timing = "stale"
    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise error
    mh.execute_n_job = fail
    with pytest.raises(TypeError) as caught:
        mh.get_cross_val_score(object(), SimpleNamespace(X_train=pd.DataFrame.sparse.from_spmatrix(X), Y_train=pd.Series(y)),
                               StratifiedKFold(2), Algorithm.GNB)
    assert caught.value is error and len(calls) == 1
    assert mh._last_cv_timing is None and logger.warnings == []


@pytest.mark.parametrize("method", ["train_picked_model", "_fit_pipeline_for_validation", "model_parameter_grid_search"])
def test_training_entry_points_retry_confirmed_sparse_rejection(method):
    mh, logger = handler()
    X, y = data()
    frame, labels = pd.DataFrame.sparse.from_spmatrix(X), pd.Series(y)
    model = GaussianNB()
    if method == "train_picked_model":
        fitted = mh.train_picked_model(model, frame, labels)
    elif method == "_fit_pipeline_for_validation":
        mh._fit_pipeline_for_validation(model, SimpleNamespace(X_train=frame, Y_train=labels))
        fitted = model
    else:
        fitted, _ = mh.model_parameter_grid_search(model, {"var_smoothing": [1e-9]}, 2, frame, labels)
    np.testing.assert_array_equal(fitted.predict(X.toarray()), y)
    assert len(logger.warnings) == 1


def test_validation_scorer_retries_only_genuine_sparse_X_rejection():
    mh, logger = handler()
    X, y = data()
    fitted = GaussianNB().fit(X.toarray(), y)
    mh.handler.config.get_scoring_mechanism = lambda: get_scorer("accuracy")
    score = mh._score_validation_pipeline(fitted, SimpleNamespace(X_train=X, X_validation=pd.DataFrame.sparse.from_spmatrix(X), Y_validation=pd.Series(y)))
    assert score == 1.0 and len(logger.warnings) == 1


@pytest.mark.parametrize("method", ["train_picked_model", "_fit_pipeline_for_validation"])
def test_fit_entry_points_do_not_repeat_unrelated_typeerror(method):
    mh, logger = handler()
    error, calls = TypeError("estimator callback bug"), []
    class BrokenModel:
        def fit(self, X, y):
            calls.append(X)
            raise error
    X, y = data()
    frame, labels = pd.DataFrame.sparse.from_spmatrix(X), pd.Series(y)
    with pytest.raises(TypeError) as caught:
        if method == "train_picked_model":
            mh.train_picked_model(BrokenModel(), frame, labels)
        else:
            mh._fit_pipeline_for_validation(BrokenModel(), SimpleNamespace(X_train=frame, Y_train=labels))
    assert caught.value is error and len(calls) == 1 and logger.warnings == []


def test_grid_search_preserves_unrelated_fit_failure_as_cause_and_does_not_retry(monkeypatch):
    import JBGHandler
    from JBGExceptions import ModelException
    mh, logger = handler()
    error, calls = TypeError("bad GridSearch callback"), []
    def fail(search, X, y):
        calls.append(X)
        raise error
    monkeypatch.setattr(JBGHandler.GridSearchCV, "fit", fail)
    X, y = data()
    with pytest.raises(ModelException) as caught:
        mh.model_parameter_grid_search(GaussianNB(), {"var_smoothing": [1e-9]}, 2,
                                       pd.DataFrame.sparse.from_spmatrix(X), pd.Series(y))
    assert caught.value.__cause__ is error
    assert len(calls) == 1 and logger.warnings == []


def predictions_handler():
    import JBGHandler
    ph = object.__new__(JBGHandler.PredictionsHandler)
    ph.handler = SimpleNamespace(logger=Logger())
    ph.calculate_probability = lambda: None
    return ph


def test_prediction_only_legacy_dense_GNB_still_works_with_sparse_input():
    ph = predictions_handler()
    X, y = data()
    model = GaussianNB().fit(X.toarray(), y)
    ph.make_predictions(model, pd.DataFrame.sparse.from_spmatrix(X), pd.Series([0, 1]))
    np.testing.assert_array_equal(ph.predictions, y)
    np.testing.assert_allclose(ph.probabilites, model.predict_proba(X.toarray()))
    assert ph.could_predict_proba and len(ph.handler.logger.warnings) == 2


def test_prediction_error_is_not_retried_or_misclassified_as_regenerate_model():
    ph = predictions_handler()
    error, calls = TypeError("predict internals failed"), []
    class BrokenModel:
        def predict(self, X):
            calls.append(X)
            raise error
    X, _ = data()
    with pytest.raises(TypeError) as caught:
        ph.make_predictions(BrokenModel(), pd.DataFrame.sparse.from_spmatrix(X), pd.Series([0, 1]))
    assert caught.value is error and len(calls) == 1
    assert ph.handler.logger.warnings == []


def test_optional_probability_error_keeps_existing_fallback_without_dense_retry():
    ph = predictions_handler()
    calls = []
    class Model:
        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)
        def predict_proba(self, X):
            calls.append(X)
            raise TypeError("probability implementation bug")
    X, _ = data()
    ph.make_predictions(Model(), pd.DataFrame.sparse.from_spmatrix(X), pd.Series([0, 1]))
    assert len(calls) == 1 and not ph.could_predict_proba
    assert np.all(ph.probabilites == -1)
    assert "probability implementation bug" in ph.handler.logger.warnings[0]
    assert "Retrying" not in ph.handler.logger.warnings[0]


def test_misprediction_second_model_failure_does_not_repeat_first_model():
    ph = predictions_handler()
    X, y = data()
    error, calls = TypeError("second model bug"), []
    class Model:
        def __init__(self, fail=False):
            self.fail = fail
        def predict(self, X):
            calls.append(self.fail)
            if self.fail:
                raise error
            return y
    frame = pd.DataFrame.sparse.from_spmatrix(X)
    with pytest.raises(TypeError) as caught:
        ph.most_mispredicted(frame, Model(True), Model(), frame, pd.Series(y))
    assert caught.value is error and calls == [False, True]
