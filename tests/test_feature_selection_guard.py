"""Reproduce SLSV's empty selection without fitting on the final holdout."""
import pickle
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import dill
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, issparse
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC

from JBGFeatureSelection import EmptyFeatureSelectionError, NonEmptySelectFromModel
from JBGHandler import ModelHandler, _SpotCheckState
from JBGMeta import Algorithm, Oversampling, Preprocess, Reduction, Undersampling


def weak_data():
    # Nonconstant input, so this reaches selection rather than a constant-data gate.
    return np.linspace(-1e-6, 1e-6, 60).reshape(30, 2), np.tile([0, 1], 15)


def strong_data():
    return make_classification(
        n_samples=90, n_features=6, n_informative=3, n_redundant=0,
        class_sep=2, random_state=23,
    )


def slsv():
    return Algorithm.SLSV.do_SLSV(max_iterations=20000, size=90)


def model_handler(workers=1):
    config = SimpleNamespace(debug=False, get_scoring_mechanism=lambda: "accuracy")
    return ModelHandler(SimpleNamespace(
        config=config, logger=Mock(), STANDARD_DESIRED_N_JOBS=workers,
    ))


def candidate(handler, X, y, state, algorithm=Algorithm.SLSV):
    dh = SimpleNamespace(X=pd.DataFrame(X), X_train=pd.DataFrame(X), Y_train=pd.Series(y))
    return handler._evaluate_spot_check_candidate(
        dh=dh, preprocessor=Preprocess.NOS, preprocessor_callable=None,
        reduction=Reduction.NOR, reduction_callable=None,
        algorithm=algorithm, algorithm_callable=algorithm.call_algorithm(20000, len(y)),
        oversampler=Oversampling.NOG, undersampler=Undersampling.NUG,
        kfold=StratifiedKFold(3, shuffle=True, random_state=1), state=state,
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_empty_selection_stops_before_downstream_fit_without_warning(sparse):
    X, y = weak_data()
    if sparse:
        X = csr_matrix(X)
    pipeline = slsv()
    downstream = pipeline.named_steps["classification"]
    downstream.fit = Mock(side_effect=AssertionError("zero-column classifier fit"))
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        with pytest.raises(EmptyFeatureSelectionError, match="selected 0 of 2"):
            pipeline.fit(X, y)
    downstream.fit.assert_not_called()
    assert not any("No features were selected" in str(item.message) for item in observed)


@pytest.mark.parametrize("sparse", [False, True])
def test_nonempty_support_transform_and_predictions_match_sklearn(sparse):
    X, y = strong_data()
    if sparse:
        X = csr_matrix(X)
    guarded = slsv().fit(X, y)
    reference = slsv()
    reference.set_params(feature_selection=SelectFromModel(
        clone(reference.named_steps["feature_selection"].estimator),
    ))
    reference.fit(X, y)
    actual = guarded.named_steps["feature_selection"]
    expected = reference.named_steps["feature_selection"]
    np.testing.assert_array_equal(actual.get_support(), expected.get_support())
    assert actual.threshold_ == expected.threshold_
    transformed = actual.transform(X)
    assert issparse(transformed) == sparse
    np.testing.assert_allclose(
        transformed.toarray() if sparse else transformed,
        expected.transform(X).toarray() if sparse else expected.transform(X),
    )
    np.testing.assert_array_equal(guarded.predict(X), reference.predict(X))


def test_clone_preserves_full_selector_parameter_contract():
    selector = NonEmptySelectFromModel(
        LinearSVC(penalty="l1", dual=False, C=0.3), threshold="0.5*mean",
        norm_order=2, max_features=2, importance_getter="coef_",
    )
    copied = clone(selector)
    assert type(copied) is NonEmptySelectFromModel
    assert copied.get_params(deep=False) == dict(
        selector.get_params(deep=False), estimator=copied.estimator,
    )
    assert copied.estimator.get_params() == selector.estimator.get_params()
    assert copied.estimator is not selector.estimator


def test_unfitted_selector_retains_sklearn_error():
    selector = slsv().named_steps["feature_selection"]
    reference = SelectFromModel(clone(selector.estimator))
    with pytest.raises(ValueError) as expected:
        reference.transform(weak_data()[0])
    with pytest.raises(type(expected.value), match="call fit") as actual:
        selector.transform(weak_data()[0])
    assert str(actual.value) == str(expected.value)


def test_prefitted_empty_selector_is_classified_without_attribute_error():
    X, y = weak_data()
    estimator = LinearSVC(penalty="l1", dual=False).fit(X, y)
    selector = NonEmptySelectFromModel(estimator, prefit=True)
    with pytest.raises(EmptyFeatureSelectionError, match="selected 0 of 2"):
        selector.transform(X)


@pytest.mark.parametrize("workers", [1, 2])
def test_empty_selection_exception_survives_real_cv_workers(workers):
    X, y = weak_data()
    with pytest.raises(EmptyFeatureSelectionError, match="downstream classifier was not run"):
        cross_val_score(slsv(), X, y, cv=3, n_jobs=workers, error_score="raise")


@pytest.mark.parametrize("serializer", [pickle, dill])
def test_new_and_historical_selectors_remain_serializable(serializer):
    X, y = strong_data()
    for selector_type in (NonEmptySelectFromModel, SelectFromModel):
        pipeline = slsv()
        pipeline.set_params(feature_selection=selector_type(
            LinearSVC(penalty="l1", dual=False, max_iter=20000),
        ))
        pipeline.fit(X, y)
        restored = serializer.loads(serializer.dumps(pipeline))
        assert type(restored.named_steps["feature_selection"]) is selector_type
        np.testing.assert_array_equal(restored.predict(X), pipeline.predict(X))


def test_cv_fits_selector_only_on_each_training_fold(monkeypatch):
    X, y = strong_data()
    fitted_rows = []
    original_fit = SelectFromModel.fit

    def record_fit(self, X, y=None, **kwargs):
        fitted_rows.append(len(X))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(SelectFromModel, "fit", record_fit)
    assert np.isfinite(cross_val_score(slsv(), X, y, cv=3, n_jobs=1)).all()
    assert fitted_rows == [60, 60, 60]


def test_one_empty_training_fold_rejects_candidate_even_if_full_fit_is_valid():
    X = np.zeros((12, 2))
    y = np.tile([0, 1], 6)
    X[:4, 0] = np.where(y[:4], 10, -10)
    assert slsv().fit(X, y).named_steps["feature_selection"].get_support().any()
    # The first training partition excludes all four informative rows.
    with pytest.raises(EmptyFeatureSelectionError):
        cross_val_score(slsv(), X, y, cv=StratifiedKFold(3), error_score="raise")


def test_cv_result_records_empty_selection_and_next_candidate_can_win():
    X, y = weak_data()
    handler = model_handler()
    state = _SpotCheckState(best_num_components=2, best_rfe_feature_selection=2)
    rows, success = candidate(handler, X, y, state)
    assert not success and state.trained_pipeline is None
    assert len(rows) == 1 and np.isnan(rows[0][4]) and np.isnan(rows[0][5])
    assert rows[0][-1].startswith("UNUSABLE: SelectFromModel(LinearSVC) selected 0 of 2")
    assert "Traceback" not in rows[0][-1]
    assert "Try a different preprocessing/model combination" in rows[0][-1]
    rows, success = candidate(handler, X, y, state, algorithm=Algorithm.LSVC)
    assert success and state.best_algorithm is Algorithm.LSVC
    assert np.isfinite(rows[0][4]) and rows[0][-1] == ""
    handler.handler.logger.print_warning.assert_not_called()


def test_unrelated_selector_failure_is_not_classified_as_empty_selection(monkeypatch):
    def broken_fit(*args, **kwargs):
        raise ValueError("unrelated selector failure")

    monkeypatch.setattr(SelectFromModel, "fit", broken_fit)
    X, y = strong_data()
    state = _SpotCheckState(best_num_components=6, best_rfe_feature_selection=6)
    rows, success = candidate(model_handler(), X, y, state)
    assert not success and "unrelated selector failure" in rows[0][-1]
    assert not rows[0][-1].startswith("UNUSABLE:")


def test_grid_search_raises_specific_empty_selection_error():
    X, y = weak_data()
    search = GridSearchCV(slsv(), {"classification__C": [0.1, 1]}, cv=3, error_score="raise")
    with pytest.raises(EmptyFeatureSelectionError):
        search.fit(X, y)


def test_ordinary_fit_and_fresh_retraining_stop_before_empty_classifier_fit():
    X, y = weak_data()
    for fit in (model_handler().train_picked_model, model_handler().retrain_picked_model):
        with pytest.raises(Exception, match="selected 0 of 2") as captured:
            fit(slsv(), X, y)
        assert isinstance(captured.value.__cause__, EmptyFeatureSelectionError)
