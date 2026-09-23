import pickle
import warnings

import numpy as np
import pytest
from numpy.linalg import LinAlgError
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from JBGMeta import Algorithm, AlgorithmGridSearchParams


_X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [2.0, 0.0],
    [2.0, 1.0],
])
_Y = np.array([0, 0, 0, 1, 1, 1])
_BAG_X = np.arange(80, dtype=float).reshape(20, 4)
_BAG_Y = np.array([0] * 10 + [1] * 10)


def test_logistic_regression_grid_uses_sklearn_18_penalty_api():
    params = AlgorithmGridSearchParams.LRN.parameters
    combinations = list(ParameterGrid(params))

    assert "penalty" not in params
    assert params["l1_ratio"] == [0.0]
    assert params["C"] == [0.1, 1, 10]
    assert len(combinations) == 18

    for combination in combinations:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            LogisticRegression(max_iter=200, **combination).fit(_X, _Y)

        messages = [str(item.message) for item in captured]
        assert not any("'penalty' was deprecated" in message for message in messages)
        assert not any("Setting penalty=None" in message for message in messages)


def test_linear_svc_grid_contains_only_supported_combinations():
    params = AlgorithmGridSearchParams.LSVC.parameters
    combinations = list(ParameterGrid(params))

    assert len(combinations) == 6
    assert all(combination["loss"] == "squared_hinge" for combination in combinations)

    for combination in combinations:
        LinearSVC(max_iter=2000, **combination).fit(_X, _Y)


def test_linear_svc_grid_avoids_convergence_warning_with_sparse_compatible_scaling():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler(with_mean=False).fit_transform(X)

    for combination in ParameterGrid(AlgorithmGridSearchParams.LSVC.parameters):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            LinearSVC(max_iter=20000, **combination).fit(X, y)

        assert not any(
            issubclass(item.category, ConvergenceWarning)
            for item in captured
        ), combination


def test_bagging_classifier_grid_contains_only_valid_fractional_subsamples():
    params = AlgorithmGridSearchParams.BGC.parameters
    combinations = list(ParameterGrid(params))

    assert params["max_samples"] == (0.5, 0.75, 1.0)
    assert params["max_features"] == (0.5, 0.75, 1.0)
    assert len(combinations) == 54

    for combination in combinations:
        BaggingClassifier(**combination).fit(_BAG_X, _BAG_Y)


def test_passive_aggressive_algorithm_uses_sgd_equivalent_without_deprecation_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        estimator = Algorithm.PAC.call_algorithm(max_iterations=200, size=len(_X))
        estimator.fit(_X, _Y)

    assert isinstance(estimator, SGDClassifier)
    assert estimator.loss == "hinge"
    assert estimator.penalty is None
    assert estimator.learning_rate == "pa1"
    assert estimator.eta0 == 1.0
    assert estimator.max_iter == 200
    assert not any(
        "PassiveAggressiveClassifier is deprecated" in str(item.message)
        for item in captured
    )


def test_sgd_classifier_grid_uses_only_current_loss_names():
    params = AlgorithmGridSearchParams.SGDE.parameters
    combinations = list(ParameterGrid(params))

    assert "log" not in params["loss"]
    assert "log_loss" in params["loss"]
    assert len(combinations) == 27

    for combination in combinations:
        SGDClassifier(max_iter=200, **combination).fit(_X, _Y)


def test_discrete_naive_bayes_grids_avoid_zero_alpha_and_nonfinite_probabilities():
    X = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 2.0, 0.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    cases = (
        (AlgorithmGridSearchParams.MNB.parameters, MultinomialNB),
        (AlgorithmGridSearchParams.BNB.parameters, BernoulliNB),
        (AlgorithmGridSearchParams.CNB.parameters, ComplementNB),
    )

    for params, estimator_type in cases:
        assert params["alpha"] == (0.01, 0.1, 1.0)

        for combination in ParameterGrid(params):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                estimator = estimator_type(**combination).fit(X, y)
                probabilities = estimator.predict_proba(X)

            assert np.isfinite(probabilities).all()
            assert not any(
                issubclass(item.category, RuntimeWarning) for item in captured
            )


def _contains_numpy_array(value):
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, dict):
        return any(_contains_numpy_array(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_numpy_array(item) for item in value)
    return False


def test_nearest_centroid_grid_contains_only_supported_combinations():
    params = AlgorithmGridSearchParams.NCT.parameters
    combinations = list(ParameterGrid(params))

    assert "euclidian" not in params["metric"]
    assert "euclidean" in params["metric"]
    assert 0.0 not in params["shrink_threshold"]
    assert None in params["shrink_threshold"]
    assert len(combinations) == 202

    for combination in combinations:
        Algorithm.NCT.call_algorithm(
            max_iterations=200, size=len(_X)
        ).set_params(**combination).fit(_X, _Y)


def test_qda_spot_check_starts_with_regularization_for_rank_deficient_covariance():
    # Duplicate feature columns make each class covariance rank deficient.
    X = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        [3.0, 0.0, 0.0],
        [3.0, 1.0, 1.0],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    with pytest.raises(LinAlgError, match="not full rank"):
        QuadraticDiscriminantAnalysis().fit(X, y)

    estimator = Algorithm.QDA.call_algorithm(max_iterations=200, size=len(X))
    configured_reg_params = AlgorithmGridSearchParams.QDA.parameters["reg_param"]

    assert estimator.reg_param == min(configured_reg_params) == 0.1
    estimator.fit(X, y)
    assert estimator.predict(X).shape == y.shape


def test_algorithm_grid_search_params_are_pickle_safe_and_do_not_store_numpy_arrays():
    for grid in AlgorithmGridSearchParams:
        assert not _contains_numpy_array(grid.value), grid.name
        assert pickle.loads(pickle.dumps(grid)) is grid


def test_algorithms_with_former_numpy_grids_are_pickle_safe():
    for algorithm_name in ("RADN", "NCT", "QDA", "GBC"):
        algorithm = Algorithm[algorithm_name]
        assert pickle.loads(pickle.dumps(algorithm)) is algorithm
