import warnings

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import ParameterGrid
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

    assert len(combinations) == 8

    for combination in combinations:
        LinearSVC(max_iter=2000, **combination).fit(_X, _Y)


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
