import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import JBGDarkNumberCorrectionRegressor as regressor_module
from JBGDarkNumberCorrectionFactor import DarkNumberCorrectionFactorEstimator
from JBGDarkNumberCorrectionRegressor import DarkNumberCorrectionFactorRegressor


class _NullLogger:
    def print_info(self, *_args, **_kwargs):
        return None

    def print_warning(self, *_args, **_kwargs):
        return None


def test_zero_recovery_is_nonfinite_and_invalid_for_regression():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0, 1] * 10)
    estimator = DarkNumberCorrectionFactorEstimator(
        estimator=LogisticRegression(),
        flip_fraction=0.5,
        n_splits=2,
        n_repeats=1,
        n_jobs=1,
        positive_class=1,
        sample_size=1.0,
        logger=_NullLogger(),
    )
    estimator._run_parallel = lambda _tasks: [0.0, 0.0]

    estimator.fit(X, y)

    assert np.isinf(estimator.score())
    assert estimator.correction_status_ == "zero_recovery"
    assert estimator.is_valid_for_regression_ is False


def test_insufficient_sample_fallback_is_not_a_regression_observation():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([1] + [0] * 9)
    estimator = DarkNumberCorrectionFactorEstimator(
        estimator=LogisticRegression(),
        flip_fraction=0.2,
        n_splits=2,
        n_repeats=1,
        n_jobs=1,
        positive_class=1,
        sample_size=1.0,
        logger=_NullLogger(),
    )

    estimator.fit(X, y)

    assert estimator.score() == 1.0
    assert estimator.correction_status_ == "insufficient_sample"
    assert estimator.is_valid_for_regression_ is False


def test_regressor_ignores_nonfinite_and_synthetic_sample_results(monkeypatch):
    values = {
        0.2: (np.inf, "zero_recovery", False),
        0.3: (2.0, "estimated", True),
        0.4: (1.0, "insufficient_sample", False),
        0.5: (1.5, "estimated", True),
    }

    class FakeCorrectionEstimator:
        def __init__(self, *args, sample_size, **kwargs):
            self.sample_size = sample_size

        def fit(self, X, y):
            score, status, valid = values[self.sample_size]
            self.correction_factor_ = score
            self.correction_status_ = status
            self.is_valid_for_regression_ = valid
            return self

        def score(self, X=None, y=None):
            return self.correction_factor_

    monkeypatch.setattr(
        regressor_module,
        "DarkNumberCorrectionFactorEstimator",
        FakeCorrectionEstimator,
    )

    regressor = DarkNumberCorrectionFactorRegressor(
        estimator=LogisticRegression(),
        sample_size_list=list(values),
        type="logbounded",
        logger=_NullLogger(),
    )
    regressor.fit(np.ones((20, 2)), np.array([0, 1] * 10))

    assert regressor.sample_results_ == [
        (0.2, np.inf),
        (0.3, 2.0),
        (0.4, 1.0),
        (0.5, 1.5),
    ]
    assert regressor.valid_sample_results_ == [(0.3, 2.0), (0.5, 1.5)]
    assert regressor.invalid_sample_results_ == [
        (0.2, np.inf, "zero_recovery"),
        (0.4, 1.0, "insufficient_sample"),
    ]
    assert np.isfinite(regressor.score())
    assert regressor.score() >= 1.0


def test_regressor_requires_enough_valid_samples(monkeypatch):
    class FakeCorrectionEstimator:
        def __init__(self, *args, sample_size, **kwargs):
            self.sample_size = sample_size

        def fit(self, X, y):
            self.correction_factor_ = 2.0 if self.sample_size == 0.5 else np.inf
            self.correction_status_ = "estimated" if self.sample_size == 0.5 else "zero_recovery"
            self.is_valid_for_regression_ = self.sample_size == 0.5
            return self

        def score(self, X=None, y=None):
            return self.correction_factor_

    monkeypatch.setattr(
        regressor_module,
        "DarkNumberCorrectionFactorEstimator",
        FakeCorrectionEstimator,
    )

    regressor = DarkNumberCorrectionFactorRegressor(
        estimator=LogisticRegression(),
        sample_size_list=[0.2, 0.3, 0.5],
        type="logbounded",
        logger=_NullLogger(),
    )

    with pytest.raises(ValueError, match="Insufficient valid correction-factor samples"):
        regressor.fit(np.ones((20, 2)), np.array([0, 1] * 10))
