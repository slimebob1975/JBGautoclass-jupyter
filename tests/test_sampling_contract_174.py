"""Revision 174 categorical-sampling contract tests without the optional ML stack."""

import numpy as np
import pytest
from imblearn.over_sampling import SMOTENC, SMOTEN

from JBGSampling import categorical_feature_indices, validate_categorical_oversampling


def test_categorical_feature_indices_follow_converted_column_order():
    assert categorical_feature_indices(
        ["numeric_a", "category_b", "numeric_c", "category_a"],
        ["category_a", "category_b"],
    ) == (1, 3)


def test_categorical_feature_indices_fail_closed_for_ambiguous_or_missing_roles():
    with pytest.raises(ValueError, match="unique converted feature names"):
        categorical_feature_indices(["a", "a"], ["a"])
    with pytest.raises(ValueError, match="absent after conversion"):
        categorical_feature_indices(["a", "b"], ["missing"])


def test_smote_nc_contract_requires_mixed_explicit_roles():
    assert validate_categorical_oversampling("SNC", (2,), 3) == (2,)
    with pytest.raises(ValueError, match="no categorical converted features"):
        validate_categorical_oversampling("SNC", (), 3)
    with pytest.raises(ValueError, match="both categorical and non-categorical"):
        validate_categorical_oversampling("SNC", (0, 1, 2), 3)


def test_smote_n_contract_requires_every_feature_to_be_categorical():
    assert validate_categorical_oversampling("SNN", (2, 0, 1), 3) == (2, 0, 1)
    with pytest.raises(ValueError, match="every converted predictor"):
        validate_categorical_oversampling("SNN", (0, 2), 3)


@pytest.mark.parametrize("indices", [(-1,), (3,), (True,), (1, 1)])
def test_categorical_contract_rejects_invalid_indices(indices):
    with pytest.raises(ValueError, match="invalid converted-feature index"):
        validate_categorical_oversampling("SNC", indices, 3)


def test_explicit_smote_nc_indices_keep_category_values_discrete():
    rng = np.random.RandomState(11)
    majority = 30
    minority = 10
    X = np.column_stack([
        np.r_[rng.normal(0.0, 1.0, majority), rng.normal(2.0, 1.0, minority)],
        np.r_[rng.normal(1.0, 0.5, majority), rng.normal(3.0, 0.5, minority)],
        np.r_[np.tile([0.0, 1.0, 2.0], 10), np.tile([1.0, 2.0], 5)],
    ])
    y = np.array([0] * majority + [1] * minority)
    categorical = validate_categorical_oversampling("SNC", (2,), X.shape[1])
    resampled, _ = SMOTENC(
        categorical_features=list(categorical), random_state=23
    ).fit_resample(X, y)
    assert set(np.unique(resampled[:, 2])).issubset(set(np.unique(X[:, 2])))


def test_smote_n_keeps_all_feature_values_discrete():
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [1, 2],
        [2, 0], [2, 1], [2, 2], [0, 0], [1, 1], [2, 2],
        [0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [2, 1],
        [2, 2], [2, 1], [1, 2], [2, 0], [1, 1], [2, 2],
    ], dtype=float)
    y = np.array([0] * 18 + [1] * 6)
    validate_categorical_oversampling("SNN", (0, 1), X.shape[1])
    resampled, _ = SMOTEN(random_state=23).fit_resample(X, y)
    for column in range(X.shape[1]):
        assert set(np.unique(resampled[:, column])).issubset(set(np.unique(X[:, column])))
