"""280d: PCA parameter policy is extracted without changing the estimator."""

import inspect

import pytest
from sklearn.decomposition import PCA

from JBGMeta import Reduction, PCA_VARIANCE_EXPLAINED
from JBGReductionPolicies import choose_pca_components


@pytest.mark.parametrize("requested, expected", [
    (None, 0.95), (1, 1), (4, 4), (3.9, 3), (0, 0),
])
def test_pca_policy_preserves_component_value_and_type(requested, expected):
    actual = choose_pca_components(8, 5, requested)
    assert actual == expected
    assert type(actual) is type(expected)


def test_pca_policy_respects_explicit_variance_target():
    assert choose_pca_components(8, 5, None, 0.8) == 0.8
    assert choose_pca_components(8, 5, 2, 0.8) == 2


def test_reduction_pca_keeps_historical_sklearn_configuration():
    default = Reduction.PCA.get_PCA(8, 5)
    assert isinstance(default, PCA)
    assert default.n_components == PCA_VARIANCE_EXPLAINED
    assert default.svd_solver == "covariance_eigh"
    explicit = Reduction.PCA.get_PCA(8, 5, 2)
    assert isinstance(explicit, PCA)
    assert explicit.n_components == 2
    assert explicit.svd_solver == "auto"


def test_reduction_policy_has_no_ml_or_gui_imports():
    from pathlib import Path
    source = Path(inspect.getfile(choose_pca_components)).read_text(encoding="utf-8")
    assert "import JBGMeta" not in source
    assert "import sklearn" not in source
    assert "import ipywidgets" not in source
