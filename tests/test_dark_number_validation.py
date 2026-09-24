import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import JBGDarkNumberValidation as validation_module
from JBGDarkNumberValidation import DarkNumberValidationHarness


def _make_data():
    X, y = make_classification(
        n_samples=420,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        weights=[0.75, 0.25],
        class_sep=2.5,
        flip_y=0.0,
        random_state=7,
    )
    return X, y


def test_injected_noise_truth_is_hidden_positives_over_observed_negatives():
    _, y = _make_data()
    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        injected_noise_fraction=0.20,
        correction_n_splits=3,
        correction_n_repeats=1,
        random_state=11,
    )

    y_noisy, hidden_mask, _, truth = harness.inject_known_noise(y)

    hidden_count = int(hidden_mask.sum())
    observed_negatives = int(np.sum(y_noisy != 1))
    assert hidden_count == int(np.sum(y == 1) * 0.20)
    assert truth == pytest.approx(hidden_count / observed_negatives)


def test_validation_harness_reports_three_estimates_and_model_specific_corrs():
    X, y = _make_data()
    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        injected_noise_fraction=0.15,
        correction_flip_fraction=0.20,
        correction_n_splits=3,
        correction_n_repeats=1,
        random_state=13,
    )

    result = harness.run_once(X, y)

    assert result.hidden_positive_count > 0
    assert result.true_dark_number > 0
    assert result.corr_cv >= 1.0
    assert result.corr_cv_source in {"direct", "regressed", "fallback/unestimable"}
    assert result.corr_retrained >= 1.0
    assert result.corr_retrained_source in {"direct", "regressed", "fallback/unestimable"}
    assert result.d_cv_test >= 0
    assert result.d_cv_full >= 0
    assert result.d_retrained_full >= 0
    assert result.interval_lower == min(result.d_cv_test, result.d_cv_full, result.d_retrained_full)
    assert result.interval_upper == max(result.d_cv_test, result.d_cv_full, result.d_retrained_full)
    assert np.isfinite(result.enrichment_cv)
    assert np.isfinite(result.enrichment_retrained)


def test_enrichment_uses_probability_ranking_against_random_baseline():
    hidden = np.array([True, True, False, False, False, False])
    observed_negative = np.ones(6, dtype=bool)
    scores = np.array([0.99, 0.98, 0.50, 0.40, 0.30, 0.20])

    enrichment = DarkNumberValidationHarness._enrichment(hidden, observed_negative, scores)

    # Top-k finds both hidden positives. Random precision is 2 / 6.
    assert enrichment == pytest.approx(3.0)



def test_validation_correction_uses_regression_when_direct_estimate_is_unusable(monkeypatch):
    class FakeDirectEstimator:
        def __init__(self, *args, **kwargs):
            self.is_valid_for_regression_ = False
            self.correction_status_ = "zero_recovery"

        def fit(self, X, y):
            return self

        def score(self, X=None, y=None):
            return np.inf

    class FakeRegressor:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def score(self, X=None, y=None):
            return 2.25

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", FakeDirectEstimator)
    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorRegressor", FakeRegressor)

    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        correction_n_splits=3,
        correction_n_repeats=1,
    )
    X, y = _make_data()

    corr, source = harness._estimate_correction(LogisticRegression(max_iter=2000), X, y, 42)

    assert corr == pytest.approx(2.25)
    assert source == "regressed"

def test_summary_reports_coverage_position_corr_stability_and_enrichment():
    runs = pd.DataFrame(
        {
            "interval_covered": [True, True, False],
            "interval_position": [0.25, 0.75, 1.20],
            "corr_cv": [2.0, 2.2, 1.8],
            "corr_retrained": [1.5, 1.6, 1.4],
            "corr_cv_source": ["direct", "regressed", "direct"],
            "corr_retrained_source": ["direct", "direct", "fallback/unestimable"],
            "enrichment_cv": [3.0, 2.5, 3.5],
            "enrichment_retrained": [2.0, 2.2, 1.8],
        }
    )

    summary = DarkNumberValidationHarness.summarize_runs(runs)

    assert summary["coverage_rate"] == pytest.approx(2 / 3)
    assert summary["mean_interval_position_when_covered"] == pytest.approx(0.5)
    assert summary["corr_cv"]["mean"] == pytest.approx(2.0)
    assert summary["corr_cv"]["std"] > 0
    assert summary["corr_retrained"]["coefficient_of_variation"] > 0
    assert summary["corr_cv_sources"] == {"direct": 2, "regressed": 1}
    assert summary["corr_retrained_sources"] == {"direct": 2, "fallback/unestimable": 1}
    assert summary["mean_enrichment_cv"] == pytest.approx(3.0)
