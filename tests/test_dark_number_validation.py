import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import JBGDarkNumberValidation as validation_module
from JBGDarkNumberValidation import DarkNumberValidationHarness


class _ConservativeProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """Always predicts negative, while retaining soft positive probability mass."""

    def __init__(self, positive_probability=0.25):
        self.positive_probability = positive_probability

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        if list(self.classes_) != [0, 1]:
            raise ValueError("Test classifier expects binary classes [0, 1]")
        positive = np.full(len(X), self.positive_probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


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


def test_validation_clone_seeds_unset_estimator_random_state_without_overwriting_explicit_seed():
    unset = MLPClassifier(random_state=None)
    seeded_unset = DarkNumberValidationHarness._clone_for_validation(unset, 17)

    assert unset.random_state is None
    assert seeded_unset.random_state == 17

    explicit = MLPClassifier(random_state=99)
    seeded_explicit = DarkNumberValidationHarness._clone_for_validation(explicit, 17)

    assert seeded_explicit.random_state == 99


def test_validation_perturbation_clone_overrides_random_state_without_mutating_source():
    source = MLPClassifier(random_state=99)

    perturbed = DarkNumberValidationHarness._clone_for_perturbation(source, 17)

    assert source.random_state == 99
    assert perturbed.random_state == 17


def test_validation_stratified_bootstrap_preserves_class_counts_and_is_reproducible():
    y = np.array([0] * 7 + [1] * 3)

    first = DarkNumberValidationHarness._stratified_bootstrap_indices(y, 123)
    second = DarkNumberValidationHarness._stratified_bootstrap_indices(y, 123)

    assert np.array_equal(first, second)
    assert len(first) == len(y)
    assert np.sum(y[first] == 0) == 7
    assert np.sum(y[first] == 1) == 3


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
    assert result.corr_cv_source in {"direct", "soft_same_model", "adaptive_noise_same_model", "perturbed_same_model", "regressed_same_model", "fallback/unestimable"}
    assert result.corr_retrained >= 1.0
    assert result.corr_retrained_source in {"direct", "soft_same_model", "adaptive_noise_same_model", "perturbed_same_model", "regressed_same_model", "fallback/unestimable"}
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



def test_validation_correction_does_not_regress_zero_recovery(monkeypatch):
    class FakeDirectEstimator:
        def __init__(self, *args, **kwargs):
            self.is_valid_for_regression_ = False
            self.correction_status_ = "zero_recovery"

        def fit(self, X, y):
            return self

        def score(self, X=None, y=None):
            return np.inf

    class ForbiddenRegressor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("zero recovery must not trigger sample-size regression")

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", FakeDirectEstimator)
    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorRegressor", ForbiddenRegressor)

    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        correction_n_splits=3,
        correction_n_repeats=1,
    )
    X, y = _make_data()

    corr, source = harness._estimate_correction(LogisticRegression(max_iter=2000), X, y, 42)

    assert corr == pytest.approx(1.0)
    assert source == "fallback/unestimable"


def test_validation_soft_same_model_fallback_recovers_probability_signal_after_zero_hard_recovery():
    X, y = _make_data()
    harness = DarkNumberValidationHarness(
        _ConservativeProbabilityClassifier(positive_probability=0.25),
        positive_class=1,
        correction_n_splits=3,
        correction_n_repeats=1,
        enable_soft_same_model_fallback=True,
    )

    corr, source = harness._estimate_correction(
        _ConservativeProbabilityClassifier(positive_probability=0.25), X, y, 42
    )

    assert source == "soft_same_model"
    assert corr == pytest.approx(4.0)


def test_validation_direct_correction_always_precedes_soft_same_model_fallback(monkeypatch):
    modes = []

    class FakeEstimator:
        def __init__(self, *args, predict_mode, **kwargs):
            modes.append(predict_mode)
            if predict_mode != "predict":
                raise AssertionError("soft fallback must not run after a valid direct estimate")
            self.is_valid_for_regression_ = True
            self.correction_status_ = "estimated"

        def fit(self, X, y):
            return self

        def score(self, X=None, y=None):
            return 2.5

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", FakeEstimator)
    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        correction_n_splits=3,
        correction_n_repeats=1,
        enable_soft_same_model_fallback=True,
    )
    X, y = _make_data()

    corr, source = harness._estimate_correction(LogisticRegression(max_iter=2000), X, y, 42)

    assert corr == pytest.approx(2.5)
    assert source == "direct"
    assert modes == ["predict"]


def test_validation_insufficient_sample_does_not_try_soft_same_model(monkeypatch):
    modes = []

    class InsufficientSampleEstimator:
        def __init__(self, *args, predict_mode, **kwargs):
            modes.append(predict_mode)
            if predict_mode != "predict":
                raise AssertionError("insufficient sample cannot be repaired by soft recovery")
            self.is_valid_for_regression_ = False
            self.correction_status_ = "insufficient_sample"

        def fit(self, X, y):
            return self

        def score(self, X=None, y=None):
            return 1.0

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", InsufficientSampleEstimator)
    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        correction_n_splits=3,
        correction_n_repeats=1,
        enable_soft_same_model_fallback=True,
    )
    X, y = _make_data()

    corr, source = harness._estimate_correction(LogisticRegression(max_iter=2000), X, y, 42)

    assert corr == pytest.approx(1.0)
    assert source == "fallback/unestimable"
    assert modes == ["predict"]


def test_validation_soft_same_model_fallback_can_remain_unestimable(monkeypatch):
    class AlwaysUnestimableEstimator:
        def __init__(self, *args, predict_mode, **kwargs):
            self.predict_mode = predict_mode
            self.is_valid_for_regression_ = False
            self.correction_status_ = "zero_recovery"

        def fit(self, X, y):
            return self

        def score(self, X=None, y=None):
            return np.inf

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", AlwaysUnestimableEstimator)
    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        correction_n_splits=3,
        correction_n_repeats=1,
        enable_soft_same_model_fallback=True,
    )
    X, y = _make_data()

    corr, source = harness._estimate_correction(LogisticRegression(max_iter=2000), X, y, 42)

    assert corr == pytest.approx(1.0)
    assert source == "fallback/unestimable"


def test_soft_same_model_comparison_uses_paired_seeds_and_reports_activation():
    X, y = _make_data()
    harness = DarkNumberValidationHarness(
        _ConservativeProbabilityClassifier(positive_probability=0.25),
        positive_class=1,
        injected_noise_fraction=0.15,
        correction_n_splits=3,
        correction_n_repeats=1,
        random_state=31,
    )

    paired, summary = harness.compare_soft_same_model_fallback(X, y, n_runs=2)

    assert paired["random_state"].tolist() == [31, 32]
    assert summary["baseline_unestimable_rate"] == pytest.approx(1.0)
    assert summary["soft_same_model_activation_rate"] == pytest.approx(1.0)
    assert summary["candidate_unestimable_rate"] == pytest.approx(0.0)
    assert summary["coverage_rate_delta"] == pytest.approx(1.0)
    assert summary["estimability_rate_delta"] == pytest.approx(1.0)
    assert summary["soft_same_model"]["corr_cv_sources"] == {"soft_same_model": 2}
    assert summary["soft_same_model"]["corr_retrained_sources"] == {"soft_same_model": 2}






def test_validation_direct_correction_always_precedes_adaptive_noise_fallback(monkeypatch):
    flip_fractions = []

    class ValidDirectEstimator:
        def __init__(self, *args, flip_fraction, **kwargs):
            flip_fractions.append(float(flip_fraction))
            self.is_valid_for_regression_ = True
            self.correction_status_ = "estimated"
        def fit(self, X, y):
            return self
        def score(self, X=None, y=None):
            return 2.5

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", ValidDirectEstimator)
    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        correction_flip_fraction=0.2,
        correction_n_splits=3,
        correction_n_repeats=1,
        enable_adaptive_noise_fallback=True,
    )
    X, y = _make_data()

    corr, source = harness._estimate_correction(LogisticRegression(max_iter=2000), X, y, 42)

    assert corr == pytest.approx(2.5)
    assert source == "direct"
    assert flip_fractions == [0.2]


def test_validation_insufficient_sample_does_not_try_adaptive_noise(monkeypatch):
    flip_fractions = []

    class InsufficientSampleEstimator:
        def __init__(self, *args, flip_fraction, **kwargs):
            flip_fractions.append(float(flip_fraction))
            self.is_valid_for_regression_ = False
            self.correction_status_ = "insufficient_sample"
        def fit(self, X, y):
            return self
        def score(self, X=None, y=None):
            return 1.0

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", InsufficientSampleEstimator)
    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        correction_flip_fraction=0.2,
        correction_n_splits=3,
        correction_n_repeats=1,
        enable_adaptive_noise_fallback=True,
    )
    X, y = _make_data()

    corr, source = harness._estimate_correction(LogisticRegression(max_iter=2000), X, y, 42)

    assert corr == pytest.approx(1.0)
    assert source == "fallback/unestimable"
    assert flip_fractions == [0.2]

def test_validation_adaptive_noise_extrapolates_hard_recovery_with_same_model(monkeypatch):
    class NoiseResponseEstimator:
        def __init__(self, *args, flip_fraction, **kwargs):
            self.flip_fraction = float(flip_fraction)
            if self.flip_fraction >= 0.2 - 1e-12:
                self.is_valid_for_regression_ = False
                self.correction_status_ = "zero_recovery"
                self._score = np.inf
            else:
                recovery = 0.6 - 2.0 * self.flip_fraction
                self.is_valid_for_regression_ = True
                self.correction_status_ = "estimated"
                self._score = 1.0 / recovery

        def fit(self, X, y):
            return self

        def score(self, X=None, y=None):
            return self._score

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", NoiseResponseEstimator)
    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        correction_flip_fraction=0.2,
        correction_n_splits=3,
        correction_n_repeats=1,
        enable_adaptive_noise_fallback=True,
    )
    X, y = _make_data()

    corr, source = harness._estimate_correction(LogisticRegression(max_iter=2000), X, y, 42)

    assert source == "adaptive_noise_same_model"
    assert corr == pytest.approx(5.0)


def test_validation_adaptive_noise_requires_three_valid_lower_noise_points(monkeypatch):
    class SparseNoiseResponseEstimator:
        def __init__(self, *args, flip_fraction, **kwargs):
            self.flip_fraction = float(flip_fraction)
            valid = self.flip_fraction in {0.15, 0.175}
            self.is_valid_for_regression_ = valid
            self.correction_status_ = "estimated" if valid else "zero_recovery"
            self._score = 4.0 if valid else np.inf

        def fit(self, X, y):
            return self

        def score(self, X=None, y=None):
            return self._score

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", SparseNoiseResponseEstimator)
    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        correction_flip_fraction=0.2,
        correction_n_splits=3,
        correction_n_repeats=1,
        enable_adaptive_noise_fallback=True,
    )
    X, y = _make_data()

    corr, source = harness._estimate_correction(LogisticRegression(max_iter=2000), X, y, 42)

    assert corr == pytest.approx(1.0)
    assert source == "fallback/unestimable"

def test_validation_perturbed_same_model_accepts_stable_shadow_clone_corrections(monkeypatch):
    scores = iter([np.inf, 4.0, 4.2, 3.8, 4.1, np.inf])

    class PerturbationEstimator:
        def __init__(self, *args, **kwargs):
            self._score = next(scores)
            valid = np.isfinite(self._score)
            self.is_valid_for_regression_ = valid
            self.correction_status_ = "estimated" if valid else "zero_recovery"

        def fit(self, X, y):
            return self

        def score(self, X=None, y=None):
            return self._score

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", PerturbationEstimator)
    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        correction_n_splits=3,
        correction_n_repeats=1,
        enable_perturbed_same_model_fallback=True,
        perturbation_clones=5,
        perturbation_min_valid=3,
        perturbation_max_cv=0.2,
    )
    X, y = _make_data()

    corr, source = harness._estimate_correction(LogisticRegression(max_iter=2000), X, y, 42)

    assert source == "perturbed_same_model"
    assert corr == pytest.approx(4.05)


def test_validation_perturbed_same_model_rejects_unstable_shadow_clones(monkeypatch):
    scores = iter([np.inf, 1.0, 2.0, 8.0, 16.0, 32.0])

    class PerturbationEstimator:
        def __init__(self, *args, **kwargs):
            self._score = next(scores)
            valid = np.isfinite(self._score)
            self.is_valid_for_regression_ = valid
            self.correction_status_ = "estimated" if valid else "zero_recovery"

        def fit(self, X, y):
            return self

        def score(self, X=None, y=None):
            return self._score

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", PerturbationEstimator)
    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        correction_n_splits=3,
        correction_n_repeats=1,
        enable_perturbed_same_model_fallback=True,
        perturbation_clones=5,
        perturbation_min_valid=3,
        perturbation_max_cv=0.2,
    )
    X, y = _make_data()

    corr, source = harness._estimate_correction(LogisticRegression(max_iter=2000), X, y, 42)

    assert corr == pytest.approx(1.0)
    assert source == "fallback/unestimable"


def test_validation_direct_correction_precedes_perturbed_same_model(monkeypatch):
    class DirectEstimator:
        calls = 0

        def __init__(self, *args, **kwargs):
            self.is_valid_for_regression_ = True
            self.correction_status_ = "estimated"

        def fit(self, X, y):
            self.__class__.calls += 1
            return self

        def score(self, X=None, y=None):
            return 2.0

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", DirectEstimator)
    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        correction_n_splits=3,
        correction_n_repeats=1,
        enable_perturbed_same_model_fallback=True,
    )
    X, y = _make_data()

    corr, source = harness._estimate_correction(LogisticRegression(max_iter=2000), X, y, 42)

    assert corr == pytest.approx(2.0)
    assert source == "direct"
    assert DirectEstimator.calls == 1


def test_validation_insufficient_sample_does_not_try_perturbed_same_model(monkeypatch):
    class InsufficientEstimator:
        calls = 0

        def __init__(self, *args, **kwargs):
            self.is_valid_for_regression_ = False
            self.correction_status_ = "insufficient_sample"

        def fit(self, X, y):
            self.__class__.calls += 1
            return self

        def score(self, X=None, y=None):
            return np.inf

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", InsufficientEstimator)
    harness = DarkNumberValidationHarness(
        LogisticRegression(max_iter=2000),
        positive_class=1,
        correction_n_splits=3,
        correction_n_repeats=1,
        enable_perturbed_same_model_fallback=True,
    )
    X, y = _make_data()

    corr, source = harness._estimate_correction(LogisticRegression(max_iter=2000), X, y, 42)

    assert corr == pytest.approx(1.0)
    assert source == "fallback/unestimable"
    assert InsufficientEstimator.calls == 1


def test_validation_correction_regresses_same_model_for_resource_failure(monkeypatch):
    class ResourceFailingDirectEstimator:
        def __init__(self, *args, **kwargs):
            self.is_valid_for_regression_ = True
            self.correction_status_ = "unknown"

        def fit(self, X, y):
            raise MemoryError("simulated memory pressure")

    class FakeRegressor:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def score(self, X=None, y=None):
            return 2.25

    monkeypatch.setattr(validation_module, "DarkNumberCorrectionFactorEstimator", ResourceFailingDirectEstimator)
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
    assert source == "regressed_same_model"

def test_summary_reports_coverage_position_corr_stability_and_enrichment():
    runs = pd.DataFrame(
        {
            "interval_covered": [True, True, False],
            "interval_position": [0.25, 0.75, 1.20],
            "corr_cv": [2.0, 2.2, 1.8],
            "corr_retrained": [1.5, 1.6, 1.4],
            "corr_cv_source": ["direct", "regressed_same_model", "direct"],
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
    assert summary["corr_cv_sources"] == {"direct": 2, "regressed_same_model": 1}
    assert summary["corr_retrained_sources"] == {"direct": 2, "fallback/unestimable": 1}
    assert summary["mean_enrichment_cv"] == pytest.approx(3.0)


def test_estimable_error_profile_excludes_unestimable_sentinel_rows():
    values = pd.Series([1.0, 0.20, 0.40])
    truth = pd.Series([0.10, 0.10, 0.10])
    estimable = pd.Series([False, True, True])

    profile = DarkNumberValidationHarness._estimable_error_profile(
        values, truth, estimable
    )

    assert profile["estimable_count"] == 2
    assert profile["total_count"] == 3
    assert profile["estimable_rate"] == pytest.approx(2 / 3)
    assert profile["mean_absolute_error"] == pytest.approx(0.20)
    assert profile["median_absolute_error"] == pytest.approx(0.20)
    assert profile["mean_signed_error"] == pytest.approx(0.20)


def test_correction_stability_profile_uses_only_accepted_perturbed_sources():
    values = pd.Series([1.0, 3.0, 4.0, 99.0])
    sources = pd.Series([
        "fallback/unestimable",
        "perturbed_same_model",
        "perturbed_same_model",
        "direct",
    ])

    profile = DarkNumberValidationHarness._correction_stability_profile(
        values, sources, "perturbed_same_model"
    )

    assert profile["accepted_count"] == 2
    assert profile["activation_rate"] == pytest.approx(0.5)
    assert profile["median"] == pytest.approx(3.5)
    assert profile["mean"] == pytest.approx(3.5)
    assert profile["min"] == pytest.approx(3.0)
    assert profile["max"] == pytest.approx(4.0)
