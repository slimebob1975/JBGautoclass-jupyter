from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse as scipy_sparse
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from JBGDarkNumberCorrectionFactor import DarkNumberCorrectionFactorEstimator
from JBGDarkNumberCorrectionRegressor import DarkNumberCorrectionFactorRegressor
from JBGDarkNumbers import DarkNumberCalculator


class _NullLogger:
    def print_info(self, *_args, **_kwargs):
        return None

    def print_warning(self, *_args, **_kwargs):
        return None


@dataclass(frozen=True)
class DarkNumberValidationRun:
    random_state: int
    positive_class: Any
    negative_class: Any
    injected_noise_fraction: float
    correction_flip_fraction: float
    hidden_positive_count: int
    observed_negative_count: int
    true_dark_number: float
    corr_cv: float
    corr_cv_source: str
    corr_retrained: float
    corr_retrained_source: str
    d_cv_test: float
    d_cv_full: float
    d_retrained_full: float
    interval_lower: float
    interval_upper: float
    interval_covered: bool
    interval_position: float
    enrichment_cv: float
    enrichment_retrained: float


class DarkNumberValidationHarness:
    """Evaluate the existing Dark Number heuristic against injected, known label noise.

    This helper is intentionally separate from the production prediction flow. It does
    not alter the published Dark Number formula. Instead it creates a controlled
    validation problem where the hidden positives are known, then measures how the
    three estimates and model-specific correction factors behave.
    """

    def __init__(
        self,
        estimator,
        positive_class,
        *,
        negative_class=None,
        injected_noise_fraction: float = 0.20,
        correction_flip_fraction: float = 0.20,
        test_size: float = 0.20,
        correction_n_splits: int = 5,
        correction_n_repeats: int = 2,
        random_state: int = 42,
        enable_soft_same_model_fallback: bool = False,
        enable_adaptive_noise_fallback: bool = False,
        enable_perturbed_same_model_fallback: bool = False,
        perturbation_clones: int = 5,
        perturbation_min_valid: int = 3,
        perturbation_max_cv: float = 0.50,
        logger=None,
    ):
        self.estimator = estimator
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.injected_noise_fraction = float(injected_noise_fraction)
        self.correction_flip_fraction = float(correction_flip_fraction)
        self.test_size = float(test_size)
        self.correction_n_splits = int(correction_n_splits)
        self.correction_n_repeats = int(correction_n_repeats)
        self.random_state = int(random_state)
        self.enable_soft_same_model_fallback = bool(enable_soft_same_model_fallback)
        self.enable_adaptive_noise_fallback = bool(enable_adaptive_noise_fallback)
        self.enable_perturbed_same_model_fallback = bool(enable_perturbed_same_model_fallback)
        self.perturbation_clones = int(perturbation_clones)
        self.perturbation_min_valid = int(perturbation_min_valid)
        self.perturbation_max_cv = float(perturbation_max_cv)
        self.logger = logger or _NullLogger()

        for name, value in (
            ("injected_noise_fraction", self.injected_noise_fraction),
            ("correction_flip_fraction", self.correction_flip_fraction),
            ("test_size", self.test_size),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be greater than 0 and less than 1")
        if self.correction_n_splits < 2:
            raise ValueError("correction_n_splits must be at least 2")
        if self.correction_n_repeats < 1:
            raise ValueError("correction_n_repeats must be at least 1")
        if self.perturbation_clones < 1:
            raise ValueError("perturbation_clones must be at least 1")
        if self.perturbation_min_valid < 1 or self.perturbation_min_valid > self.perturbation_clones:
            raise ValueError("perturbation_min_valid must be between 1 and perturbation_clones")
        if not np.isfinite(self.perturbation_max_cv) or self.perturbation_max_cv < 0:
            raise ValueError("perturbation_max_cv must be finite and non-negative")

    @staticmethod
    def _clone_for_validation(estimator, random_state: int):
        """Clone an estimator and seed otherwise-unset random_state parameters.

        Paired validation must not let stochastic initialization differ between the
        baseline and fallback candidate. Explicitly configured random_state values
        are preserved; only None-valued parameters are filled with the validation
        seed.
        """
        seeded = clone(estimator)
        try:
            params = seeded.get_params(deep=True)
        except AttributeError:
            return seeded

        updates = {
            name: int(random_state)
            for name, value in params.items()
            if name.endswith("random_state") and value is None
        }
        if updates:
            seeded.set_params(**updates)
        return seeded

    @staticmethod
    def _clone_for_perturbation(estimator, random_state: int):
        """Clone an estimator and deliberately perturb all random_state parameters.

        This helper is validation-only. Unlike ``_clone_for_validation``, explicit
        seeds are overwritten because the purpose is to sample nearby stochastic
        realizations of the same estimator family and hyperparameters.
        """
        perturbed = clone(estimator)
        try:
            params = perturbed.get_params(deep=True)
        except AttributeError:
            return perturbed

        updates = {
            name: int(random_state)
            for name in params
            if name.endswith("random_state")
        }
        if updates:
            perturbed.set_params(**updates)
        return perturbed

    @staticmethod
    def _stratified_bootstrap_indices(y, random_state: int) -> np.ndarray:
        """Bootstrap each observed class independently while preserving class counts."""
        y_array = np.asarray(y)
        rng = np.random.default_rng(int(random_state))
        sampled = []
        for label in pd.unique(y_array):
            class_indices = np.flatnonzero(y_array == label)
            sampled.append(rng.choice(class_indices, size=class_indices.size, replace=True))
        indices = np.concatenate(sampled)
        rng.shuffle(indices)
        return indices.astype(int, copy=False)

    @staticmethod
    def _take_rows(X, indices):
        if hasattr(X, "iloc"):
            return X.iloc[indices]
        if scipy_sparse.issparse(X):
            return X[indices]
        return np.asarray(X)[indices]

    @staticmethod
    def _binary(series, positive_class) -> pd.Series:
        return pd.Series(np.asarray(series) == positive_class, index=getattr(series, "index", None)).astype(int)

    def inject_known_noise(self, y, random_state: int | None = None):
        y_array = np.asarray(y).copy()
        positive_indices = np.flatnonzero(y_array == self.positive_class)
        if positive_indices.size == 0:
            raise ValueError(f"No observations found for positive_class={self.positive_class!r}")

        negative_class = self.negative_class
        if negative_class is None:
            other_classes = [value for value in pd.unique(y_array) if value != self.positive_class]
            if not other_classes:
                raise ValueError("At least one negative class is required to inject label noise")
            negative_class = other_classes[0]

        if negative_class == self.positive_class:
            raise ValueError("negative_class must differ from positive_class")

        n_flip = int(positive_indices.size * self.injected_noise_fraction)
        if n_flip < 1:
            raise ValueError(
                "Injected noise fraction is too small to hide any positive observations; "
                "increase the fraction or use more positive examples"
            )

        seed = self.random_state if random_state is None else int(random_state)
        rng = np.random.default_rng(seed)
        hidden_indices = rng.choice(positive_indices, size=n_flip, replace=False)

        y_noisy = y_array.copy()
        y_noisy[hidden_indices] = negative_class
        hidden_mask = np.zeros(y_array.shape[0], dtype=bool)
        hidden_mask[hidden_indices] = True

        observed_negative_count = int(np.sum(y_noisy != self.positive_class))
        true_dark_number = n_flip / observed_negative_count

        return y_noisy, hidden_mask, negative_class, true_dark_number

    def _new_correction_estimator(self, model, random_state: int, predict_mode: str):
        return DarkNumberCorrectionFactorEstimator(
            estimator=clone(model),
            flip_fraction=self.correction_flip_fraction,
            n_splits=self.correction_n_splits,
            n_repeats=self.correction_n_repeats,
            n_jobs=1,
            predict_mode=predict_mode,
            random_state=random_state,
            positive_class=self.positive_class,
            sample_size=1.0,
            parallel_backend="threads",
            logger=self.logger,
        )

    @staticmethod
    def _validated_correction(estimator, label: str) -> float:
        if not getattr(estimator, "is_valid_for_regression_", True):
            raise ValueError(
                f"{label} correction-factor estimator did not produce an observed estimate "
                f"(status={getattr(estimator, 'correction_status_', 'unknown')})."
            )
        corr = float(estimator.score())
        if not np.isfinite(corr) or corr <= 0:
            raise ValueError(f"Invalid {label.lower()} correction factor: {corr}")
        return corr

    def _estimate_soft_same_model_correction(self, model, X, y, random_state: int) -> tuple[float, str]:
        if not hasattr(model, "predict_proba"):
            self.logger.print_warning(
                "Soft same-model correction fallback is unavailable because the target estimator "
                "does not expose predict_proba(); using 1.0 as fallback/unestimable."
            )
            return 1.0, "fallback/unestimable"

        estimator = self._new_correction_estimator(
            model, random_state=random_state, predict_mode="predict_proba"
        )
        try:
            estimator.fit(X, y)
            corr = self._validated_correction(estimator, "Soft same-model")
            return corr, "soft_same_model"
        except Exception as soft_error:
            self.logger.print_warning(
                f"Soft same-model correction fallback failed: {soft_error}. "
                "Using 1.0 as fallback/unestimable."
            )
            return 1.0, "fallback/unestimable"

    def _adaptive_noise_candidate_fractions(self) -> list[float]:
        """Return lower hard-label noise levels used to estimate the target response curve."""
        multipliers = (0.50, 0.625, 0.75, 0.875)
        return sorted({
            round(self.correction_flip_fraction * multiplier, 6)
            for multiplier in multipliers
            if 0.0 < self.correction_flip_fraction * multiplier < self.correction_flip_fraction
        })

    def _estimate_adaptive_noise_same_model_correction(
        self, model, X, y, random_state: int
    ) -> tuple[float, str]:
        """Estimate target hard-recovery correction from lower same-model noise levels.

        The method deliberately keeps the estimator, hard class decisions, data, CV
        structure and seeds unchanged. Only the injected correction-noise fraction is
        reduced. Recovery rates (1 / correction factor) are fitted against noise level
        and linearly extrapolated to the configured target noise. No estimate is emitted
        unless at least three lower-noise points are observed and the extrapolated
        recovery remains in the physical interval (0, 1].
        """
        points: list[tuple[float, float]] = []
        for flip_fraction in self._adaptive_noise_candidate_fractions():
            estimator = DarkNumberCorrectionFactorEstimator(
                estimator=clone(model),
                flip_fraction=flip_fraction,
                n_splits=self.correction_n_splits,
                n_repeats=self.correction_n_repeats,
                n_jobs=1,
                predict_mode="predict",
                random_state=random_state,
                positive_class=self.positive_class,
                sample_size=1.0,
                parallel_backend="threads",
                logger=self.logger,
            )
            try:
                estimator.fit(X, y)
                corr = self._validated_correction(
                    estimator, f"Adaptive-noise {flip_fraction:.3f}"
                )
            except Exception as error:
                self.logger.print_info(
                    f"Adaptive-noise same-model point {flip_fraction:.3f} was not usable: {error}"
                )
                continue

            recovery = 1.0 / corr
            if np.isfinite(recovery) and 0.0 < recovery <= 1.0:
                points.append((float(flip_fraction), float(recovery)))

        if len(points) < 3:
            self.logger.print_warning(
                "Adaptive-noise same-model fallback requires at least three valid lower-noise "
                f"hard-recovery points; got {len(points)}. Using 1.0 as fallback/unestimable."
            )
            return 1.0, "fallback/unestimable"

        fractions = np.asarray([point[0] for point in points], dtype=float)
        recoveries = np.asarray([point[1] for point in points], dtype=float)
        slope, intercept = np.polyfit(fractions, recoveries, 1)
        predicted_recovery = float(slope * self.correction_flip_fraction + intercept)

        if not np.isfinite(predicted_recovery) or not 0.0 < predicted_recovery <= 1.0:
            self.logger.print_warning(
                "Adaptive-noise same-model extrapolation did not produce a physical recovery "
                f"rate at target noise {self.correction_flip_fraction:.3f}: "
                f"{predicted_recovery}. Using 1.0 as fallback/unestimable."
            )
            return 1.0, "fallback/unestimable"

        corr = 1.0 / predicted_recovery
        self.logger.print_info(
            "Adaptive-noise same-model fallback used hard-recovery points "
            f"{[(round(f, 6), round(r, 6)) for f, r in points]} and extrapolated "
            f"target recovery={predicted_recovery:.6f}, corr={corr:.6f}."
        )
        return float(corr), "adaptive_noise_same_model"

    def _estimate_perturbed_same_model_correction(
        self, model, X, y, random_state: int
    ) -> tuple[float, str]:
        """Estimate correction from stable local shadow-clones of the same model.

        Each shadow clone keeps the estimator family and all ordinary hyperparameters,
        but receives a different random seed and a class-stratified bootstrap sample of
        the same size. The fallback is accepted only when enough clones independently
        produce finite hard-recovery correction factors and their relative dispersion
        is bounded. The robust median is returned; otherwise the result remains
        unestimable.
        """
        valid_corrs: list[float] = []
        for clone_index in range(self.perturbation_clones):
            perturb_seed = int(random_state) + 1009 * (clone_index + 1)
            indices = self._stratified_bootstrap_indices(y, perturb_seed)
            X_boot = self._take_rows(X, indices)
            y_boot = np.asarray(y)[indices]
            shadow = self._clone_for_perturbation(model, perturb_seed)
            estimator = DarkNumberCorrectionFactorEstimator(
                estimator=shadow,
                flip_fraction=self.correction_flip_fraction,
                n_splits=self.correction_n_splits,
                n_repeats=self.correction_n_repeats,
                n_jobs=1,
                predict_mode="predict",
                random_state=perturb_seed,
                positive_class=self.positive_class,
                sample_size=1.0,
                parallel_backend="threads",
                logger=self.logger,
            )
            try:
                estimator.fit(X_boot, y_boot)
                corr = self._validated_correction(
                    estimator, f"Perturbed same-model clone {clone_index + 1}"
                )
            except Exception as error:
                self.logger.print_info(
                    f"Perturbed same-model clone {clone_index + 1}/{self.perturbation_clones} "
                    f"was not usable: {error}"
                )
                continue
            valid_corrs.append(float(corr))

        if len(valid_corrs) < self.perturbation_min_valid:
            self.logger.print_warning(
                "Perturbed same-model fallback requires at least "
                f"{self.perturbation_min_valid} valid shadow-clone correction factors; "
                f"got {len(valid_corrs)} of {self.perturbation_clones}. "
                "Using 1.0 as fallback/unestimable."
            )
            return 1.0, "fallback/unestimable"

        values = np.asarray(valid_corrs, dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        coefficient_of_variation = std / mean if mean > 0 else float("inf")
        median = float(np.median(values))
        if (
            not np.isfinite(median)
            or median <= 0
            or not np.isfinite(coefficient_of_variation)
            or coefficient_of_variation > self.perturbation_max_cv
        ):
            self.logger.print_warning(
                "Perturbed same-model correction factors were not stable enough: "
                f"values={[round(value, 6) for value in valid_corrs]}, "
                f"cv={coefficient_of_variation:.6f}, limit={self.perturbation_max_cv:.6f}. "
                "Using 1.0 as fallback/unestimable."
            )
            return 1.0, "fallback/unestimable"

        self.logger.print_info(
            "Perturbed same-model fallback accepted stable shadow-clone correction factors "
            f"{[round(value, 6) for value in valid_corrs]}; "
            f"median={median:.6f}, cv={coefficient_of_variation:.6f}."
        )
        return median, "perturbed_same_model"

    def _estimate_correction(self, model, X, y, random_state: int) -> tuple[float, str]:
        recovery_level_unestimable_statuses = {
            "zero_recovery",
            "nonfinite",
            "nan_recovery",
        }
        statistically_unestimable_statuses = recovery_level_unestimable_statuses | {
            "insufficient_sample",
        }
        estimator = self._new_correction_estimator(
            model, random_state=random_state, predict_mode="predict"
        )

        direct_error = None
        try:
            estimator.fit(X, y)
            corr = self._validated_correction(estimator, "Direct")
            return corr, "direct"
        except Exception as ex:
            direct_error = ex

        status = getattr(estimator, "correction_status_", "unknown")
        if status in statistically_unestimable_statuses:
            if (
                self.enable_perturbed_same_model_fallback
                and status in recovery_level_unestimable_statuses
            ):
                self.logger.print_warning(
                    f"Direct correction-factor estimate is statistically unestimable (status={status}). "
                    "Trying perturbed shadow-clones with hard predictions from the same target estimator."
                )
                return self._estimate_perturbed_same_model_correction(
                    model, X, y, random_state
                )

            if (
                self.enable_adaptive_noise_fallback
                and status in recovery_level_unestimable_statuses
            ):
                self.logger.print_warning(
                    f"Direct correction-factor estimate is statistically unestimable (status={status}). "
                    "Trying lower injected-noise levels with hard predictions from the same target estimator."
                )
                return self._estimate_adaptive_noise_same_model_correction(
                    model, X, y, random_state
                )

            if (
                self.enable_soft_same_model_fallback
                and status in recovery_level_unestimable_statuses
            ):
                self.logger.print_warning(
                    f"Direct correction-factor estimate is statistically unestimable (status={status}). "
                    "Trying soft probability recovery with the same target estimator."
                )
                return self._estimate_soft_same_model_correction(model, X, y, random_state)

            self.logger.print_warning(
                f"Direct correction-factor estimate is statistically unestimable (status={status}). "
                "No validation-only statistical same-model fallback is enabled; "
                "using 1.0 as fallback/unestimable."
            )
            return 1.0, "fallback/unestimable"

        if not isinstance(direct_error, (MemoryError, SystemError)):
            self.logger.print_warning(
                f"Direct correction-factor estimate failed: {direct_error}. "
                "Sample-size regression was not attempted because the failure was not classified "
                "as resource-constrained; using 1.0 as fallback/unestimable."
            )
            return 1.0, "fallback/unestimable"

        self.logger.print_warning(
            f"Direct correction-factor estimate hit a resource/execution failure: {direct_error}. "
            "Trying regression over valid finite sample estimates with the same target estimator."
        )
        try:
            regressor = DarkNumberCorrectionFactorRegressor(
                estimator=clone(model),
                sample_size_list=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                flip_fraction=self.correction_flip_fraction,
                n_splits=self.correction_n_splits,
                n_repeats=self.correction_n_repeats,
                n_jobs=1,
                predict_mode="predict",
                random_state=random_state,
                positive_class=self.positive_class,
                type="logbounded",
                logger=self.logger,
            )
            regressor.fit(X, y)
            corr = float(regressor.score())
            if not np.isfinite(corr) or corr <= 0:
                raise ValueError(f"Invalid regressed correction factor: {corr}")
            return corr, "regressed_same_model"
        except Exception as regression_error:
            self.logger.print_warning(
                f"Correction-factor regression failed: {regression_error}. "
                "Using 1.0 as fallback/unestimable."
            )
            return 1.0, "fallback/unestimable"

    def _estimate_dark_number(self, y_real, y_pred, corr: float) -> float:
        calculator = DarkNumberCalculator()
        real_binary = self._binary(pd.Series(np.asarray(y_real)), self.positive_class)
        pred_binary = self._binary(pd.Series(np.asarray(y_pred)), self.positive_class)
        return float(calculator.compute_dark_number(real_binary, pred_binary, corr=corr))

    @staticmethod
    def _positive_probabilities(model, X, positive_class):
        if not hasattr(model, "predict_proba"):
            raise ValueError("Validation enrichment requires an estimator with predict_proba()")
        classes = list(model.classes_)
        if positive_class not in classes:
            raise ValueError(f"positive_class={positive_class!r} is not present in fitted model classes")
        class_index = classes.index(positive_class)
        return np.asarray(model.predict_proba(X))[:, class_index]

    @staticmethod
    def _enrichment(hidden_mask, observed_negative_mask, scores) -> float:
        candidate_indices = np.flatnonzero(observed_negative_mask)
        hidden_count = int(np.sum(hidden_mask[candidate_indices]))
        if hidden_count == 0 or candidate_indices.size == 0:
            return float("nan")

        candidate_scores = np.asarray(scores)[candidate_indices]
        order = np.argsort(candidate_scores)[::-1]
        top_indices = candidate_indices[order[:hidden_count]]
        precision_at_k = float(np.mean(hidden_mask[top_indices]))
        random_precision = hidden_count / candidate_indices.size
        return precision_at_k / random_precision if random_precision > 0 else float("nan")

    @staticmethod
    def _interval_position(truth: float, lower: float, upper: float) -> float:
        width = upper - lower
        if width == 0:
            return 0.5 if truth == lower else float("nan")
        return (truth - lower) / width

    def run_once(self, X, y, random_state: int | None = None) -> DarkNumberValidationRun:
        seed = self.random_state if random_state is None else int(random_state)
        y_clean = np.asarray(y)
        y_noisy, hidden_mask, negative_class, truth = self.inject_known_noise(y_clean, seed)

        indices = np.arange(y_noisy.shape[0])
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=y_noisy,
            random_state=seed,
        )

        X_train = self._take_rows(X, train_idx)
        X_test = self._take_rows(X, test_idx)
        y_train = y_noisy[train_idx]
        y_test = y_noisy[test_idx]

        cv_model = self._clone_for_validation(self.estimator, seed)
        cv_model.fit(X_train, y_train)

        retrained_model = self._clone_for_validation(self.estimator, seed)
        retrained_model.fit(X, y_noisy)

        corr_cv, corr_cv_source = self._estimate_correction(cv_model, X_train, y_train, seed)
        corr_retrained, corr_retrained_source = self._estimate_correction(retrained_model, X, y_noisy, seed)

        pred_cv_test = cv_model.predict(X_test)
        pred_cv_full = cv_model.predict(X)
        pred_retrained_full = retrained_model.predict(X)

        d_cv_test = self._estimate_dark_number(y_test, pred_cv_test, corr_cv)
        d_cv_full = self._estimate_dark_number(y_noisy, pred_cv_full, corr_cv)
        d_retrained_full = self._estimate_dark_number(y_noisy, pred_retrained_full, corr_retrained)

        estimates = np.array([d_cv_test, d_cv_full, d_retrained_full], dtype=float)
        interval_lower = float(np.min(estimates))
        interval_upper = float(np.max(estimates))
        interval_covered = bool(interval_lower <= truth <= interval_upper)
        interval_position = float(self._interval_position(truth, interval_lower, interval_upper))

        observed_negative_mask = y_noisy != self.positive_class
        prob_cv = self._positive_probabilities(cv_model, X, self.positive_class)
        prob_retrained = self._positive_probabilities(retrained_model, X, self.positive_class)
        enrichment_cv = float(self._enrichment(hidden_mask, observed_negative_mask, prob_cv))
        enrichment_retrained = float(self._enrichment(hidden_mask, observed_negative_mask, prob_retrained))

        return DarkNumberValidationRun(
            random_state=seed,
            positive_class=self.positive_class,
            negative_class=negative_class,
            injected_noise_fraction=self.injected_noise_fraction,
            correction_flip_fraction=self.correction_flip_fraction,
            hidden_positive_count=int(np.sum(hidden_mask)),
            observed_negative_count=int(np.sum(observed_negative_mask)),
            true_dark_number=float(truth),
            corr_cv=corr_cv,
            corr_cv_source=corr_cv_source,
            corr_retrained=corr_retrained,
            corr_retrained_source=corr_retrained_source,
            d_cv_test=d_cv_test,
            d_cv_full=d_cv_full,
            d_retrained_full=d_retrained_full,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            interval_covered=interval_covered,
            interval_position=interval_position,
            enrichment_cv=enrichment_cv,
            enrichment_retrained=enrichment_retrained,
        )

    def run_repeated(self, X, y, n_runs: int = 5):
        if n_runs < 1:
            raise ValueError("n_runs must be at least 1")

        rows = [
            asdict(self.run_once(X, y, random_state=self.random_state + run_index))
            for run_index in range(n_runs)
        ]
        runs = pd.DataFrame(rows)
        return runs, self.summarize_runs(runs)

    def _spawn(
        self, *, enable_soft_same_model_fallback: bool = False,
        enable_adaptive_noise_fallback: bool = False,
        enable_perturbed_same_model_fallback: bool = False,
    ):
        return DarkNumberValidationHarness(
            estimator=self.estimator,
            positive_class=self.positive_class,
            negative_class=self.negative_class,
            injected_noise_fraction=self.injected_noise_fraction,
            correction_flip_fraction=self.correction_flip_fraction,
            test_size=self.test_size,
            correction_n_splits=self.correction_n_splits,
            correction_n_repeats=self.correction_n_repeats,
            random_state=self.random_state,
            enable_soft_same_model_fallback=enable_soft_same_model_fallback,
            enable_adaptive_noise_fallback=enable_adaptive_noise_fallback,
            enable_perturbed_same_model_fallback=enable_perturbed_same_model_fallback,
            perturbation_clones=self.perturbation_clones,
            perturbation_min_valid=self.perturbation_min_valid,
            perturbation_max_cv=self.perturbation_max_cv,
            logger=self.logger,
        )

    def compare_soft_same_model_fallback(self, X, y, n_runs: int = 5):
        """Run paired seeds with and without the validation-only soft fallback.

        The candidate path is allowed to differ from baseline only when the normal
        hard/direct correction is statistically unestimable. The same validation
        seed is also injected into otherwise-unset estimator random_state parameters
        so stochastic initialization cannot confound the comparison. Production
        behavior is not changed by this helper.
        """
        baseline_runs, baseline_summary = self._spawn(
            enable_soft_same_model_fallback=False
        ).run_repeated(X, y, n_runs=n_runs)
        candidate_runs, candidate_summary = self._spawn(
            enable_soft_same_model_fallback=True
        ).run_repeated(X, y, n_runs=n_runs)

        if not baseline_runs["random_state"].equals(candidate_runs["random_state"]):
            raise RuntimeError("Paired Dark Number validation runs must use identical random states")

        paired = pd.DataFrame(
            {
                "random_state": baseline_runs["random_state"],
                "true_dark_number": baseline_runs["true_dark_number"],
                "baseline_corr_cv_source": baseline_runs["corr_cv_source"],
                "candidate_corr_cv_source": candidate_runs["corr_cv_source"],
                "baseline_corr_retrained_source": baseline_runs["corr_retrained_source"],
                "candidate_corr_retrained_source": candidate_runs["corr_retrained_source"],
                "baseline_interval_covered": baseline_runs["interval_covered"],
                "candidate_interval_covered": candidate_runs["interval_covered"],
            }
        )

        estimate_columns = ("d_cv_test", "d_cv_full", "d_retrained_full")
        baseline_mae = {}
        candidate_mae = {}
        truth = pd.to_numeric(baseline_runs["true_dark_number"], errors="coerce")
        for column in estimate_columns:
            baseline_values = pd.to_numeric(baseline_runs[column], errors="coerce")
            candidate_values = pd.to_numeric(candidate_runs[column], errors="coerce")
            paired[f"baseline_{column}"] = baseline_values
            paired[f"candidate_{column}"] = candidate_values
            paired[f"baseline_abs_error_{column}"] = (baseline_values - truth).abs()
            paired[f"candidate_abs_error_{column}"] = (candidate_values - truth).abs()
            baseline_mae[column] = float(paired[f"baseline_abs_error_{column}"].mean())
            candidate_mae[column] = float(paired[f"candidate_abs_error_{column}"].mean())

        candidate_sources = pd.concat(
            [candidate_runs["corr_cv_source"], candidate_runs["corr_retrained_source"]],
            ignore_index=True,
        )
        baseline_sources = pd.concat(
            [baseline_runs["corr_cv_source"], baseline_runs["corr_retrained_source"]],
            ignore_index=True,
        )
        baseline_unestimable_rate = float(
            (baseline_sources == "fallback/unestimable").mean()
        )
        candidate_unestimable_rate = float(
            (candidate_sources == "fallback/unestimable").mean()
        )
        comparison_summary = {
            "n_runs": int(n_runs),
            "baseline": baseline_summary,
            "soft_same_model": candidate_summary,
            "baseline_mean_absolute_error": baseline_mae,
            "soft_same_model_mean_absolute_error": candidate_mae,
            "mean_absolute_error_delta": {
                column: candidate_mae[column] - baseline_mae[column]
                for column in estimate_columns
            },
            "coverage_rate_delta": baseline_unestimable_rate - candidate_unestimable_rate,
            "estimability_rate_delta": baseline_unestimable_rate - candidate_unestimable_rate,
            "interval_coverage_rate_delta": (
                candidate_summary["coverage_rate"] - baseline_summary["coverage_rate"]
            ),
            "soft_same_model_activation_rate": float(
                (candidate_sources == "soft_same_model").mean()
            ),
            "baseline_unestimable_rate": baseline_unestimable_rate,
            "candidate_unestimable_rate": candidate_unestimable_rate,
        }
        return paired, comparison_summary

    def compare_adaptive_noise_fallback(self, X, y, n_runs: int = 5):
        """Run paired hard/direct baseline versus adaptive-noise same-model candidate."""
        baseline_runs, baseline_summary = self._spawn().run_repeated(X, y, n_runs=n_runs)
        candidate_runs, candidate_summary = self._spawn(
            enable_adaptive_noise_fallback=True
        ).run_repeated(X, y, n_runs=n_runs)

        if not baseline_runs["random_state"].equals(candidate_runs["random_state"]):
            raise RuntimeError("Paired Dark Number validation runs must use identical random states")

        paired = pd.DataFrame({
            "random_state": baseline_runs["random_state"],
            "true_dark_number": baseline_runs["true_dark_number"],
            "baseline_corr_cv": baseline_runs["corr_cv"],
            "candidate_corr_cv": candidate_runs["corr_cv"],
            "baseline_corr_cv_source": baseline_runs["corr_cv_source"],
            "candidate_corr_cv_source": candidate_runs["corr_cv_source"],
            "baseline_corr_retrained": baseline_runs["corr_retrained"],
            "candidate_corr_retrained": candidate_runs["corr_retrained"],
            "baseline_corr_retrained_source": baseline_runs["corr_retrained_source"],
            "candidate_corr_retrained_source": candidate_runs["corr_retrained_source"],
            "baseline_interval_covered": baseline_runs["interval_covered"],
            "candidate_interval_covered": candidate_runs["interval_covered"],
        })

        estimate_columns = ("d_cv_test", "d_cv_full", "d_retrained_full")
        baseline_mae = {}
        candidate_mae = {}
        truth = pd.to_numeric(baseline_runs["true_dark_number"], errors="coerce")
        for column in estimate_columns:
            baseline_values = pd.to_numeric(baseline_runs[column], errors="coerce")
            candidate_values = pd.to_numeric(candidate_runs[column], errors="coerce")
            paired[f"baseline_{column}"] = baseline_values
            paired[f"candidate_{column}"] = candidate_values
            paired[f"baseline_abs_error_{column}"] = (baseline_values - truth).abs()
            paired[f"candidate_abs_error_{column}"] = (candidate_values - truth).abs()
            baseline_mae[column] = float(paired[f"baseline_abs_error_{column}"].mean())
            candidate_mae[column] = float(paired[f"candidate_abs_error_{column}"].mean())

        candidate_sources = pd.concat(
            [candidate_runs["corr_cv_source"], candidate_runs["corr_retrained_source"]],
            ignore_index=True,
        )
        baseline_sources = pd.concat(
            [baseline_runs["corr_cv_source"], baseline_runs["corr_retrained_source"]],
            ignore_index=True,
        )
        baseline_unestimable_rate = float(
            (baseline_sources == "fallback/unestimable").mean()
        )
        candidate_unestimable_rate = float(
            (candidate_sources == "fallback/unestimable").mean()
        )
        summary = {
            "n_runs": int(n_runs),
            "baseline": baseline_summary,
            "adaptive_noise_same_model": candidate_summary,
            "baseline_mean_absolute_error": baseline_mae,
            "adaptive_noise_same_model_mean_absolute_error": candidate_mae,
            "mean_absolute_error_delta": {
                column: candidate_mae[column] - baseline_mae[column]
                for column in estimate_columns
            },
            "coverage_rate_delta": baseline_unestimable_rate - candidate_unestimable_rate,
            "estimability_rate_delta": baseline_unestimable_rate - candidate_unestimable_rate,
            "interval_coverage_rate_delta": (
                candidate_summary["coverage_rate"] - baseline_summary["coverage_rate"]
            ),
            "adaptive_noise_activation_rate": float(
                (candidate_sources == "adaptive_noise_same_model").mean()
            ),
            "baseline_unestimable_rate": baseline_unestimable_rate,
            "candidate_unestimable_rate": candidate_unestimable_rate,
        }
        return paired, summary

    @staticmethod
    def _estimable_error_profile(values, truth, estimable_mask) -> dict:
        """Summarize truth error only where the candidate produced an estimate.

        ``fallback/unestimable`` uses 1.0 as an internal sentinel in the validation
        harness. Robustness metrics must therefore exclude those rows rather than
        accidentally treating the sentinel as a real estimate.
        """
        numeric_values = pd.to_numeric(values, errors="coerce")
        numeric_truth = pd.to_numeric(truth, errors="coerce")
        mask = pd.Series(estimable_mask, index=numeric_values.index).astype(bool)
        mask &= numeric_values.notna() & numeric_truth.notna()
        signed_error = (numeric_values - numeric_truth)[mask]
        absolute_error = signed_error.abs()
        count = int(mask.sum())
        total = int(len(mask))
        if count == 0:
            return {
                "estimable_count": 0,
                "total_count": total,
                "estimable_rate": 0.0 if total else float("nan"),
                "mean_absolute_error": float("nan"),
                "median_absolute_error": float("nan"),
                "p90_absolute_error": float("nan"),
                "mean_signed_error": float("nan"),
                "signed_error_std": float("nan"),
            }
        return {
            "estimable_count": count,
            "total_count": total,
            "estimable_rate": count / total if total else float("nan"),
            "mean_absolute_error": float(absolute_error.mean()),
            "median_absolute_error": float(absolute_error.median()),
            "p90_absolute_error": float(absolute_error.quantile(0.90)),
            "mean_signed_error": float(signed_error.mean()),
            "signed_error_std": float(signed_error.std(ddof=0)),
        }

    @staticmethod
    def _correction_stability_profile(values, sources, accepted_source: str) -> dict:
        numeric_values = pd.to_numeric(values, errors="coerce")
        source_values = pd.Series(sources, index=numeric_values.index)
        accepted = numeric_values[(source_values == accepted_source) & numeric_values.notna()]
        count = int(len(accepted))
        total = int(len(numeric_values))
        if count == 0:
            return {
                "accepted_count": 0,
                "total_count": total,
                "activation_rate": 0.0 if total else float("nan"),
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "coefficient_of_variation": float("nan"),
                "iqr": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        mean = float(accepted.mean())
        std = float(accepted.std(ddof=0))
        q25 = float(accepted.quantile(0.25))
        q75 = float(accepted.quantile(0.75))
        return {
            "accepted_count": count,
            "total_count": total,
            "activation_rate": count / total if total else float("nan"),
            "mean": mean,
            "median": float(accepted.median()),
            "std": std,
            "coefficient_of_variation": std / mean if mean != 0 else float("nan"),
            "iqr": q75 - q25,
            "min": float(accepted.min()),
            "max": float(accepted.max()),
        }

    def compare_perturbed_same_model_fallback(self, X, y, n_runs: int = 5):
        """Run paired hard/direct baseline versus perturbed same-model candidate."""
        baseline_runs, baseline_summary = self._spawn().run_repeated(X, y, n_runs=n_runs)
        candidate_runs, candidate_summary = self._spawn(
            enable_perturbed_same_model_fallback=True
        ).run_repeated(X, y, n_runs=n_runs)

        if not baseline_runs["random_state"].equals(candidate_runs["random_state"]):
            raise RuntimeError("Paired Dark Number validation runs must use identical random states")

        paired = pd.DataFrame({
            "random_state": baseline_runs["random_state"],
            "true_dark_number": baseline_runs["true_dark_number"],
            "baseline_corr_cv": baseline_runs["corr_cv"],
            "candidate_corr_cv": candidate_runs["corr_cv"],
            "baseline_corr_cv_source": baseline_runs["corr_cv_source"],
            "candidate_corr_cv_source": candidate_runs["corr_cv_source"],
            "baseline_corr_retrained": baseline_runs["corr_retrained"],
            "candidate_corr_retrained": candidate_runs["corr_retrained"],
            "baseline_corr_retrained_source": baseline_runs["corr_retrained_source"],
            "candidate_corr_retrained_source": candidate_runs["corr_retrained_source"],
            "baseline_interval_covered": baseline_runs["interval_covered"],
            "candidate_interval_covered": candidate_runs["interval_covered"],
        })

        estimate_columns = ("d_cv_test", "d_cv_full", "d_retrained_full")
        baseline_mae = {}
        candidate_mae = {}
        truth = pd.to_numeric(baseline_runs["true_dark_number"], errors="coerce")
        for column in estimate_columns:
            baseline_values = pd.to_numeric(baseline_runs[column], errors="coerce")
            candidate_values = pd.to_numeric(candidate_runs[column], errors="coerce")
            paired[f"baseline_{column}"] = baseline_values
            paired[f"candidate_{column}"] = candidate_values
            paired[f"baseline_abs_error_{column}"] = (baseline_values - truth).abs()
            paired[f"candidate_abs_error_{column}"] = (candidate_values - truth).abs()
            baseline_mae[column] = float(paired[f"baseline_abs_error_{column}"].mean())
            candidate_mae[column] = float(paired[f"candidate_abs_error_{column}"].mean())

        candidate_sources = pd.concat(
            [candidate_runs["corr_cv_source"], candidate_runs["corr_retrained_source"]],
            ignore_index=True,
        )
        baseline_sources = pd.concat(
            [baseline_runs["corr_cv_source"], baseline_runs["corr_retrained_source"]],
            ignore_index=True,
        )
        baseline_unestimable_rate = float(
            (baseline_sources == "fallback/unestimable").mean()
        )
        candidate_unestimable_rate = float(
            (candidate_sources == "fallback/unestimable").mean()
        )
        cv_estimable = candidate_runs["corr_cv_source"] != "fallback/unestimable"
        retrained_estimable = candidate_runs["corr_retrained_source"] != "fallback/unestimable"
        robust_error_profiles = {
            "d_cv_test": self._estimable_error_profile(
                candidate_runs["d_cv_test"], truth, cv_estimable
            ),
            "d_cv_full": self._estimable_error_profile(
                candidate_runs["d_cv_full"], truth, cv_estimable
            ),
            "d_retrained_full": self._estimable_error_profile(
                candidate_runs["d_retrained_full"], truth, retrained_estimable
            ),
        }
        correction_stability = {
            "corr_cv": self._correction_stability_profile(
                candidate_runs["corr_cv"],
                candidate_runs["corr_cv_source"],
                "perturbed_same_model",
            ),
            "corr_retrained": self._correction_stability_profile(
                candidate_runs["corr_retrained"],
                candidate_runs["corr_retrained_source"],
                "perturbed_same_model",
            ),
        }

        summary = {
            "n_runs": int(n_runs),
            "baseline": baseline_summary,
            "perturbed_same_model": candidate_summary,
            "baseline_mean_absolute_error": baseline_mae,
            "perturbed_same_model_mean_absolute_error": candidate_mae,
            "mean_absolute_error_delta": {
                column: candidate_mae[column] - baseline_mae[column]
                for column in estimate_columns
            },
            "perturbed_estimable_error_profile": robust_error_profiles,
            "perturbed_correction_stability": correction_stability,
            "primary_full_data_metric": "d_cv_full",
            "coverage_rate_delta": baseline_unestimable_rate - candidate_unestimable_rate,
            "estimability_rate_delta": baseline_unestimable_rate - candidate_unestimable_rate,
            "interval_coverage_rate_delta": (
                candidate_summary["coverage_rate"] - baseline_summary["coverage_rate"]
            ),
            "perturbed_same_model_activation_rate": float(
                (candidate_sources == "perturbed_same_model").mean()
            ),
            "baseline_unestimable_rate": baseline_unestimable_rate,
            "candidate_unestimable_rate": candidate_unestimable_rate,
        }
        return paired, summary

    @staticmethod
    def summarize_runs(runs: pd.DataFrame) -> dict:
        if runs.empty:
            raise ValueError("At least one validation run is required")

        def stability(column):
            values = pd.to_numeric(runs[column], errors="coerce").dropna()
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            cv = std / mean if mean != 0 else float("nan")
            return {"mean": mean, "std": std, "coefficient_of_variation": cv}

        positions = pd.to_numeric(runs["interval_position"], errors="coerce")
        covered_positions = positions[runs["interval_covered"].astype(bool)]

        summary = {
            "n_runs": int(len(runs)),
            "coverage_rate": float(runs["interval_covered"].astype(float).mean()),
            "mean_interval_position_when_covered": float(covered_positions.mean()) if not covered_positions.empty else float("nan"),
            "corr_cv": stability("corr_cv"),
            "corr_retrained": stability("corr_retrained"),
            "mean_enrichment_cv": float(pd.to_numeric(runs["enrichment_cv"], errors="coerce").mean()),
            "mean_enrichment_retrained": float(pd.to_numeric(runs["enrichment_retrained"], errors="coerce").mean()),
        }
        if "corr_cv_source" in runs:
            summary["corr_cv_sources"] = runs["corr_cv_source"].value_counts(dropna=False).to_dict()
        if "corr_retrained_source" in runs:
            summary["corr_retrained_sources"] = runs["corr_retrained_source"].value_counts(dropna=False).to_dict()
        return summary
