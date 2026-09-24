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

    def _estimate_correction(self, model, X, y, random_state: int) -> tuple[float, str]:
        try:
            estimator = DarkNumberCorrectionFactorEstimator(
                estimator=clone(model),
                flip_fraction=self.correction_flip_fraction,
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
            estimator.fit(X, y)
            if not getattr(estimator, "is_valid_for_regression_", True):
                raise ValueError(
                    "Direct correction-factor estimator did not produce an observed estimate "
                    f"(status={getattr(estimator, 'correction_status_', 'unknown')})."
                )
            corr = float(estimator.score())
            if not np.isfinite(corr) or corr <= 0:
                raise ValueError(f"Invalid direct correction factor: {corr}")
            return corr, "direct"
        except Exception as direct_error:
            self.logger.print_warning(
                f"Direct correction-factor estimate failed: {direct_error}. "
                "Trying regression over valid finite sample estimates."
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
            return corr, "regressed"
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

        cv_model = clone(self.estimator)
        cv_model.fit(X_train, y_train)

        retrained_model = clone(self.estimator)
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
