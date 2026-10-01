from __future__ import annotations

import numpy as np
from scipy import sparse as scipy_sparse
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from joblib import Parallel, delayed
from pickle import PicklingError
from typing import TYPE_CHECKING
from time import perf_counter
from JBGDarkNumberExecution import FallbackProgress, resolve_fallback_workers, run_isolated_tasks

if TYPE_CHECKING:
    from JBGLogger import JBGLogger

DEBUG_LOGGING = False
BACKEND_PROCESSES = "processes"
BACKEND_THREADS = "threads"

class NaNValueError(Exception):
    """Raised when a NaN value is encountered."""
    pass

def evaluate_split(estimator, X, y, train_idx, repeat_idx,
                   flip_fraction, positive_class, predict_mode, random_state):
    """
    Top-level evaluation function for Parallel.
    """
    rng = np.random.default_rng(random_state + repeat_idx if random_state is not None else None)
    X_train, y_train = X[train_idx], y[train_idx]
    y_train_flipped = y_train.copy()

    pos_indices = np.where(y_train == positive_class)[0]
    n_flip = int(len(pos_indices) * flip_fraction)
    if n_flip == 0:
        return 0.0

    flip_indices = rng.choice(pos_indices, size=n_flip, replace=False)

    other_class = [c for c in np.unique(y_train) if c != positive_class]
    if not other_class:
        raise ValueError("Cannot find a negative class different from the positive_class.")
    y_train_flipped[flip_indices] = other_class[0]

    model = clone(estimator)
    model.fit(X_train, y_train_flipped)

    X_flipped = X_train[flip_indices]
    if predict_mode == 'predict':
        y_pred = model.predict(X_flipped)
        reidentified = np.sum(y_pred == positive_class)
    elif predict_mode == 'predict_proba':
        class_index = list(model.classes_).index(positive_class)
        y_proba = model.predict_proba(X_flipped)[:, class_index]
        reidentified = np.sum(y_proba)
    else:
        raise ValueError("predict_mode must be 'predict' or 'predict_proba'")

    return reidentified / n_flip


class DarkNumberCorrectionFactorEstimator(BaseEstimator):
    def __init__(self,
                 estimator,
                 flip_fraction: float = 0.1,
                 n_splits: int = 10,
                 n_repeats: int = 3,
                 n_jobs: int = None,
                 predict_mode: str = "predict",
                 random_state: int = 42,
                 positive_class: int = 1,
                 sample_size: float = 0.1,
                 parallel_backend: str = BACKEND_PROCESSES,
                 logger: JBGLogger = None,
                 progress=None,
                 isolated_execution: bool = False):
        self.estimator = estimator
        self.flip_fraction = flip_fraction
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.predict_mode = predict_mode
        self.random_state = random_state
        self.positive_class = positive_class
        self.sample_size = sample_size
        self.correction_factor_ = None
        self.correction_status_ = None
        self.is_valid_for_regression_ = False
        self.recovery_results_ = None
        self.mean_recovery_ = None
        self.parallel_backend = parallel_backend
        self.logger = logger
        self.progress = progress
        self.isolated_execution = isolated_execution

    @staticmethod
    def _compute_min_sample_size(y,
                                 flip_fraction: float,
                                 n_splits: int,
                                 positive_class: int,
                                 min_flips: int = 3,
                                 min_pos_per_fold: int = 2) -> int:
        total_positives = np.sum(y == positive_class)
        min_pos_required = int(np.ceil(min_flips / flip_fraction))
        min_pos_for_cv = n_splits * min_pos_per_fold
        required_positives = max(min_pos_required, min_pos_for_cv)

        pos_ratio = total_positives / len(y) if len(y) > 0 else 0
        if pos_ratio == 0:
            raise ValueError("No positive examples in data.")

        return int(np.ceil(required_positives / pos_ratio))

    def fit(self, X, y):
        # Preserve sparse text matrices; CSR supports the row indexing used below.
        X = X.tocsr() if scipy_sparse.issparse(X) else np.asarray(X)
        y = np.asarray(y)

        # Check minimum sample requirement
        min_sample_needed = self._compute_min_sample_size(
            y,
            flip_fraction=self.flip_fraction,
            n_splits=self.n_splits,
            positive_class=self.positive_class
        )

        n_samples = X.shape[0]
        effective_size = min(n_samples, int(n_samples * self.sample_size))
        if effective_size < min_sample_needed:
            if self.logger:
                self.logger.print_info(
                    f"[Warning] Sample size {self.sample_size} gives effective size {effective_size}, "
                    f"less than required {min_sample_needed}. Skipping."
                )
            # Keep the historical score() fallback for direct callers, but mark it as
            # synthetic so the regression fallback cannot mistake 1.0 for an observed
            # correction factor.
            self.correction_factor_ = 1.0
            self.correction_status_ = "insufficient_sample"
            self.is_valid_for_regression_ = False
            self.recovery_results_ = None
            self.mean_recovery_ = None
            return self

        if effective_size < n_samples:
            X_sub, _, y_sub, _ = train_test_split(
                X, y,
                train_size=effective_size,
                stratify=y,
                random_state=self.random_state
            )
            if DEBUG_LOGGING:
                self.logger.print_info(f"[DEBUG] Subsampled to {len(X_sub)} rows (sample_size={self.sample_size}).")
        else:
            X_sub, y_sub = X, y

        skf = StratifiedKFold(n_splits=self.n_splits,
                              shuffle=True,
                              random_state=self.random_state)

        # Build tasks once
        tasks = [
            delayed(evaluate_split)(
                self.estimator, X_sub, y_sub, train_idx, repeat_idx,
                self.flip_fraction, self.positive_class,
                self.predict_mode, self.random_state
            )
            for repeat_idx in range(self.n_repeats)
            for train_idx, _ in skf.split(X_sub, y_sub)
        ]

        # Try running tasks with retry logic
        results = self._run_parallel(tasks)
        self.recovery_results_ = [float(value) for value in results]

        # If result contains NaN values, raise ValueError
        if np.isnan(results).any():
            self.correction_status_ = "nan_recovery"
            self.is_valid_for_regression_ = False
            raise NaNValueError("Result contains NaN values, which is not allowed.")

        mean_r = float(np.mean(results))
        self.mean_recovery_ = mean_r
        self.correction_factor_ = 1.0 / mean_r if mean_r > 0 else np.inf
        if np.isfinite(self.correction_factor_) and self.correction_factor_ > 0:
            self.correction_status_ = "estimated"
            self.is_valid_for_regression_ = True
        else:
            self.correction_status_ = "zero_recovery" if mean_r <= 0 else "nonfinite"
            self.is_valid_for_regression_ = False

        if DEBUG_LOGGING and self.logger:
            self.logger.print_info(
                f"[DEBUG] Correction factor: {self.correction_factor_} from results {results}"
            )

        return self

    def _run_parallel(self, tasks):
        """
        Run tasks with retry logic, reusing the same task list.
        """
        if self.isolated_execution:
            results, self.n_jobs = run_isolated_tasks(
                tasks, self.n_jobs, progress=self.progress, logger=self.logger,
                workers_changed=lambda workers: setattr(self, "n_jobs", workers),
            )
            return results

        while True:
            try:
                if DEBUG_LOGGING:
                    self.logger.print_info(f"[DEBUG] Running Parallel with backend={self.parallel_backend}, n_jobs={self.n_jobs}")
                return Parallel(
                    n_jobs=self.n_jobs,
                    prefer=self.parallel_backend,
                    pre_dispatch="2*n_jobs",
                    batch_size="auto",
                    max_nbytes=None
                )(tasks)

            except (SystemError, MemoryError, PicklingError) as e:
                if DEBUG_LOGGING:
                    self.logger.print_info(f"[DEBUG] Parallel failed with {type(e).__name__}: {e}.")
                if self.n_jobs and self.n_jobs > 1:
                    self.n_jobs = max(1, self.n_jobs // 2)
                    if DEBUG_LOGGING:
                        self.logger.print_info(f"[DEBUG] Retrying with n_jobs={self.n_jobs}")
                else:
                    raise
            except (AttributeError, TypeError) as e:
                self.logger.print_info(f"[WARNING] Pickling/backend error: {e}")
                if self.parallel_backend == BACKEND_PROCESSES:
                    self.parallel_backend = BACKEND_THREADS
                    if DEBUG_LOGGING:
                        self.logger.print_info(f"[DEBUG] Switching backend to threads and retrying. Reason: {str(e)}")
                else:
                    raise
            except NaNValueError as e:
                if self.parallel_backend == BACKEND_THREADS:
                    self.parallel_backend = BACKEND_PROCESSES
                    if DEBUG_LOGGING:
                        self.logger.print_info(f"[DEBUG] Switching backend to processes and retrying. Reason: {str(e)}")
                else:
                    raise
            else:
                raise

    def score(self, X=None, y=None):
        return self.correction_factor_


def estimate_perturbed_same_model_correction(
    estimator,
    X,
    y,
    *,
    flip_fraction: float,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    positive_class,
    perturbation_clones: int = 5,
    perturbation_min_valid: int = 3,
    perturbation_max_cv: float = 0.50,
    logger=None,
    n_jobs: int = 1,
):
    """Estimate a correction factor from stable perturbed clones of the same model.

    This is the production counterpart of the 087/088 validation experiment. Each
    shadow clone keeps the estimator family and ordinary hyperparameters, but receives
    a different random seed and a class-stratified bootstrap sample. The candidate is
    accepted only when enough clones independently produce finite hard-recovery
    correction factors and their coefficient of variation is bounded.

    Returns ``(corr, source, details)``. ``details`` contains only aggregate/clone
    correction metadata; no source rows or raw text are retained.
    """
    if perturbation_clones < 1:
        raise ValueError("perturbation_clones must be at least 1")
    if perturbation_min_valid < 1 or perturbation_min_valid > perturbation_clones:
        raise ValueError("perturbation_min_valid must be between 1 and perturbation_clones")
    if not np.isfinite(perturbation_max_cv) or perturbation_max_cv < 0:
        raise ValueError("perturbation_max_cv must be finite and non-negative")

    fits_per_clone = int(n_splits) * int(n_repeats)
    if int(n_splits) < 2 or int(n_repeats) < 1:
        raise ValueError("n_splits must be at least 2 and n_repeats at least 1")
    workers, reason = resolve_fallback_workers(estimator, n_jobs, fits_per_clone)
    with FallbackProgress(logger, int(perturbation_clones), fits_per_clone, workers, reason) as progress:
        corr, source, details = _estimate_perturbed_same_model_correction(
            estimator, X, y, flip_fraction=flip_fraction, n_splits=n_splits,
            n_repeats=n_repeats, random_state=random_state, positive_class=positive_class,
            perturbation_clones=perturbation_clones, perturbation_min_valid=perturbation_min_valid,
            perturbation_max_cv=perturbation_max_cv, logger=logger,
            n_jobs=workers, progress=progress,
        )
    details.update({
        "planned_fits": progress.total,
        "completed_fits": progress.completed,
        "skipped_fits": progress.skipped,
        "execution_seconds": progress.seconds,
        "initial_workers": workers,
    })
    return corr, source, details


def _estimate_perturbed_same_model_correction(
    estimator, X, y, *, flip_fraction, n_splits, n_repeats, random_state,
    positive_class, perturbation_clones, perturbation_min_valid, perturbation_max_cv,
    logger, n_jobs, progress,
):
    """Keep the five-clone statistical experiment independent of execution policy."""
    y_array = np.asarray(y)
    valid_corrs: list[float] = []
    clone_results: list[dict] = []

    def take_rows(data, indices):
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        if scipy_sparse.issparse(data):
            return data[indices]
        return np.asarray(data)[indices]

    def perturbed_clone(source, seed: int):
        shadow = clone(source)
        try:
            params = shadow.get_params(deep=True)
        except AttributeError:
            return shadow
        updates = {
            name: int(seed)
            for name in params
            if name.endswith("random_state")
        }
        # Parallelism belongs to the independent fits, not nested ensemble jobs.
        updates.update({name: 1 for name, value in params.items()
                        if name.endswith("n_jobs") and value not in (None, 1)})
        if updates:
            shadow.set_params(**updates)
        return shadow

    def stratified_bootstrap_indices(seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        sampled = []
        for label in np.unique(y_array):
            class_indices = np.flatnonzero(y_array == label)
            sampled.append(rng.choice(class_indices, size=class_indices.size, replace=True))
        indices = np.concatenate(sampled)
        rng.shuffle(indices)
        return indices.astype(int, copy=False)

    for clone_index in range(int(perturbation_clones)):
        progress.start_clone(clone_index)
        clone_started = perf_counter()
        perturb_seed = int(random_state) + 1009 * (clone_index + 1)
        indices = stratified_bootstrap_indices(perturb_seed)
        X_boot = take_rows(X, indices)
        y_boot = y_array[indices]
        shadow = perturbed_clone(estimator, perturb_seed)
        correction_estimator = DarkNumberCorrectionFactorEstimator(
            estimator=shadow,
            flip_fraction=float(flip_fraction),
            n_splits=int(n_splits),
            n_repeats=int(n_repeats),
            n_jobs=n_jobs,
            predict_mode="predict",
            random_state=perturb_seed,
            positive_class=positive_class,
            sample_size=1.0,
            parallel_backend=BACKEND_PROCESSES,
            logger=logger,
            progress=progress.fit_completed,
            isolated_execution=True,
        )
        try:
            correction_estimator.fit(X_boot, y_boot)
            if not getattr(correction_estimator, "is_valid_for_regression_", True):
                raise ValueError(
                    "shadow-clone correction estimator did not produce an observed estimate "
                    f"(status={getattr(correction_estimator, 'correction_status_', 'unknown')})"
                )
            corr = float(correction_estimator.score())
            if not np.isfinite(corr) or corr <= 0:
                raise ValueError(f"invalid shadow-clone correction factor: {corr}")
        except Exception as error:
            clone_results.append({
                "clone": clone_index + 1,
                "seed": perturb_seed,
                "status": "unusable",
                "reason": str(error),
            })
            if logger:
                logger.print_info(
                    f"EXPERIMENTAL perturbed same-model clone {clone_index + 1}/"
                    f"{perturbation_clones} was not usable: {error}"
                )
            n_jobs = min(n_jobs, correction_estimator.n_jobs)
            progress.finish_clone(perf_counter() - clone_started, "unusable", n_jobs)
            continue

        valid_corrs.append(corr)
        clone_results.append({
            "clone": clone_index + 1,
            "seed": perturb_seed,
            "status": "estimated",
            "corr": corr,
        })
        n_jobs = min(n_jobs, correction_estimator.n_jobs)
        progress.finish_clone(perf_counter() - clone_started, f"corr={corr:.6g}", n_jobs)

    details = {
        "clone_count": int(perturbation_clones),
        "valid_clone_count": len(valid_corrs),
        "min_valid": int(perturbation_min_valid),
        "max_cv": float(perturbation_max_cv),
        "clone_corrections": list(valid_corrs),
        "clone_results": clone_results,
        "accepted": False,
        "reason": "insufficient_valid_clones",
        "final_workers": n_jobs,
    }

    if len(valid_corrs) < perturbation_min_valid:
        if logger:
            logger.print_warning(
                "EXPERIMENTAL perturbed same-model fallback rejected: requires at least "
                f"{perturbation_min_valid} valid shadow-clone correction factors; got "
                f"{len(valid_corrs)} of {perturbation_clones}."
            )
        return 1.0, "fallback/unestimable", details

    values = np.asarray(valid_corrs, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    coefficient_of_variation = std / mean if mean > 0 else float("inf")
    median = float(np.median(values))
    details.update({
        "median": median,
        "mean": mean,
        "std": std,
        "cv": coefficient_of_variation,
        "min": float(values.min()),
        "max": float(values.max()),
    })

    if (
        not np.isfinite(median)
        or median <= 0
        or not np.isfinite(coefficient_of_variation)
        or coefficient_of_variation > perturbation_max_cv
    ):
        details["reason"] = "unstable_correction_factors"
        if logger:
            logger.print_warning(
                "EXPERIMENTAL perturbed same-model fallback rejected as unstable: "
                f"values={[round(value, 6) for value in valid_corrs]}, "
                f"cv={coefficient_of_variation:.6f}, limit={perturbation_max_cv:.6f}."
            )
        return 1.0, "fallback/unestimable", details

    details["accepted"] = True
    details["reason"] = "accepted"
    if logger:
        logger.print_warning(
            "EXPERIMENTAL perturbed same-model fallback accepted stable shadow-clone "
            f"correction factors {[round(value, 6) for value in valid_corrs]}; "
            f"median={median:.6f}, cv={coefficient_of_variation:.6f}."
        )
    return median, "perturbed_same_model", details
