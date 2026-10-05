"""Validation-only sensitivity study for the Dark Number correction flip fraction.

The experiment deliberately freezes one fetched dataset and one cross-trained model,
then varies only the correction-factor label-flip fraction and correction seed.  It is
intended to answer whether the current 20% default is too aggressive for rare target
classes without changing production Dark Number behavior.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
import re

import joblib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse as scipy_sparse
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split

from JBGDarkNumberCorrectionFactor import (
    BACKEND_THREADS,
    DarkNumberCorrectionFactorEstimator,
    estimate_perturbed_same_model_correction,
)
from JBGDarkNumberValidationRunner import (
    LAST_RUN_STATE_FILENAME,
    config_from_last_run_snapshot,
    load_last_run_snapshot,
    load_validation_inputs,
    resolve_sql_credentials,
)
from JBGDarkNumbers import DarkNumberCalculator
from JBGDarkNumberSensitivityCheckpoint import (
    SensitivityCheckpoint, checkpoint_lock, file_fingerprint, runtime_identity,
    load_input_snapshot, save_input_snapshot,
)
from JBGStreamedLogger import JBGLogger


DEFAULT_FLIP_FRACTIONS = (0.05, 0.10, 0.15, 0.20)
LOW_RECOVERY_WARNING_THRESHOLD = 0.05
RECOVERY_LEVEL_UNESTIMABLE_STATUSES = {"zero_recovery", "nonfinite", "nan_recovery"}
STATISTICALLY_UNESTIMABLE_STATUSES = RECOVERY_LEVEL_UNESTIMABLE_STATUSES | {
    "insufficient_sample"
}


def _take_rows(data, indices):
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    if scipy_sparse.issparse(data):
        return data[indices]
    return np.asarray(data)[indices]


def _seed_unset_random_states(estimator, random_state: int):
    """Clone one estimator and make otherwise-unset stochastic parameters reproducible."""
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


def describe_model_pipeline(estimator) -> dict[str, Any]:
    """Return auditable model identity for the loaded/fixed sensitivity pipeline."""
    estimator_type = type(estimator)
    identity: dict[str, Any] = {
        "type": estimator_type.__name__,
        "module": estimator_type.__module__,
        "repr": repr(estimator),
    }
    raw_steps = getattr(estimator, "steps", None)
    if raw_steps:
        steps = []
        for name, step in raw_steps:
            step_type = type(step)
            steps.append(
                {
                    "name": str(name),
                    "type": step_type.__name__,
                    "module": step_type.__module__,
                    "repr": repr(step),
                }
            )
        identity["steps"] = steps
        final_estimator = raw_steps[-1][1]
    else:
        identity["steps"] = []
        final_estimator = estimator
    final_type = type(final_estimator)
    identity["final_estimator_type"] = final_type.__name__
    identity["final_estimator_module"] = final_type.__module__
    identity["final_estimator_repr"] = repr(final_estimator)
    return identity


def format_model_identity(identity: dict[str, Any]) -> str:
    """Compact one potentially multiline sklearn representation for terminal/log output."""
    steps = identity.get("steps") or []
    step_text = " -> ".join(f"{step['name']}={step['type']}" for step in steps)
    pipeline_text = f"{identity['type']} [{step_text}]" if step_text else str(identity["type"])
    compact_repr = " ".join(str(identity.get("repr", "")).split())
    final_repr = " ".join(str(identity.get("final_estimator_repr", "")).split())
    return (
        f"{pipeline_text}; final={identity.get('final_estimator_type')}; "
        f"final_repr={final_repr}; repr={compact_repr}"
    )


def _array_bytes(values: np.ndarray) -> bytes:
    array = np.ascontiguousarray(values)
    return array.view(np.uint8).tobytes()


def dataset_fingerprint(X, y) -> str:
    """Return a compact fingerprint proving that every fraction/seed used one dataset."""
    digest = hashlib.sha256()
    digest.update(str(tuple(X.shape)).encode("utf-8"))
    if scipy_sparse.issparse(X):
        matrix = X.tocsr()
        digest.update(str(matrix.dtype).encode("utf-8"))
        digest.update(_array_bytes(matrix.indptr))
        digest.update(_array_bytes(matrix.indices))
        digest.update(_array_bytes(matrix.data))
    else:
        frame = pd.DataFrame(X)
        digest.update(joblib.hash((list(frame.columns), [str(dtype) for dtype in frame.dtypes])).encode("ascii"))
        hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
        digest.update(_array_bytes(hashed))
    y_hashed = pd.util.hash_pandas_object(pd.Series(np.asarray(y)), index=True).to_numpy(dtype=np.uint64)
    digest.update(_array_bytes(y_hashed))
    return digest.hexdigest()


def split_fingerprint(train_indices: np.ndarray, test_indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(_array_bytes(np.asarray(train_indices, dtype=np.int64)))
    digest.update(_array_bytes(np.asarray(test_indices, dtype=np.int64)))
    return digest.hexdigest()


def resolve_targets(config, y, requested_targets: list[str] | None) -> list[str]:
    observed = [str(value) for value in pd.unique(np.asarray(y))]
    if requested_targets:
        selected = [str(value) for value in requested_targets]
    else:
        configured = str(config.get_dark_number_target() or "")
        selected = [configured] if configured else observed
    missing = [target for target in selected if target not in observed]
    if missing:
        raise ValueError(
            f"Requested Dark Number sensitivity target(s) are not present: {missing}. "
            f"Observed labels: {observed}."
        )
    return selected


def fit_fixed_cross_trained_model(
    source_estimator,
    X,
    y,
    *,
    test_size: float,
    split_seed: int,
):
    """Create the one fixed CV-style model used by every sensitivity cell."""
    y_array = np.asarray(y)
    indices = np.arange(y_array.shape[0])
    train_idx, test_idx = train_test_split(
        indices,
        test_size=float(test_size),
        stratify=y_array,
        random_state=int(split_seed),
    )
    model = _seed_unset_random_states(source_estimator, int(split_seed))
    model.fit(_take_rows(X, train_idx), y_array[train_idx])
    return model, train_idx, test_idx


def recovery_count_diagnostics(
    y,
    *,
    positive_class: str,
    flip_fraction: float,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    recovery_results: Iterable[float] | None,
) -> dict[str, Any]:
    """Translate fold recovery rates into hard flipped/recovered counts.

    ``DarkNumberCorrectionFactorEstimator`` uses hard ``predict`` in this study, so a
    fold recovery rate multiplied by that fold's integer flip count is an integer
    recovered count (up to floating-point representation).
    """
    y_array = np.asarray(y)
    splitter = StratifiedKFold(
        n_splits=int(n_splits),
        shuffle=True,
        random_state=int(random_state),
    )
    fold_flip_counts = []
    for _repeat_index in range(int(n_repeats)):
        for train_idx, _ in splitter.split(np.zeros(y_array.shape[0]), y_array):
            positives = int(np.sum(y_array[train_idx] == positive_class))
            fold_flip_counts.append(int(positives * float(flip_fraction)))

    flip_counts = np.asarray(fold_flip_counts, dtype=int)
    total_flipped = int(flip_counts.sum())
    mean_flipped = float(flip_counts.mean()) if len(flip_counts) else float("nan")
    if recovery_results is None:
        return {
            "correction_fits": int(len(flip_counts)),
            "total_flipped": total_flipped,
            "total_recovered": float("nan"),
            "mean_flipped_per_fit": mean_flipped,
            "pooled_recovery": float("nan"),
            "recovery_std": float("nan"),
            "recovery_min": float("nan"),
            "recovery_max": float("nan"),
        }

    rates = np.asarray(list(recovery_results), dtype=float)
    if len(rates) != len(fold_flip_counts):
        raise ValueError(
            "Recovery-result count does not match the configured correction CV tasks: "
            f"{len(rates)} results versus {len(fold_flip_counts)} tasks."
        )

    recovered_counts = np.rint(rates * flip_counts).astype(int)
    total_recovered = int(recovered_counts.sum())
    return {
        "correction_fits": int(len(rates)),
        "total_flipped": total_flipped,
        "total_recovered": total_recovered,
        "mean_flipped_per_fit": mean_flipped,
        "pooled_recovery": (total_recovered / total_flipped) if total_flipped else float("nan"),
        "recovery_std": float(rates.std(ddof=0)) if len(rates) else float("nan"),
        "recovery_min": float(rates.min()) if len(rates) else float("nan"),
        "recovery_max": float(rates.max()) if len(rates) else float("nan"),
    }


def calculate_fixed_d_cv_full(config, y, fixed_predictions, fixed_probabilities, target: str, corr: float):
    result = DarkNumberCalculator().compute_dark_numbers(
        pd.Series(np.asarray(y)),
        pd.Series(np.asarray(fixed_predictions)),
        pd.Series(np.asarray(fixed_probabilities, dtype=float)),
        type=config.get_dark_number_calculation_type(),
        corrs={target: float(corr)},
        targets=[target],
    )
    row = result.iloc[0]
    return float(row["dark_number"]), row["alphas"], str(row["type"])


def run_sensitivity_cell(
    config,
    logger,
    fixed_model,
    X_train,
    y_train,
    y_full,
    fixed_predictions,
    fixed_probabilities,
    *,
    target: str,
    flip_fraction: float,
    correction_seed: int,
    correction_n_splits: int,
    correction_n_repeats: int,
    perturbation_clones: int,
    perturbation_min_valid: int,
    perturbation_max_cv: float,
) -> dict[str, Any]:
    estimator = DarkNumberCorrectionFactorEstimator(
        estimator=clone(fixed_model),
        flip_fraction=float(flip_fraction),
        n_splits=int(correction_n_splits),
        n_repeats=int(correction_n_repeats),
        n_jobs=1,
        predict_mode="predict",
        random_state=int(correction_seed),
        positive_class=target,
        sample_size=1.0,
        parallel_backend=BACKEND_THREADS,
        logger=logger,
    )

    direct_error = ""
    try:
        estimator.fit(X_train, y_train)
    except Exception as error:  # preserve status/diagnostics from estimator where available
        direct_error = str(error)

    direct_status = str(getattr(estimator, "correction_status_", "unknown") or "unknown")
    mean_recovery = getattr(estimator, "mean_recovery_", None)
    recovery_results = getattr(estimator, "recovery_results_", None)
    direct_corr_raw = getattr(estimator, "correction_factor_", None)
    direct_valid = bool(getattr(estimator, "is_valid_for_regression_", False))
    direct_corr = float(direct_corr_raw) if direct_corr_raw is not None else float("nan")
    if not np.isfinite(direct_corr) or direct_corr <= 0:
        direct_valid = False

    counts = recovery_count_diagnostics(
        y_train,
        positive_class=target,
        flip_fraction=flip_fraction,
        n_splits=correction_n_splits,
        n_repeats=correction_n_repeats,
        random_state=correction_seed,
        recovery_results=recovery_results,
    )

    final_corr = direct_corr if direct_valid else 1.0
    final_source = "direct" if direct_valid else "fallback/unestimable"
    fallback_eligible = direct_status in RECOVERY_LEVEL_UNESTIMABLE_STATUSES
    fallback_attempted = False
    fallback_accepted = False
    fallback_valid_clones = 0
    fallback_cv = float("nan")
    fallback_reason = ""

    if (
        not direct_valid
        and fallback_eligible
        and config.should_use_experimental_perturbed_dark_number_fallback()
    ):
        fallback_attempted = True
        try:
            final_corr, final_source, details = estimate_perturbed_same_model_correction(
                fixed_model,
                X_train,
                y_train,
                flip_fraction=float(flip_fraction),
                n_splits=int(correction_n_splits),
                n_repeats=int(correction_n_repeats),
                random_state=int(correction_seed),
                positive_class=target,
                perturbation_clones=int(perturbation_clones),
                perturbation_min_valid=int(perturbation_min_valid),
                perturbation_max_cv=float(perturbation_max_cv),
                logger=logger,
            )
            fallback_accepted = final_source == "perturbed_same_model"
            fallback_valid_clones = int(details.get("valid_clone_count", 0))
            fallback_cv = float(details.get("cv", float("nan")))
            fallback_reason = str(details.get("reason", ""))
        except Exception as error:
            final_corr = 1.0
            final_source = "fallback/unestimable"
            fallback_reason = str(error)

    low_recovery_warning = bool(
        direct_valid
        and mean_recovery is not None
        and np.isfinite(mean_recovery)
        and 0 < float(mean_recovery) < LOW_RECOVERY_WARNING_THRESHOLD
    )
    d_cv_full, alphas, calculation_type = calculate_fixed_d_cv_full(
        config,
        y_full,
        fixed_predictions,
        fixed_probabilities,
        target,
        final_corr,
    )
    d_cv_full_estimable = d_cv_full if final_source != "fallback/unestimable" else float("nan")

    return {
        "target": target,
        "flip_fraction": float(flip_fraction),
        "correction_seed": int(correction_seed),
        "target_positives_train": int(np.sum(np.asarray(y_train) == target)),
        "direct_status": direct_status,
        "direct_error": direct_error,
        "direct_valid": direct_valid,
        "direct_mean_recovery": float(mean_recovery) if mean_recovery is not None else float("nan"),
        "direct_corr": direct_corr,
        "low_recovery_warning": low_recovery_warning,
        **counts,
        "fallback_eligible": fallback_eligible,
        "fallback_attempted": fallback_attempted,
        "fallback_accepted": fallback_accepted,
        "fallback_valid_clones": fallback_valid_clones,
        "fallback_cv": fallback_cv,
        "fallback_reason": fallback_reason,
        "corr": float(final_corr),
        "corr_source": final_source,
        "calculation_type": calculation_type,
        "alphas": str(alphas),
        "d_cv_full": float(d_cv_full),
        "d_cv_full_estimable": float(d_cv_full_estimable),
    }


def _finite_profile(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "cv": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=0))
    return {
        "count": int(numeric.size),
        "mean": mean,
        "median": float(numeric.median()),
        "std": std,
        "cv": (std / mean) if mean > 0 else float("nan"),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def summarize_sensitivity(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, fraction), group in runs.groupby(["target", "flip_fraction"], sort=True):
        estimable = group["corr_source"] != "fallback/unestimable"
        direct = group["corr_source"] == "direct"
        fallback = group["corr_source"] == "perturbed_same_model"
        corr_profile = _finite_profile(group.loc[estimable, "corr"])
        recovery_profile = _finite_profile(group.loc[group["direct_valid"], "direct_mean_recovery"])
        d_profile = _finite_profile(group.loc[estimable, "d_cv_full_estimable"])
        rows.append(
            {
                "target": target,
                "flip_fraction": float(fraction),
                "runs": int(len(group)),
                "mean_flipped_per_fit": float(pd.to_numeric(group["mean_flipped_per_fit"], errors="coerce").mean()),
                "mean_total_flipped": float(pd.to_numeric(group["total_flipped"], errors="coerce").mean()),
                "mean_total_recovered": float(pd.to_numeric(group["total_recovered"], errors="coerce").mean()),
                "mean_pooled_recovery": float(pd.to_numeric(group["pooled_recovery"], errors="coerce").mean()),
                "direct_statuses": str(group["direct_status"].value_counts(dropna=False).to_dict()),
                "direct_rate": float(direct.mean()),
                "fallback_activation_rate": float(group["fallback_attempted"].mean()),
                "perturbed_accept_rate": float(fallback.mean()),
                "estimable_rate": float(estimable.mean()),
                "low_recovery_warning_rate": float(group["low_recovery_warning"].mean()),
                "mean_recovery": recovery_profile["mean"],
                "recovery_std": recovery_profile["std"],
                "recovery_cv": recovery_profile["cv"],
                "corr_mean": corr_profile["mean"],
                "corr_median": corr_profile["median"],
                "corr_std": corr_profile["std"],
                "corr_cv": corr_profile["cv"],
                "corr_min": corr_profile["min"],
                "corr_max": corr_profile["max"],
                "d_cv_full_mean": d_profile["mean"],
                "d_cv_full_std": d_profile["std"],
                "d_cv_full_cv": d_profile["cv"],
            }
        )
    return pd.DataFrame(rows)


def run_noise_sensitivity(config, logger, X, y, pipeline, *, checkpoint_path: Path | None = None,
                          source_artifact_fingerprint: str | None = None, **settings):
    """Hold one process lock throughout validation and incremental checkpoint writes."""
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    with checkpoint_lock(checkpoint_path):
        return _run_noise_sensitivity(
            config, logger, X, y, pipeline, checkpoint_path=checkpoint_path,
            source_artifact_fingerprint=source_artifact_fingerprint, **settings,
        )


def _run_noise_sensitivity(
    config,
    logger,
    X,
    y,
    pipeline,
    *,
    targets: list[str] | None,
    fractions: list[float],
    runs: int,
    seed: int,
    split_seed: int,
    correction_n_splits: int,
    correction_n_repeats: int,
    perturbation_clones: int,
    perturbation_min_valid: int,
    perturbation_max_cv: float,
    checkpoint_path: Path | None,
    source_artifact_fingerprint: str | None,
):
    y_array = np.asarray(y).astype(str)
    selected_targets = resolve_targets(config, y_array, targets)
    source_model_identity = describe_model_pipeline(pipeline)
    logger.print_info(f"Sensitivity source pipeline: {format_model_identity(source_model_identity)}")
    logger.print_info(f"Resolved sensitivity targets: {selected_targets}.")
    train_idx, test_idx = train_test_split(
        np.arange(len(y_array)), test_size=float(config.get_test_size()),
        stratify=y_array, random_state=int(split_seed),
    )
    X_train = _take_rows(X, train_idx)
    y_train = y_array[train_idx]
    metadata = {
        "dataset_fingerprint": dataset_fingerprint(X, y_array),
        "split_fingerprint": split_fingerprint(train_idx, test_idx),
        "rows": int(len(y_array)),
        "features": int(X.shape[1]),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "split_seed": int(split_seed),
        "test_size": float(config.get_test_size()),
        "targets": selected_targets,
        "fractions": [float(value) for value in fractions],
        "runs_per_fraction": int(runs),
        "correction_seed_start": int(seed),
        "correction_n_splits": int(correction_n_splits),
        "correction_n_repeats": int(correction_n_repeats),
        "experimental_fallback_enabled": bool(
            config.should_use_experimental_perturbed_dark_number_fallback()
        ),
        "calculation_type": config.get_dark_number_calculation_type(),
        "source_model_identity": source_model_identity,
        "full_class_support": {str(label): int(np.sum(y_array == label)) for label in pd.unique(y_array)},
        "train_class_support": {str(label): int(np.sum(y_train == label)) for label in pd.unique(y_train)},
    }
    metadata.update(
        perturbation_clones=int(perturbation_clones),
        perturbation_min_valid=int(perturbation_min_valid),
        perturbation_max_cv=float(perturbation_max_cv),
    )
    planned_cells = [(target, float(fraction), int(seed) + run_index)
                     for target in selected_targets for fraction in fractions
                     for run_index in range(int(runs))]
    if not planned_cells or len(set(planned_cells)) != len(planned_cells):
        raise ValueError("Sensitivity grid must be non-empty and must not contain duplicate targets/fractions.")
    checkpoint = None
    if checkpoint_path is not None:
        # Compare full hyperparameters, not sklearn's truncated/address-bearing repr.
        experiment = {
            key: value for key, value in metadata.items() if key != "source_model_identity"
        }
        experiment.update(
            model_parameters_fingerprint=joblib.hash(
                _seed_unset_random_states(pipeline, split_seed).get_params(deep=True)
            ),
            source_artifact_fingerprint=source_artifact_fingerprint,
            runtime=runtime_identity(),
        )
        checkpoint = SensitivityCheckpoint(checkpoint_path, experiment, planned_cells, metadata)
        logger.print_info(f"Sensitivity checkpoint: {checkpoint_path}")
        logger.print_info(f"Checkpoint progress: {len(checkpoint.rows)}/{len(planned_cells)} completed cells.")

    bundle = checkpoint.load_fixed_experiment() if checkpoint is not None else None
    if bundle is None:
        fixed_model, train_idx, test_idx = fit_fixed_cross_trained_model(
            pipeline, X, y_array, test_size=config.get_test_size(), split_seed=split_seed,
        )
        fixed_predictions = np.asarray(fixed_model.predict(X))
        if not hasattr(fixed_model, "predict_proba"):
            raise ValueError("Noise sensitivity requires predict_proba() to mirror configured Dark Number alpha handling.")
        fixed_probabilities = np.asarray([max(row) for row in fixed_model.predict_proba(X)], dtype=float)
        fixed_model_identity = describe_model_pipeline(fixed_model)
        metadata.update(
            fixed_model_identity=fixed_model_identity,
            fixed_model_type=type(fixed_model).__name__,
            fixed_model_full_accuracy=float(np.mean(fixed_predictions == y_array)),
            fixed_model_fingerprint=joblib.hash(fixed_model),
            fixed_predictions_fingerprint=joblib.hash((fixed_predictions, fixed_probabilities)),
        )
        if checkpoint is not None:
            checkpoint.save_fixed_experiment(
                dict(model=fixed_model, predictions=fixed_predictions, probabilities=fixed_probabilities),
                metadata,
            )
    else:
        fixed_model = bundle["model"]
        fixed_predictions = bundle["predictions"]
        fixed_probabilities = bundle["probabilities"]
        metadata = dict(checkpoint.payload["metadata"])
        # The exact original fitted pipeline/predictions are restored, not refitted.
        if (joblib.hash(fixed_model) != metadata.get("fixed_model_fingerprint")
                or joblib.hash((fixed_predictions, fixed_probabilities))
                != metadata.get("fixed_predictions_fingerprint")):
            raise ValueError("Sensitivity checkpoint fixed-model/prediction fingerprint mismatch.")
        fixed_model_identity = metadata["fixed_model_identity"]
        logger.print_info("Restored original fixed sensitivity model and predictions; no refit required.")

    logger.print_info(
        "Fixed sensitivity experiment: "
        f"dataset={metadata['dataset_fingerprint'][:12]}, split={metadata['split_fingerprint'][:12]}, "
        f"rows={metadata['rows']}, train_rows={metadata['train_rows']}, "
        f"split_seed={split_seed}."
    )
    logger.print_info(f"Fixed sensitivity pipeline: {format_model_identity(fixed_model_identity)}")

    rows = list(checkpoint.rows) if checkpoint is not None else []
    total_cells = len(planned_cells)
    completed_cells = len(rows)
    completed_keys = {SensitivityCheckpoint.cell_key(row) for row in rows}
    for target in selected_targets:
        logger.print_info(
            f"Dark Number noise sensitivity target={target!r}; train positives="
            f"{int(np.sum(y_train == target))}."
        )
        for fraction in fractions:
            logger.print_info(
                f"Sensitivity fraction {fraction:.0%}: {runs} correction seeds; fixed dataset/split/model."
            )
            for run_index in range(int(runs)):
                correction_seed = int(seed) + run_index
                if (target, float(fraction), correction_seed) in completed_keys:
                    continue
                row = run_sensitivity_cell(
                    config,
                    logger,
                    fixed_model,
                    X_train,
                    y_train,
                    y_array,
                    fixed_predictions,
                    fixed_probabilities,
                    target=target,
                    flip_fraction=float(fraction),
                    correction_seed=correction_seed,
                    correction_n_splits=correction_n_splits,
                    correction_n_repeats=correction_n_repeats,
                    perturbation_clones=perturbation_clones,
                    perturbation_min_valid=perturbation_min_valid,
                    perturbation_max_cv=perturbation_max_cv,
                )
                if checkpoint is not None:
                    checkpoint.append(row)
                rows.append(row)
                completed_cells += 1
                logger.print_info(
                    f"[{completed_cells}/{total_cells}] Sensitivity result target={target}, "
                    f"fraction={fraction:.0%}, seed={correction_seed}: "
                    f"status={row['direct_status']}, recovery={row['direct_mean_recovery']:.6g}, "
                    f"corr={row['corr']:.6g}, source={row['corr_source']}, "
                    f"D_cv_full={row['d_cv_full']:.6g}."
                )

    if checkpoint is not None:
        checkpoint.finish()
        metadata.update(checkpoint_path=str(checkpoint_path), completed_cells=len(rows), total_cells=total_cells)
    detailed = pd.DataFrame(rows)
    summary = summarize_sensitivity(detailed)
    return detailed, summary, metadata


def save_outputs(config, detailed: pd.DataFrame, summary: pd.DataFrame, metadata: dict) -> tuple[Path, Path, Path]:
    output_dir = Path(config.script_path) / "output" / "csvs"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    model_name = str(config.io.model_name).replace(" ", "_")
    stem = f"dark_number_noise_sensitivity_{model_name}_{timestamp}"
    detail_path = output_dir / f"{stem}.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    metadata_path = output_dir / f"{stem}_metadata.json"
    detailed.to_csv(detail_path, sep=";", index=False)
    summary.to_csv(summary_path, sep=";", index=False)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)
    return detail_path, summary_path, metadata_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only Dark Number correction-noise sensitivity study. One dataset, split, "
            "and cross-trained model are fixed while only correction flip fraction and seed vary."
        )
    )
    parser.add_argument("--state", type=Path, default=Path.cwd() / LAST_RUN_STATE_FILENAME)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Checkpoint JSON to create or resume. Defaults to a per-model file under output/csvs. "
             "Use a new path to start a separate study; mismatches are refused.",
    )
    parser.add_argument("--sql-username", default=None)
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Target class. Defaults to the configured Dark Number target, or all observed classes if unset.",
    )
    parser.add_argument(
        "--fraction",
        action="append",
        type=float,
        default=None,
        help="Correction flip fraction. Repeat to override defaults 0.05/0.10/0.15/0.20.",
    )
    parser.add_argument("--runs", type=int, default=9, help="Correction seeds per fraction (default: 9).")
    parser.add_argument("--seed", type=int, default=42, help="First correction seed (default: 42).")
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed train/test split and model seed (default: 42).")
    parser.add_argument("--correction-splits", type=int, default=5)
    parser.add_argument("--correction-repeats", type=int, default=2)
    parser.add_argument("--perturbation-clones", type=int, default=5)
    parser.add_argument("--perturbation-min-valid", type=int, default=3)
    parser.add_argument("--perturbation-max-cv", type=float, default=0.50)
    return parser


def _validated_fractions(values: list[float] | None) -> list[float]:
    fractions = list(DEFAULT_FLIP_FRACTIONS if values is None else values)
    if not fractions:
        raise ValueError("At least one flip fraction is required.")
    for value in fractions:
        if not np.isfinite(value) or not 0.0 < float(value) < 1.0:
            raise ValueError("Every --fraction must be finite, greater than 0, and less than 1.")
    return [float(value) for value in fractions]


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)
    fractions = _validated_fractions(args.fraction)
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if args.correction_splits < 2:
        raise ValueError("--correction-splits must be at least 2.")
    if args.correction_repeats < 1:
        raise ValueError("--correction-repeats must be at least 1.")
    if args.perturbation_clones < 1:
        raise ValueError("--perturbation-clones must be at least 1.")
    if args.perturbation_min_valid < 1 or args.perturbation_min_valid > args.perturbation_clones:
        raise ValueError("--perturbation-min-valid must be between 1 and --perturbation-clones.")
    if not np.isfinite(args.perturbation_max_cv) or args.perturbation_max_cv < 0:
        raise ValueError("--perturbation-max-cv must be finite and non-negative.")

    snapshot = load_last_run_snapshot(args.state)
    sql_username, sql_password = resolve_sql_credentials(snapshot, args.sql_username)
    config = config_from_last_run_snapshot(snapshot, sql_username, sql_password)
    model_path = args.model or Path(config.get_model_filename())
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model artifact not found: {model_path}")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    logger = JBGLogger(
        quiet=False,
        in_terminal=True,
        log_filename=f"jbg-dark-number-noise-sensitivity_{timestamp}_pid{os.getpid()}.log",
    )
    logger.print_info("Dark Number correction-noise sensitivity runner (validation only; production unchanged).")
    logger.print_info(f"Validation log: {logger.get_log_filename()}")
    logger.print_info(f"Last-run state: {args.state}")
    logger.print_info(f"Saved model artifact: {model_path}")

    model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(config.io.model_name)) or "model"
    checkpoint_path = args.checkpoint or (
        Path(config.script_path) / "output" / "csvs" /
        f"dark_number_noise_sensitivity_{model_name}_checkpoint.json"
    )
    checkpoint_path = checkpoint_path.resolve()

    # Lock before loading/fetching snapshots so two invocations cannot replace the basis.
    with logger.capture_console_output(), checkpoint_lock(checkpoint_path):
        request = {
            "last_run_settings_fingerprint": joblib.hash(snapshot),
            "source_artifact_fingerprint": file_fingerprint(model_path),
            "runtime": runtime_identity(),
        }
        inputs = load_input_snapshot(checkpoint_path, request)
        if inputs is None:
            X, y, pipeline = load_validation_inputs(config, logger, model_path)
            inputs = dict(X=X, y=y, pipeline=pipeline,
                          dataset_fingerprint=dataset_fingerprint(X, np.asarray(y).astype(str)))
            save_input_snapshot(checkpoint_path, request, inputs)
            logger.print_info("Saved original sensitivity input snapshot before fixed-model training.")
        else:
            X, y, pipeline = inputs["X"], inputs["y"], inputs["pipeline"]
            if dataset_fingerprint(X, np.asarray(y).astype(str)) != inputs.get("dataset_fingerprint"):
                raise ValueError("Sensitivity input dataset fingerprint mismatch.")
            logger.print_info("Restored original sensitivity dataset snapshot; no new SQL selection/shuffle.")
        detailed, summary, metadata = _run_noise_sensitivity(
            config,
            logger,
            X,
            y,
            pipeline,
            checkpoint_path=checkpoint_path,
            source_artifact_fingerprint=request["source_artifact_fingerprint"],
            targets=args.target,
            fractions=fractions,
            runs=args.runs,
            seed=args.seed,
            split_seed=args.split_seed,
            correction_n_splits=args.correction_splits,
            correction_n_repeats=args.correction_repeats,
            perturbation_clones=args.perturbation_clones,
            perturbation_min_valid=args.perturbation_min_valid,
            perturbation_max_cv=args.perturbation_max_cv,
        )
        detail_path, summary_path, metadata_path = save_outputs(config, detailed, summary, metadata)
        logger.print_info("Dark Number correction-noise sensitivity summary")
        logger.display_matrix("Dark Number correction-noise sensitivity", summary, precision=6)
        logger.print_info(f"Detailed sensitivity CSV: {detail_path}")
        logger.print_info(f"Sensitivity summary CSV: {summary_path}")
        logger.print_info(f"Sensitivity metadata JSON: {metadata_path}")
        logger.print_info("Sensitivity study completed. Production Dark Number settings and model artifacts were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
