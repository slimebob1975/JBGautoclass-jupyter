"""Run the existing cross-validation invocation without owning result diagnostics.

ModelHandler remains responsible for fold selection, target scoring, CV timing,
reporting and all post-CV state updates.  The context keeps the historical
execute_n_job behavior (including job caps and monkeypatch hooks).
"""
from __future__ import annotations

import time
from typing import Any

from sklearn.model_selection import cross_validate
from JBGNativeCatBoost import prepare_model_input
from JBGEstimatorInput import call_with_sparse_input_retry
from JBGNystroemRuntime import prepare_runtime_cv


def execute_cross_validation(
    context: Any, pipeline: Any, dh: Any, kfold: Any, algorithm: Any,
    diagnostic_scorer: Any, fit_params: dict,
) -> tuple[Any, bool, float]:
    """Return CV results, whether Nystroem profiling ran, and wall time."""
    # We want to execute the job with as many workers as possible. Dense data keeps
    # the historical NumPy path, while pandas sparse text data is converted to SciPy
    # CSR so sklearn can preserve sparsity through cross-validation.
    # CV has one task per fold. execute_n_job applies CPU and positive
    # configured caps; a global -1 must not erase this finite task bound.
    n_jobs_desired = len(kfold) if isinstance(kfold, list) else kfold.get_n_splits()
    estimator_X = prepare_model_input(dh.X_train, pipeline, prefer_numpy=True)
    context._last_execution_n_jobs = None
    cv_pipeline, cv_scorer, profiled = pipeline, diagnostic_scorer, False
    try:
        cv_pipeline, cv_scorer, profiled = prepare_runtime_cv(
            pipeline, diagnostic_scorer, getattr(algorithm, 'name', ''),
        )
    except Exception as profile_error:
        context.handler.logger.print_warning(
            f"Nystroem runtime diagnostics could not be prepared: {profile_error}. "
            "Continuing with the original CV call."
        )
    started = time.perf_counter()
    cv_results = call_with_sparse_input_retry(
        lambda X_input: context.execute_n_job(
            cross_validate, cv_pipeline, X_input, dh.Y_train.to_numpy(), cv=kfold,
            scoring=cv_scorer, n_jobs_desired=n_jobs_desired,
            params=fit_params, error_score='raise',
        ),
        estimator_X, logger=context.handler.logger, context="cross-validation", estimator=pipeline,
    )
    wall_seconds = time.perf_counter() - started
    return cv_results, profiled, wall_seconds
