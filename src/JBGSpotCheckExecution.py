"""Spot-check candidate execution extracted without changing the ModelHandler contract.

The owner parameter is the existing ModelHandler.  Preflight, RFE search, CV,
result rows, diagnostics and failure handling retain their original ordering.
"""
from __future__ import annotations

import time
from math import ceil

import numpy as np

from JBGMeta import Reduction
from JBGExceptions import ModelException


def evaluate_spot_check_candidate(owner, dh, preprocessor, preprocessor_callable, reduction,
                                  reduction_callable, algorithm, algorithm_callable,
                                  oversampler, undersampler, kfold, state):
    skip_reason = owner.get_preflight_skip_reason(
        preprocessor, preprocessor_callable, reduction, algorithm, dh.X_train,
        estimator=algorithm_callable, Y_train=dh.Y_train
    )
    if skip_reason:
        failure = f"SKIPPED: {skip_reason}"
        return [owner._build_spot_check_result(
            preprocessor=preprocessor,
            reduction=reduction,
            algorithm=algorithm,
            components=dh.X_train.shape[1],
            cv_score=np.nan,
            cv_stdev=np.nan,
            elapsed_time=0.0,
            failure=failure,
            oversampler=oversampler, undersampler=undersampler,
        )], False

    max_features_selection = dh.X.shape[1]
    min_features_selection = 0 if reduction == Reduction.RFE else max_features_selection
    total_features = max_features_selection
    rfe_round = 0
    rfe_round_limit = owner._get_rfe_search_round_limit(total_features)
    num_components = max_features_selection
    rfe_score = 0.0
    num_features = max_features_selection
    first_feature_selection = True
    candidate_results = []
    candidate_success = False

    while first_feature_selection or min_features_selection < max_features_selection:
        if not first_feature_selection:
            num_features = ceil((min_features_selection + max_features_selection) / 2)
            if num_features == max_features_selection:
                break
        else:
            first_feature_selection = False
            num_features = max_features_selection

        if reduction == Reduction.RFE and rfe_round >= rfe_round_limit:
            owner.handler.logger.print_warning(
                f"Stopping RFE search {preprocessor.name}-{reduction.name}-{algorithm.name} "
                f"after {rfe_round} rounds; limit is {rfe_round_limit}."
            )
            break

        bounds_before = (min_features_selection, max_features_selection)
        if reduction == Reduction.RFE:
            rfe_round += 1
            owner.handler.logger.print_progress(
                message=(
                    f"RFE search {preprocessor.name}-{reduction.name}-{algorithm.name}: "
                    f"round {rfe_round}/{rfe_round_limit}, target features "
                    f"{num_features}/{total_features}, search interval "
                    f"[{min_features_selection}, {max_features_selection}]."
                )
            )

        t0 = time.time()

        try:
            current_pipeline, cv_results, failure = owner.create_pipeline_and_cv(
                reduction, algorithm, preprocessor, reduction_callable, algorithm_callable,
                preprocessor_callable, oversampler, undersampler, kfold, dh, num_features
            )

            # Spot-checking is CV-only. Do not fit candidates against or score the
            # final holdout here; it remains untouched until final evaluation.

            if current_pipeline is not None:
                num_components = owner.get_components_from_pipeline(
                    reduction, current_pipeline, num_features
                )
            else:
                num_components = num_features
        except ModelException as ex:
            owner.handler.logger.print_warning(f"ModelException: {str(ex)}")
            break
        except Exception as ex:
            owner.handler.logger.print_warning(f"Exception: {str(ex)}")

        elapsed_time = time.time() - t0
        temp_cv_score, temp_cv_stdev = owner._summarize_cv_results(cv_results)
        rfe_score, max_features_selection, min_features_selection = owner.calculate_current_features(
            temp_cv_score, rfe_score, num_features, max_features_selection, min_features_selection
        )
        rfe_search_stalled = (
            reduction == Reduction.RFE
            and rfe_round > 1
            and bounds_before == (min_features_selection, max_features_selection)
        )

        failure, candidate_is_stable = owner._consider_spot_check_candidate(
            state=state,
            pipeline=current_pipeline,
            preprocessor=preprocessor,
            reduction=reduction,
            algorithm=algorithm,
            cv_score=temp_cv_score,
            cv_stdev=temp_cv_stdev,
            num_features=num_features,
            num_components=num_components,
            failure=failure,
        )
        if state.trained_pipeline is current_pipeline and current_pipeline is not None:
            state.best_oversampler = oversampler
            state.best_undersampler = undersampler
        if candidate_is_stable:
            candidate_success = True
            if getattr(owner.handler.config.mode, "model_benchmark_150", False):
                owner._benchmark_candidate_150(current_pipeline, dh, kfold,
                    algorithm, preprocessor, reduction, oversampler, undersampler)

        candidate_results.append(owner._build_spot_check_result(
            preprocessor=preprocessor,
            reduction=reduction,
            algorithm=algorithm,
            components=min(num_components, num_features),
            cv_score=temp_cv_score,
            cv_stdev=temp_cv_stdev,
            elapsed_time=elapsed_time,
            failure=failure,
            oversampler=oversampler, undersampler=undersampler,
            diagnostics=(owner._last_cv_diagnostics[1]
                         if getattr(owner, "_last_cv_diagnostics", None) is not None
                         and owner._last_cv_diagnostics[0] is current_pipeline else None),
        ))

        if rfe_search_stalled:
            owner.handler.logger.print_warning(
                f"Stopping RFE search {preprocessor.name}-{reduction.name}-{algorithm.name}: "
                f"search interval did not shrink from "
                f"[{bounds_before[0]}, {bounds_before[1]}]."
            )
            break

    return candidate_results, candidate_success
