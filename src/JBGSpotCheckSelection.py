"""Select a successful spot-check candidate without owning training or CV execution.

Keep candidate acceptance, ties, and best-state updates identical to the legacy
ModelHandler path. The callback preserves timing and other handler-specific state.
"""
from __future__ import annotations

from typing import Any, Callable

from JBGSpotCheckContracts import SpotCheckBestState, SpotCheckSelectionResult

import numpy as np


def consider_spot_check_candidate(
    *, state: SpotCheckBestState, pipeline: Any, preprocessor: Any, reduction: Any,
    algorithm: Any, cv_score: float, cv_stdev: float, num_features: int,
    num_components: int, failure: str,
    is_best_run_yet: Callable[[float, float, float, float], bool],
    on_best_candidate: Callable[..., None],
) -> SpotCheckSelectionResult:
    """Return (failure, is_stable) and update the existing best state if better."""
    candidate_success = (
        pipeline is not None
        and not failure
        and np.isfinite(cv_score)
        and np.isfinite(cv_stdev)
    )
    if not candidate_success:
        return failure, False

    if is_best_run_yet(cv_score, cv_stdev, state.best_cv_score, state.best_stdev):
        on_best_candidate(
            state=state,
            pipeline=pipeline,
            preprocessor=preprocessor,
            reduction=reduction,
            algorithm=algorithm,
            cv_score=cv_score,
            cv_stdev=cv_stdev,
            num_features=num_features,
            num_components=num_components,
        )

    return failure, True
