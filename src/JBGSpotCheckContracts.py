"""Static contracts for spot-check outputs and mutable best-candidate state.

These are typing-only declarations. The runtime representation remains the
original lists, tuple pairs and ModelHandler-owned _SpotCheckState instance.
"""
from __future__ import annotations

from typing import Any, Protocol, TypeAlias

SpotCheckResultRow: TypeAlias = list[Any]
SpotCheckExecutionResult: TypeAlias = tuple[list[SpotCheckResultRow], bool]
SpotCheckSelectionResult: TypeAlias = tuple[str, bool]


class SpotCheckBestState(Protocol):
    """Fields shared between candidate selection and its caller."""

    best_cv_score: float
    best_stdev: float
    trained_pipeline: Any
    best_algorithm: Any
    best_preprocessor: Any
    best_reduction: Any
    best_rfe_feature_selection: int
    best_num_components: int
    best_cv_timing: Any
    best_oversampler: Any
    best_undersampler: Any


class SpotCheckLogger(Protocol):
    """Logging operations required by candidate execution."""

    def print_warning(self, message: str) -> Any: ...
    def print_progress(self, message: str) -> Any: ...


class SpotCheckMode(Protocol):
    model_benchmark_150: bool


class SpotCheckConfig(Protocol):
    mode: SpotCheckMode


class SpotCheckHandlerContext(Protocol):
    logger: SpotCheckLogger
    config: SpotCheckConfig


class SpotCheckExecutionOwner(Protocol):
    """Structural port used by spot-check execution; no ModelHandler import."""

    handler: SpotCheckHandlerContext
    _last_cv_diagnostics: Any

    def get_preflight_skip_reason(self, *args: Any, **kwargs: Any) -> Any: ...
    def _build_spot_check_result(self, **kwargs: Any) -> SpotCheckResultRow: ...
    def _get_rfe_search_round_limit(self, n_features: int) -> int: ...
    def create_pipeline_and_cv(self, *args: Any, **kwargs: Any) -> Any: ...
    def get_components_from_pipeline(self, *args: Any, **kwargs: Any) -> Any: ...
    def _summarize_cv_results(self, cv_results: Any) -> Any: ...
    def calculate_current_features(self, *args: Any, **kwargs: Any) -> Any: ...
    def _consider_spot_check_candidate(self, **kwargs: Any) -> SpotCheckSelectionResult: ...
    def _benchmark_candidate_150(self, *args: Any, **kwargs: Any) -> Any: ...
