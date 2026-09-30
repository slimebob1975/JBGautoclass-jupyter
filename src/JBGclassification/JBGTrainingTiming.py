"""Transient CV measurements and approximate final-search wall-clock budgets."""
from dataclasses import dataclass
from math import ceil, isfinite
from statistics import mean


@dataclass(frozen=True)
class CVTiming:
    folds: int
    workers: int
    mean_fit_seconds: float
    mean_score_seconds: float
    wall_seconds: float

    @classmethod
    def from_results(cls, results, folds, workers, wall_seconds):
        """Missing/invalid telemetry must never prevent otherwise successful training."""
        try:
            fits = [float(value) for value in results["fit_time"]]
            scores = [float(value) for value in results["score_time"]]
            if (folds < 2 or workers < 1 or len(fits) != folds or len(scores) != folds
                    or not isfinite(wall_seconds) or wall_seconds < 0
                    or any(not isfinite(value) or value < 0 for value in fits + scores)
                    or mean(fits) <= 0):
                return None
            return cls(folds, min(workers, folds), mean(fits), mean(scores), wall_seconds)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None


@dataclass(frozen=True)
class GridSearchEstimate:
    seconds: float
    workers: int
    cv_fits: int
    refit_seconds: float
    overhead_seconds: float


def estimate_grid_search(timing, combinations, folds, requested_workers):
    """Use measured fold work, concurrent batches and one sequential refit.

    Do not assume speedup beyond the concurrency actually measured during CV.
    Refit scales the mean fold fit by the approximate training-row ratio k/(k-1).
    Both that scaling and equal costs across parameter combinations are assumptions.
    """
    if timing is None or folds != timing.folds or combinations < 1 or requested_workers < 1:
        return None
    values = (timing.mean_fit_seconds, timing.mean_score_seconds, timing.wall_seconds)
    if (timing.folds < 2 or timing.workers < 1 or timing.mean_fit_seconds <= 0
            or any(not isfinite(value) or value < 0 for value in values)):
        return None
    cv_fits = combinations * folds
    workers = min(requested_workers, timing.workers, folds, cv_fits)
    fold_seconds = timing.mean_fit_seconds + timing.mean_score_seconds
    observed_batches = ceil(folds / min(timing.workers, folds))
    overhead = max(0.0, timing.wall_seconds - observed_batches * fold_seconds)
    refit = timing.mean_fit_seconds * folds / (folds - 1)
    seconds = ceil(cv_fits / workers) * fold_seconds + refit + overhead
    if not isfinite(seconds):
        return None
    return GridSearchEstimate(seconds, workers, cv_fits, refit, overhead)


def format_approximate_duration(seconds):
    """Round upwards to useful display precision, retaining sub-minute searches."""
    if seconds < 1:
        return "~<1 s"
    if seconds < 60:
        return f"~{ceil(seconds)} s"
    minutes = ceil(seconds / 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"~{hours} h {minutes} min"
    return f"~{minutes} min"
