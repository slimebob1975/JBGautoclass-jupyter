"""Transient CV measurements and approximate final-search wall-clock budgets."""
from dataclasses import dataclass, replace
from math import ceil, floor, isfinite, log2
from statistics import mean, median
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import NamedTemporaryFile
import time

import joblib
import numpy as np
from sklearn.base import clone


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
    base_seconds: float = None
    calibration_samples: int = 0
    calibration_factor: float = 1.0
    calibration_min: float = 1.0
    calibration_max: float = 1.0


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


def grid_search_timing_key(model, search_params, scorer, X, y, folds, workers, source,
                           cv_workers=None, cv_timing=None):
    """Match comparable work, never persist data, credentials or fitted state.

    This is a workload/context signature, not a fingerprint of dataset contents.
    Unsupported custom objects simply disable optional historical calibration.
    """
    try:
        if not source or not source[2]:
            return None
        labels, counts = np.unique(np.asarray(y).reshape(-1), return_counts=True)
        if sum(counts) != X.shape[0] or len(X.shape) != 2:
            return None
        timing_regime = None
        if cv_timing is not None:
            base = estimate_grid_search(cv_timing, 1, folds, workers)
            if base is None:
                return None
            # Do not carry a ratio dominated by cold CV startup into a warm CV
            # baseline (or one with materially different per-fold cost). Coarse
            # factor-of-two bands favor a safe miss over unrelated calibration.
            timing_regime = (
                floor(log2(max(0.001, cv_timing.mean_fit_seconds + cv_timing.mean_score_seconds))),
                floor(log2(max(0.001, base.overhead_seconds))),
            )
        versions = tuple((name, getattr(sys.modules.get(name), "__version__", None))
                         for name in ("numpy", "scipy", "sklearn", "joblib", "torch", "skorch", "tensorflow"))
        context = (
            "grid-search-timing-118", source, tuple(X.shape),
            type(X).__module__, type(X).__qualname__,
            tuple(str(column) for column in getattr(X, "columns", ())),
            tuple(str(dtype) for dtype in getattr(X, "dtypes", ())),
            str(getattr(X, "dtype", "")), labels, counts,
            clone(model), search_params, scorer, folds, workers, cv_workers, timing_regime,
            platform.node(), platform.machine(), platform.platform(), os.cpu_count(),
            sys.version_info[:3], versions,
            tuple((name, os.environ.get(name)) for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "JOBLIB_START_METHOD", "LOKY_MAX_CPU_COUNT",
            )),
        )
        return joblib.hash(context, hash_name="sha1")
    except Exception:
        return None


class GridSearchTimingHistory:
    """Small, optional JSON history; atomic replacement, no executable payloads."""
    VERSION = 1
    MAX_PROFILES = 64
    MAX_SAMPLES = 5
    MAX_AGE_SECONDS = 30 * 24 * 60 * 60
    MAX_BYTES = 1024 * 1024

    def __init__(self, path):
        self.path = Path(path)

    @staticmethod
    def _valid_key(key):
        return isinstance(key, str) and len(key) == 40 and all(c in "0123456789abcdef" for c in key)

    def _read(self, now):
        if not self.path.exists():
            return {}
        if self.path.stat().st_size > self.MAX_BYTES:
            raise ValueError("timing history exceeds its size limit")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != self.VERSION:
            raise ValueError("unsupported timing history format")
        profiles = payload.get("profiles")
        if not isinstance(profiles, dict):
            raise ValueError("invalid timing history profiles")
        clean = {}
        for key, samples in profiles.items():
            if not self._valid_key(key) or not isinstance(samples, list):
                continue
            valid = []
            for sample in samples:
                try:
                    timestamp, base, actual = (float(sample[k]) for k in ("timestamp", "base_seconds", "actual_seconds"))
                    if (all(isfinite(v) for v in (timestamp, base, actual))
                            and now - self.MAX_AGE_SECONDS <= timestamp <= now
                            and base > 0 and actual > 0 and isfinite(actual / base)):
                        valid.append(dict(timestamp=timestamp, base_seconds=base, actual_seconds=actual))
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
            if valid:
                clean[key] = sorted(valid, key=lambda row: row["timestamp"])[-self.MAX_SAMPLES:]
        newest = sorted(clean, key=lambda key: clean[key][-1]["timestamp"])[-self.MAX_PROFILES:]
        return {key: clean[key] for key in newest}

    def observations(self, key):
        if not self._valid_key(key):
            return []
        return self._read(time.time()).get(key, [])

    def record(self, key, estimate, actual_seconds):
        if not self._valid_key(key) or estimate is None:
            return False
        base = estimate.base_seconds if estimate.base_seconds is not None else estimate.seconds
        if not all(isfinite(v) and v > 0 for v in (base, actual_seconds, actual_seconds / base if base > 0 else 0)):
            return False
        now = time.time()
        profiles = self._read(now)
        profiles[key] = (profiles.get(key, []) + [dict(
            timestamp=now, base_seconds=base, actual_seconds=actual_seconds,
        )])[-self.MAX_SAMPLES:]
        newest = sorted(profiles, key=lambda k: profiles[k][-1]["timestamp"])[-self.MAX_PROFILES:]
        payload = {"version": self.VERSION, "profiles": {k: profiles[k] for k in newest}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                    prefix="grid-timing-", suffix=".tmp", delete=False) as file:
                temporary_path = Path(file.name)
                json.dump(payload, file, allow_nan=False)
                file.write("\n")
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return True


def calibrate_grid_search_estimate(estimate, observations):
    """Median actual/base ratios from matching successful searches only.

    Always normalize against the uncalibrated estimate, avoiding feedback that
    would train the next correction against an already corrected forecast.
    """
    if estimate is None:
        return None
    base = estimate.base_seconds if estimate.base_seconds is not None else estimate.seconds
    ratios = []
    for observation in observations:
        try:
            previous_base = float(observation["base_seconds"])
            actual = float(observation["actual_seconds"])
            if not all(isfinite(v) and v > 0 for v in (previous_base, actual)):
                continue
            ratio = actual / previous_base
            if isfinite(ratio) and ratio > 0:
                ratios.append(ratio)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    if not ratios:
        return estimate
    factor = median(ratios)
    seconds = base * factor
    if not isfinite(seconds) or seconds <= 0:
        return estimate
    return replace(estimate, seconds=seconds, base_seconds=base,
                   calibration_samples=len(ratios), calibration_factor=factor,
                   calibration_min=min(ratios), calibration_max=max(ratios))


def format_grid_search_timing_diagnostics(search, actual_seconds):
    """Describe measured parameter/refit costs without inferring startup causes."""
    try:
        fits = [float(v) for v in search.cv_results_["mean_fit_time"]]
        scores = [float(v) for v in search.cv_results_["mean_score_time"]]
        refit = float(search.refit_time_)
        if (not fits or len(fits) != len(scores)
                or any(not isfinite(v) or v < 0 for v in fits + scores + [refit, actual_seconds])
                or refit > actual_seconds):
            return None
        return (
            f"Grid search measured timing: candidate mean fit {min(fits):.3g}–{max(fits):.3g} s/fold "
            f"(mean {mean(fits):.3g} s); scoring mean {mean(scores):.3g} s/fold; "
            f"refit {refit:.3g} s; CV/dispatch phase {actual_seconds - refit:.3g} s. "
            "CV/dispatch includes worker startup, scheduling and contention; these are not separately measured."
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None


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


def format_actual_duration(seconds):
    """Human-readable elapsed duration without calling an actual measurement approximate."""
    if seconds < 60:
        return f"{seconds:.2f} s"
    whole_seconds = int(round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes} min {seconds} s"
    return f"{minutes} min {seconds} s"


def format_grid_search_comparison(estimate, actual_seconds):
    """Compare successful search-fit + refit durations using unrounded seconds."""
    if estimate is None or not isfinite(estimate.seconds) or estimate.seconds <= 0:
        return None
    if not isfinite(actual_seconds) or actual_seconds < 0:
        return None
    ratio = 100 * actual_seconds / estimate.seconds
    deviation = ratio - 100
    return (
        f"Grid search completed (CV fits + refit): estimated "
        f"{format_approximate_duration(estimate.seconds)}; actual {format_actual_duration(actual_seconds)}; "
        f"actual/estimate {ratio:.1f}%; deviation {deviation:+.1f}% "
        "(percentages use unrounded durations; negative = shorter, positive = longer)."
    )
