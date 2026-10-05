"""Bounded execution and parent-process progress for experimental correction fits."""
from pickle import PicklingError
from time import perf_counter
from inspect import signature
import re

from joblib import Parallel, cpu_count, delayed, parallel_backend, __version__ as joblib_version
from joblib.externals.loky.process_executor import BrokenProcessPool, TerminatedWorkerError
from threadpoolctl import threadpool_limits


MAX_FALLBACK_WORKERS = 8
_joblib_release = tuple(int(part) for part in re.match(r"(\d+)\.(\d+)", joblib_version).groups())
_STREAM_MODE = ("generator_unordered" if _joblib_release >= (1, 4) else "generator")
_HAS_STREAMING = "return_as" in signature(Parallel).parameters


def resolve_fallback_workers(estimator, requested, fit_count):
    """Respect CPU/global caps; keep framework/checkpoint estimators sequential."""
    requested = cpu_count() if requested is None or requested < 0 else max(1, requested)
    workers = max(1, min(requested, cpu_count(), MAX_FALLBACK_WORKERS, fit_count))
    if not _HAS_STREAMING:
        return 1, "installed joblib has no streaming results; sequential fit progress"
    params = estimator.get_params(deep=True)
    components = [estimator, *params.values()]
    for component in components:
        module = type(component).__module__.lower()
        name = type(component).__name__.lower()
        if getattr(component, "CHECKPOINT_DIR", None) or any(part in module or part in name for part in
               ("torch", "skorch", "keras", "tensorflow", "jax", "flax")):
            return 1, "framework/checkpoint estimator kept sequential"
    return workers, "bounded independent fit processes"


def _evaluate_indexed(index, task):
    function, args, kwargs = task
    return index, function(*args, **kwargs)


def run_isolated_tasks(tasks, n_jobs, progress=None, logger=None, workers_changed=None):
    """Stream completions, preserve submission order and retry only missing results.

    Callbacks stay in the parent process. No logger/widget is sent to workers.
    Independent process fits use one native thread and large arrays may be memmapped.
    Pickling/backend failures use a sequential fallback instead of shared fit threads.
    """
    completed = {}
    workers = max(1, n_jobs) if _HAS_STREAMING else 1
    if workers_changed is not None:
        workers_changed(workers)

    def collect(iterator):
        for index, value in iterator:
            completed[index] = value
            if progress is not None:
                progress(len(completed), len(tasks))

    while len(completed) < len(tasks):
        pending = [index for index in range(len(tasks)) if index not in completed]
        try:
            if workers == 1:
                with threadpool_limits(limits=1), parallel_backend("loky", n_jobs=1,
                                                                 inner_max_num_threads=1):
                    collect(_evaluate_indexed(index, tasks[index]) for index in pending)
            else:
                with parallel_backend("loky", inner_max_num_threads=1):
                    with Parallel(n_jobs=workers, return_as=_STREAM_MODE,
                                  batch_size=1, pre_dispatch="n_jobs", max_nbytes="1M") as pool:
                        collect(pool(delayed(_evaluate_indexed)(index, tasks[index])
                                     for index in pending))
        except (MemoryError, SystemError, PicklingError, BrokenProcessPool,
                TerminatedWorkerError) as error:
            if workers == 1:
                raise
            workers = (1 if isinstance(error, PicklingError) or
                       isinstance(error, BrokenProcessPool) and not isinstance(error, TerminatedWorkerError)
                       else max(1, workers // 2))
            if workers_changed is not None:
                workers_changed(workers)
            if logger is not None:
                logger.print_warning(
                    f"Experimental correction fit execution hit {type(error).__name__}; "
                    f"retrying unfinished fits with {workers} worker(s). "
                    f"Retained {len(completed)}/{len(tasks)} completed results."
                )
    return [completed[index] for index in range(len(tasks))], workers


class FallbackProgress:
    """Account for completed and skipped fit slots, without inventing elapsed-time ETA."""
    def __init__(self, logger, clones, fits_per_clone, workers, reason):
        self.logger = logger
        self.clones = clones
        self.fits_per_clone = fits_per_clone
        self.total = clones * fits_per_clone
        self.workers = workers
        self.reason = reason
        self.completed = 0
        self.skipped = 0
        self.clone_completed = 0
        self.clone_index = 0
        self.offset = 0
        self.key = "dark_number_shadow_fits"
        self.has_bar = logger is not None and all(callable(getattr(logger, method, None))
            for method in ("start_inline_progress", "update_inline_progress", "end_inline_progress"))

    def __enter__(self):
        self.started = perf_counter()
        if self.logger is not None:
            self.logger.print_info(
                f"EXPERIMENTAL fallback work: {self.clones} shadow clones x "
                f"{self.fits_per_clone} fold/repeat fits = {self.total} fits; "
                f"up to {self.workers} concurrent fit(s), one native thread per fit "
                f"({self.reason}). Progress counts completed or skipped fit slots, not time."
            )
        if self.has_bar:
            self.logger.start_inline_progress(self.key, "Shadow fits", self.total,
                "Percent fit slots accounted for (completed or skipped after clone failure)")
        return self

    def start_clone(self, index):
        self.clone_index = index
        self.clone_completed = 0
        self.offset = self.completed + self.skipped

    def fit_completed(self, completed, total):
        self.completed += completed - self.clone_completed
        self.clone_completed = completed
        self._update(self.offset + completed)

    def finish_clone(self, seconds, status, workers):
        self.skipped += self.fits_per_clone - self.clone_completed
        self._update(self.completed + self.skipped)
        if self.logger is not None:
            self.logger.print_info(
                f"EXPERIMENTAL shadow clone {self.clone_index + 1}/{self.clones}: "
                f"{status}; {self.clone_completed}/{self.fits_per_clone} fits returned; "
                f"{seconds:.2f} s; workers={workers}; overall completed={self.completed}, "
                f"skipped={self.skipped}, planned={self.total}."
            )

    def _update(self, accounted):
        if self.has_bar:
            self.logger.update_inline_progress(self.key, accounted,
                f"Shadow fits accounted for (completed {self.completed}, skipped {self.skipped})")

    def __exit__(self, exc_type, exc, tb):
        self.seconds = perf_counter() - self.started
        if self.has_bar:
            self.logger.end_inline_progress(self.key,
                set_100=exc_type is None and self.completed + self.skipped == self.total)
        if self.logger is not None:
            self.logger.print_info(
                f"EXPERIMENTAL fallback {'finished' if exc_type is None else 'interrupted'} "
                f"after {self.seconds:.2f} s: completed={self.completed}, "
                f"skipped={self.skipped}, planned={self.total} fits."
            )
