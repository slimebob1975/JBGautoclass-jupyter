"""Best-effort, file-only resource monitoring for ordinary GridSearchCV.

Never mutate ipywidgets from this thread; never abort fitting from monitor.
"""
from contextlib import contextmanager
import os
import threading
import time

import psutil


def safe_grid_workers(requested, *, available=None, total=None, min_fraction=0.15, min_bytes=1024**3):
    """Conservatively cap new searches at one worker under strong memory pressure.

    Metrics failure is non-fatal and leaves the caller's concurrency unchanged.
    """
    requested = max(1, int(requested))
    try:
        if available is None or total is None:
            snapshot = psutil.virtual_memory()
            available, total = snapshot.available, snapshot.total
        if total <= 0 or available < 0:
            return requested, None
        if requested > 1 and (available / total < min_fraction or available < min_bytes):
            return 1, (f"GridSearchCV memory guard: free RAM {available / 2**20:.0f} MiB "
                       f"({available / total:.1%}); reducing workers {requested} -> 1 "
                       "before search starts")
    except (OSError, RuntimeError, ValueError, AttributeError):
        return requested, None
    return requested, None


def _memory_snapshot():
    """Best-effort process-tree RSS; shared pages may be counted more than once."""
    mem = psutil.virtual_memory()
    root = psutil.Process(os.getpid())
    rss = 0
    counted = 0
    for process in [root, *root.children(recursive=True)]:
        try:
            rss += process.memory_info().rss
            counted += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return (mem.available / 2**20, mem.total, rss / 2**20, counted)


@contextmanager
def training_heartbeat(log_file, operation, interval=30.0):
    """Record liveness even when GUI outputs are unavailable; never affect fit."""
    done = threading.Event()
    started = time.monotonic()

    def emit(state):
        try:
            free_mib, total, tree_rss_mib, processes = _memory_snapshot()
            percent = (free_mib * 2**20 / total) if total > 0 else 0
            log_file.write_message(
                f"Training {state}: {operation}; elapsed={time.monotonic()-started:.0f}s; "
                f"system_available={free_mib:.0f} MiB ({percent:.1%}); "
                f"process_tree_rss={tree_rss_mib:.0f} MiB; processes={processes}",
                level="ERROR" if state == "failed" else "INFO",
            )
        except Exception:
            # This observer must not introduce new training failures.
            pass

    def watch():
        while not done.wait(interval):
            emit("running")

    emit("started")
    thread = threading.Thread(target=watch, name="jbg-grid-heartbeat", daemon=True)
    try:
        thread.start()
        try:
            yield
        except BaseException:
            emit("failed")
            raise
        else:
            emit("finished")
    finally:
        done.set()
        if thread.is_alive():
            thread.join(timeout=1.0)
