"""Start one worker process per stored run job (revision 285c).

The dispatcher lives in the web server process. On every poll it:

1. reaps finished worker processes it started,
2. marks running jobs whose heartbeat stopped as interrupted (their worker died),
3. returns SQL jobs awaiting credentials to the queue once their owner has
   signed in to SQL again,
4. starts workers for the oldest pending jobs, up to ``max_workers`` jobs in
   progress (including jobs still running from before a server restart).

SQL credentials go only to the worker's stdin, as one JSON line; never into
arguments, the environment, the job store or a log. Workers are detached from
the server's console and process group, so stopping the server does not stop a
running job (agreed restart behaviour); stop a job by cancelling it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from uuid import uuid4
from datetime import timedelta
from pathlib import Path
from typing import Callable

from JBGApplicationContracts import DataSourceKind, RunReference, RunStatus
from JBGRunJobStore import RunJobStore
from JBGRunWorker import EXIT_FINISHED, EXIT_NO_CREDENTIALS, EXIT_NOT_CLAIMED

SRC = Path(__file__).resolve().parent


def _detach_options() -> dict:
    if os.name == "nt":
        # Own process group: Ctrl+C in the server console does not reach it.
        # Own hidden console: closing the server window does not end it, and
        # the engine's own child processes open no visible windows.
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def worker_environment(base: dict | None = None) -> dict:
    """Environment for a worker: import path, hash seed and UTF-8 output."""
    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH")]))
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


class RunDispatcher:
    """Turns pending job records into worker processes."""

    def __init__(self, store: RunJobStore, *, store_url: str | None = None,
                 store_from_env: bool = False, store_schema: str | None = None,
                 log_dir: Path, sessions=None, artifacts: Path | None = None,
                 script_path: Path | None = None, max_workers: int = 1,
                 results_root: Path | None = None,
                 poll_seconds: float = 5.0, heartbeat_seconds: float = 30.0,
                 stale_after: timedelta = timedelta(minutes=5),
                 python: str = sys.executable,
                 launcher: Callable | None = None) -> None:
        if not isinstance(store, RunJobStore):
            raise TypeError("RunJobStore required")
        if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if stale_after <= timedelta(seconds=heartbeat_seconds * 2):
            raise ValueError("stale_after must exceed two heartbeat intervals")
        if bool(store_url) == bool(store_from_env):
            raise ValueError("give either store_url (no secrets) or store_from_env")
        self._store = store
        self._store_url = store_url
        self._store_from_env = store_from_env
        self._store_schema = store_schema
        self._log_dir = Path(log_dir)
        self._sessions = sessions
        self._artifacts = artifacts
        self._script_path = script_path
        self._results_root = results_root
        self._max_workers = max_workers
        self._poll_seconds = poll_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._stale_after = stale_after
        self._python = python
        self._launcher = launcher or subprocess.Popen
        self._processes: dict[str, tuple[object, object]] = {}   # run key -> (process, log)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._counter = 0
        self._token = uuid4().hex[:8]                  # 299g

    # --- One polling round ---------------------------------------------------------------

    def poll_once(self) -> tuple[RunReference, ...]:
        """Run one dispatch round; returns the jobs for which workers were started."""
        with self._lock:
            self._reap()
            self._store.mark_stale_interrupted(self._stale_after)
            self._requeue_signed_in()
            busy = {record.run.key for record in self._store.running()} | set(self._processes)
            capacity = self._max_workers - len(busy)
            started = []
            for record in self._store.pending(limit=max(capacity, 0) + len(self._processes) + 5):
                if capacity <= 0:
                    break
                if record.run.key in self._processes:
                    continue                      # started, not yet claimed
                credentials = None
                if record.source is DataSourceKind.SQL:
                    credentials = self._credentials(record.owner, record.spec.get("configuration", {}).get("connection"))
                    if credentials is None:
                        self._store.mark_awaiting_credentials(record.run)
                        continue
                self._launch(record.run, credentials)
                started.append(record.run)
                capacity -= 1
            return tuple(started)

    def _credentials(self, owner: str, connection=None):
        if self._sessions is None:
            return None
        return self._sessions.credentials_for_owner(owner, connection)

    def _requeue_signed_in(self) -> None:
        for record in self._store.awaiting_credentials():
            if self._credentials(record.owner, record.spec.get("configuration", {}).get("connection")) is not None:
                self._store.requeue(record.run)

    def _command(self, run: RunReference, worker_id: str) -> list[str]:
        store = (["--store-from-env"] if self._store_from_env
                 else ["--store-url", self._store_url])
        command = [self._python, "-m", "JBGRunWorker", *store,
                   "--run", run.key, "--worker-id", worker_id,
                   "--heartbeat-seconds", str(self._heartbeat_seconds),
                   "--log-dir", str(self._log_dir)]
        if self._store_schema and not self._store_from_env:
            command += ["--store-schema", self._store_schema]
        if self._artifacts is not None:
            command += ["--artifacts", str(self._artifacts)]
        if self._script_path is not None:
            command += ["--script-path", str(self._script_path)]
        if self._results_root is not None:
            command += ["--results-root", str(self._results_root)]
        return command

    def _launch(self, run: RunReference, credentials) -> None:
        self._counter += 1
        # 299g: unique per dispatcher too (a token), so a worker's exit can be
        # tied to its own job only, also when two servers share a process ID.
        worker_id = f"pid{os.getpid()}-{self._token}-{self._counter}"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self._log_dir / f"run_{run.key}.log", "ab")
        try:
            process = self._launcher(
                self._command(run, worker_id), stdin=subprocess.PIPE, stdout=log,
                stderr=subprocess.STDOUT, cwd=str(SRC), env=worker_environment(),
                **_detach_options())
        except Exception:
            log.close()
            self._store.fail_unstarted(run, "WorkerStartFailed")
            return
        line = ""
        if credentials is not None:
            line = json.dumps({"windows": credentials.windows, "username": credentials.username,
                               "password": credentials.password})
        try:
            process.stdin.write((line + "\n").encode("utf-8"))
            process.stdin.close()
        except OSError:
            pass                                   # the worker failed at start; reaped later
        self._processes[run.key] = (process, log, worker_id)

    def _reap(self) -> None:
        for key, (process, log, worker_id) in list(self._processes.items()):
            code = process.poll()
            if code is None:
                continue
            del self._processes[key]
            log.close()
            run = RunReference(key)
            # 299g: a worker of this server ended while its job is still
            # running (killed, crashed): interrupted now, not after the
            # heartbeat goes stale. A job claimed by another worker is left
            # to the heartbeat, as before.
            if code != EXIT_NOT_CLAIMED and self._store.interrupt(run, worker_id):
                continue
            if code == EXIT_NO_CREDENTIALS:
                self._store.mark_awaiting_credentials(run)
            elif code not in (EXIT_FINISHED, EXIT_NOT_CLAIMED):
                # Crashed before claiming (import error, bad arguments): do not
                # restart it forever. A claimed job is left to the heartbeat.
                if self._store.get(run).status is RunStatus.PENDING:
                    self._store.fail_unstarted(run, "WorkerStartFailed")

    # --- Background loop ---------------------------------------------------------------------

    def wake(self) -> None:
        """Poll now, e.g. right after a job was submitted."""
        self._wake.set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="jbg-run-dispatcher",
                                        daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as error:      # keep dispatching after a database hiccup
                print(f"[JBGRunDispatcher] poll failed: {type(error).__name__}", file=sys.stderr)
            self._wake.wait(self._poll_seconds)
            self._wake.clear()

    def stop(self) -> None:
        """Stop polling. Running workers continue; their logs are released here."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(10)
            self._thread = None
        with self._lock:
            for _, log, _ in self._processes.values():
                log.close()
            self._processes.clear()

    def stop_run(self, run: RunReference) -> bool:
        """299g: "Stop now": end the run's worker process with its child
        processes (a worker of this server, or one left by an earlier server
        session, found by its command line), then record the run as
        cancelled. True if the run was running and is now cancelled."""
        with self._lock:
            entry = self._processes.get(run.key)
            if entry is not None:
                _end_process(entry[0])
            else:
                for pid in _worker_pids(run.key):
                    _end_pid(pid)
            stopped = self._store.stop(run)
            self._reap()                          # releases the log file at once
        return stopped

    def active_runs(self) -> tuple[RunReference, ...]:
        with self._lock:
            return tuple(RunReference(key) for key in self._processes)


# --- 299g: ending a worker process --------------------------------------------------------

def _end_pid(pid: int) -> None:
    """End a process and its children (joblib workers), children first."""
    try:
        import psutil
        parent = psutil.Process(pid)
        family = parent.children(recursive=True) + [parent]
    except Exception:
        return                                     # already gone
    for process in family:
        try:
            process.kill()
        except Exception:
            pass
    try:
        psutil.wait_procs(family, timeout=5)
    except Exception:
        pass


def _end_process(process) -> None:
    pid = getattr(process, "pid", None)
    if isinstance(pid, int):
        _end_pid(pid)
    else:
        kill = getattr(process, "kill", None)       # e.g. a test double
        if callable(kill):
            kill()


def _worker_pids(run_key: str) -> list:
    """Worker processes of ``run_key`` on this computer (local mode)."""
    try:
        import psutil
    except ImportError:
        return []
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            line = process.info.get("cmdline") or []
        except Exception:
            continue
        if "JBGRunWorker" in line and "--run" in line:
            at = line.index("--run")
            if at + 1 < len(line) and line[at + 1] == run_key:
                found.append(process.info["pid"])
    return found
