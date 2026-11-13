"""299g (asked 2026-11-13): stopping a run that does not stop.

"Cancel the run" is a request the worker honours at its next checkpoint;
every inner step (an algorithm combination, rows fetched) is now one too.
After cancelling, "Stop now" ends the worker process at once. A worker of
this server that ends while its job still runs is marked interrupted at
once, instead of when its heartbeat goes stale (5 minutes).
"""
import re
import subprocess
import sys
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlalchemy")

from JBGApplicationContracts import DataSourceKind, RunReference, RunStatus  # noqa: E402
from JBGRunDispatcher import RunDispatcher, _worker_pids  # noqa: E402
from JBGRunJobStore import RunJobStore  # noqa: E402
from test_revision_285c_dispatcher import FakeLauncher, FakeProcess  # noqa: E402

SPEC = {"spec_version": 1}


@pytest.fixture
def store(tmp_path):
    store = RunJobStore.sqlite(tmp_path / "jobs.sqlite")
    store.create_tables()
    yield store
    store.close()


def _dispatcher(store, tmp_path, launcher):
    return RunDispatcher(store, store_url=f"sqlite:///{tmp_path / 'jobs.sqlite'}",
                         log_dir=tmp_path / "logs", launcher=launcher, heartbeat_seconds=1,
                         stale_after=timedelta(minutes=5))


def _claimed_by_its_worker(store, launcher):
    command = launcher.processes[0].command
    run = RunReference(command[command.index("--run") + 1])
    store.claim(run, command[command.index("--worker-id") + 1])
    return run


class KillableProcess(FakeProcess):
    def kill(self):
        self.code = -9


class KillableLauncher(FakeLauncher):
    def __call__(self, command, **options):
        process = KillableProcess(command, **options)
        self.processes.append(process)
        return process


# --- The store ---------------------------------------------------------------------------

def test_only_the_jobs_own_worker_can_interrupt_it(store):
    run = store.create("owner", DataSourceKind.CSV, SPEC)
    store.claim(run, "w1")
    assert store.interrupt(run, "w2") is False and store.get(run).status is RunStatus.RUNNING
    assert store.interrupt(run, "w1") is True
    record = store.get(run)
    assert record.status is RunStatus.INTERRUPTED and record.finished_at is not None


def test_stopping_cancels_a_running_job_only(store):
    run = store.create("owner", DataSourceKind.CSV, SPEC)
    assert store.stop(run) is False                               # still pending
    store.claim(run, "w")
    assert store.stop(run) is True
    record = store.get(run)
    assert record.status is RunStatus.CANCELLED and record.cancel_requested


# --- The dispatcher ----------------------------------------------------------------------

def test_a_killed_worker_of_this_server_is_interrupted_at_once(store, tmp_path):
    launcher = FakeLauncher()
    dispatcher = _dispatcher(store, tmp_path, launcher)
    store.create("owner", DataSourceKind.CSV, SPEC)
    dispatcher.poll_once()
    run = _claimed_by_its_worker(store, launcher)
    launcher.processes[0].code = 1                                # killed or crashed
    dispatcher.poll_once()
    assert store.get(run).status is RunStatus.INTERRUPTED
    assert dispatcher.active_runs() == ()
    dispatcher.stop()


def test_a_worker_that_finished_its_job_changes_nothing(store, tmp_path):
    launcher = FakeLauncher()
    dispatcher = _dispatcher(store, tmp_path, launcher)
    store.create("owner", DataSourceKind.CSV, SPEC)
    dispatcher.poll_once()
    run = _claimed_by_its_worker(store, launcher)
    worker = launcher.processes[0].command[launcher.processes[0].command.index("--worker-id") + 1]
    store.finish(run, worker, RunStatus.SUCCEEDED, result={"schema_version": 1})
    launcher.processes[0].code = 0
    dispatcher.poll_once()
    assert store.get(run).status is RunStatus.SUCCEEDED
    dispatcher.stop()


def test_stop_now_ends_the_worker_and_cancels_the_run(store, tmp_path):
    launcher = KillableLauncher()
    dispatcher = _dispatcher(store, tmp_path, launcher)
    store.create("owner", DataSourceKind.CSV, SPEC)
    dispatcher.poll_once()
    run = _claimed_by_its_worker(store, launcher)
    assert dispatcher.stop_run(run) is True
    assert launcher.processes[0].code == -9
    assert store.get(run).status is RunStatus.CANCELLED
    assert dispatcher.active_runs() == ()                         # its log is released
    dispatcher.stop()


def test_stop_now_finds_a_worker_left_by_an_earlier_server(store, tmp_path):
    run = store.create("owner", DataSourceKind.CSV, SPEC)
    store.claim(run, "pid1-1")                                    # an earlier server's worker
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                                "JBGRunWorker", "--run", run.key])
    try:
        deadline = time.time() + 10
        while process.pid not in _worker_pids(run.key) and time.time() < deadline:
            time.sleep(0.1)
        assert _worker_pids(run.key) == [process.pid]
        dispatcher = _dispatcher(store, tmp_path, FakeLauncher())
        assert dispatcher.stop_run(run) is True
        assert process.wait(timeout=10) is not None
        assert store.get(run).status is RunStatus.CANCELLED
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


# --- The worker --------------------------------------------------------------------------

def test_inner_steps_are_checkpoints(store):
    from JBGRunWorker import RunCancelled, StoreProgressLogger
    run = store.create("owner", DataSourceKind.CSV, SPEC)
    store.claim(run, "w")
    engine = SimpleNamespace(start_inline_progress=lambda *a: None,
                             update_inline_progress=lambda *a: None)
    observed = StoreProgressLogger(engine, store, run, SimpleNamespace(lost=False))
    observed.start_inline_progress("k", "Algorithm combinations", 10, "")
    store.request_cancel(run)
    observed._checked_at = -1e9                                   # the last check was long ago
    with pytest.raises(RunCancelled):
        observed.update_inline_progress("k", 3, "")


def test_inner_checks_read_the_database_at_most_once_a_second(store):
    from JBGRunWorker import StoreProgressLogger
    run = store.create("owner", DataSourceKind.CSV, SPEC)
    engine = SimpleNamespace(update_inline_progress=lambda *a: None)
    observed = StoreProgressLogger(engine, store, run, SimpleNamespace(lost=False))
    reads = []
    observed.checkpoint = lambda: reads.append(1)
    for step in range(50):
        observed.update_inline_progress("k", step, "")
    assert len(reads) == 1


# --- The web interface -------------------------------------------------------------------

def test_the_run_panel_offers_stop_now_after_cancelling(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from JBGFastAPIServer import create_server_app
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    from JBGMetadataStore import MetadataStoreSettings
    from JBGSQLServer import SQLServerSettings
    owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
    client = TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs", sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))
    with client:
        location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                               follow_redirects=False).headers["location"]
        project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
        client.post(f"/ui/projects/{project}/open")
        jobs = client.app.state.jbg["store"]
        run = jobs.create(owner, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
            "project_id": project, "configuration_name": "LR"}})
        jobs.claim(run, "pid1-1")
        page = client.get(f"/ui/run?run={run.key}").text
        assert "Avbryt körningen" in page and "Stoppa nu" not in page
        assert client.get(f"/api/runs/{run.key}/overview").json()["cancel_requested"] is False
        client.post(f"/ui/run/{run.key}/cancel", data={"back": "/ui/run"})
        page = client.get(f"/ui/run?run={run.key}").text
        assert "Avbryter: körningen stannar efter steget" in page
        assert '<button class="btn act warning">Stoppa nu</button>' in page
        assert "Avbryt körningen" not in page
        done = client.post(f"/ui/run/{run.key}/stop", data={"back": "/ui/run"},
                           follow_redirects=False)
        assert done.headers["location"] == f"/ui/run?run={run.key}&message=stopped"
        assert jobs.get(run).status is RunStatus.CANCELLED
        assert "Körningen är stoppad." in client.get(done.headers["location"]).text
        again = client.post(f"/ui/run/{run.key}/stop", data={"back": "/ui/run"},
                            follow_redirects=False)
        assert again.headers["location"].endswith("message=stop_failed")
        foreign = jobs.create("someone else", DataSourceKind.CSV, SPEC)
        jobs.claim(foreign, "w")
        client.post(f"/ui/run/{foreign.key}/stop", data={"back": "/ui/run"})
        assert jobs.get(foreign).status is RunStatus.RUNNING


def test_worker_ids_are_unique_per_dispatcher_and_a_lost_claim_interrupts_nothing(store, tmp_path):
    """Two servers in one process (a restart in the tests) gave the same IDs;
    a worker that lost the race for a job must not interrupt the winner's run."""
    from JBGRunWorker import EXIT_NOT_CLAIMED
    first, second = FakeLauncher(), FakeLauncher()
    run = store.create("owner", DataSourceKind.CSV, SPEC)
    a, b = _dispatcher(store, tmp_path, first), _dispatcher(store, tmp_path, second)
    a.poll_once()
    b.poll_once()                                                 # the job is still pending
    ids = [p.command[p.command.index("--worker-id") + 1] for p in first.processes + second.processes]
    assert len(set(ids)) == 2
    store.claim(run, ids[0])                                      # the first worker wins
    second.processes[0].code = EXIT_NOT_CLAIMED
    b.poll_once()
    assert store.get(run).status is RunStatus.RUNNING
    a.stop()
    b.stop()
