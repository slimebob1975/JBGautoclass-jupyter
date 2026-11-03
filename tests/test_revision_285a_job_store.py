"""285a: durable run jobs and progress events (SQLAlchemy over SQLite files)."""
from datetime import timedelta
from threading import Barrier, Thread

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import update  # noqa: E402

from JBGApplicationContracts import DataSourceKind, RunReference, RunStatus  # noqa: E402
from JBGRunJobStore import RunJobStore, TABLE_PREFIX  # noqa: E402

SPEC = {"scope": {"tenant_id": "t", "project_id": "p"}, "dataset": "artifact-1",
        "configuration": {"connection": {"sql_username": "", "sql_password": ""}}}


@pytest.fixture
def store(tmp_path):
    jobs = RunJobStore.sqlite(tmp_path / "jobs.sqlite")
    jobs.create_tables()
    return jobs


def _running(store, owner="owner-a", worker="worker-1"):
    run = store.create(owner, DataSourceKind.CSV, SPEC)
    assert store.claim(run, worker)
    return run


# --- Creating and reading -------------------------------------------------------------

def test_job_round_trip_and_restart_survival(tmp_path):
    path = tmp_path / "jobs.sqlite"
    first = RunJobStore.sqlite(path)
    first.create_tables()
    run = first.create("owner-a", DataSourceKind.SQL, SPEC)
    first.record_progress(run, "Starting up ...", 0.0)
    first._engine.dispose()                      # the server process stops

    reopened = RunJobStore.sqlite(path)          # a new process opens the same file
    reopened.create_tables()                     # idempotent
    record = reopened.get(run)
    assert record.owner == "owner-a" and record.source is DataSourceKind.SQL
    assert record.status is RunStatus.PENDING and record.spec == SPEC
    assert record.created_at.tzinfo is not None
    assert [event.message for event in reopened.progress(run)] == ["Starting up ..."]


def test_foreign_and_unknown_runs_look_the_same(store):
    run = store.create("owner-a", DataSourceKind.CSV, SPEC)
    assert store.get_owned("owner-a", run).run == run
    with pytest.raises(KeyError):
        store.get_owned("owner-b", run)
    with pytest.raises(KeyError):
        store.get_owned("owner-a", RunReference("00000000-0000-4000-8000-000000000000"))
    assert [r.run for r in store.list_for_owner("owner-a")] == [run]
    assert store.list_for_owner("owner-b") == ()


@pytest.mark.parametrize("spec", [
    {"configuration": {"connection": {"sql_password": "secret"}}},
    {"password": "secret"},
    {"items": [{"Password": "x"}]},
    {"value": object()},
    {1: "non-string key"},
])
def test_spec_must_be_plain_json_without_passwords(store, spec):
    with pytest.raises(ValueError):
        store.create("owner-a", DataSourceKind.CSV, spec)


# --- Lifecycle ------------------------------------------------------------------------

def test_claim_is_won_exactly_once_across_threads(tmp_path):
    path = tmp_path / "jobs.sqlite"
    setup = RunJobStore.sqlite(path)
    setup.create_tables()
    run = setup.create("owner-a", DataSourceKind.CSV, SPEC)
    barrier, wins = Barrier(4), []

    def contender(index):
        store = RunJobStore.sqlite(path)       # separate engine, as in another process
        barrier.wait()
        wins.append(store.claim(run, f"worker-{index}"))

    threads = [Thread(target=contender, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert sorted(wins) == [False, False, False, True]
    assert setup.get(run).status is RunStatus.RUNNING


def test_only_the_claiming_worker_heartbeats_and_finishes(store):
    run = _running(store)
    assert not store.heartbeat(run, "worker-2")
    assert store.heartbeat(run, "worker-1")
    assert not store.finish(run, "worker-2", RunStatus.SUCCEEDED, result={"kind": "x"})
    assert store.finish(run, "worker-1", RunStatus.SUCCEEDED,
                        result={"schema_version": 1, "kind": "classification",
                                "mispredicted_count": 3, "class_count": 2})
    record = store.get(run)
    assert record.status is RunStatus.SUCCEEDED and record.finished_at is not None
    assert record.result["mispredicted_count"] == 3
    assert not store.finish(run, "worker-1", RunStatus.FAILED)   # already finished
    assert not store.heartbeat(run, "worker-1")


def test_finish_validates_status_result_and_error(store):
    run = _running(store)
    with pytest.raises(ValueError):
        store.finish(run, "worker-1", RunStatus.RUNNING)
    with pytest.raises(ValueError):
        store.finish(run, "worker-1", RunStatus.SUCCEEDED, result={"frame": object()})
    with pytest.raises(ValueError):
        store.finish(run, "worker-1", RunStatus.FAILED, error_type="x" * 200)
    assert store.finish(run, "worker-1", RunStatus.FAILED, error_type="RuntimeError")
    assert store.get(run).error_type == "RuntimeError"


def test_cancel_pending_at_once_and_running_cooperatively(store):
    pending = store.create("owner-a", DataSourceKind.CSV, SPEC)
    assert store.request_cancel(pending) is RunStatus.CANCELLED
    assert not store.claim(pending, "worker-1")

    running = _running(store)
    assert store.request_cancel(running) is RunStatus.RUNNING
    assert store.cancel_requested(running)
    assert store.finish(running, "worker-1", RunStatus.CANCELLED)

    finished = _running(store)
    store.finish(finished, "worker-1", RunStatus.SUCCEEDED)
    assert store.request_cancel(finished) is RunStatus.SUCCEEDED
    assert not store.cancel_requested(finished)


def test_stale_running_jobs_are_marked_interrupted(store):
    stale, fresh = _running(store), _running(store, worker="worker-2")
    old = store.get(stale).heartbeat_at - timedelta(minutes=10)
    with store._engine.begin() as connection:
        connection.execute(update(store._jobs).where(store._jobs.c.run_id == stale.key)
                           .values(heartbeat_at=old))
    assert store.mark_stale_interrupted(timedelta(minutes=2)) == (stale,)
    assert store.get(stale).status is RunStatus.INTERRUPTED
    assert store.get(fresh).status is RunStatus.RUNNING
    assert not store.finish(stale, "worker-1", RunStatus.SUCCEEDED)
    with pytest.raises(ValueError):
        store.mark_stale_interrupted(timedelta(0))


def test_pending_lists_oldest_first(store):
    first = store.create("owner-a", DataSourceKind.CSV, SPEC)
    second = store.create("owner-b", DataSourceKind.SQL, SPEC)
    store.claim(first, "worker-1")
    third = store.create("owner-a", DataSourceKind.CSV, SPEC)
    assert [record.run for record in store.pending()] == [second, third]


# --- Progress -------------------------------------------------------------------------

def test_progress_is_bounded_and_validated(tmp_path):
    store = RunJobStore.sqlite(tmp_path / "jobs.sqlite", max_events=3)
    store.create_tables()
    run = store.create("owner-a", DataSourceKind.CSV, SPEC)
    for index in range(5):
        store.record_progress(run, f"step {index}", index / 4)
    assert [e.message for e in store.progress(run)] == ["step 2", "step 3", "step 4"]
    with pytest.raises(ValueError):
        store.record_progress(run, "x", 1.5)
    with pytest.raises(TypeError):
        store.record_progress(run, "x", True)
    with pytest.raises(KeyError):
        store.record_progress(RunReference("00000000-0000-4000-8000-000000000000"), "x")


# --- SQL Server compatibility without a server -----------------------------------------

def test_tables_compile_for_sql_server_in_a_schema():
    from sqlalchemy.dialects import mssql
    from sqlalchemy.schema import CreateTable
    from JBGRunJobStore import _metadata
    _, jobs, events = _metadata("example_schema")
    for table in (jobs, events):
        ddl = str(CreateTable(table).compile(dialect=mssql.dialect()))
        assert f"example_schema.{TABLE_PREFIX}" in ddl
    assert "DATETIMEOFFSET" in str(CreateTable(jobs).compile(dialect=mssql.dialect()))
