"""285c: dispatcher, restart handling and the durable run service.

The dispatcher is tested with a fake process launcher, so every exit code and
restart situation is deterministic. The native class starts a real worker
process through the dispatcher (skipped without the full ML dependencies).
"""
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import update  # noqa: E402

from JBGApplicationContracts import DataSourceKind, DatasetReference, RunReference, RunStatus  # noqa: E402
from JBGDurableRuns import DurableRunService  # noqa: E402
from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode  # noqa: E402
from JBGRunDispatcher import RunDispatcher, _detach_options, worker_environment  # noqa: E402
from JBGRunJobStore import RunJobStore  # noqa: E402
from JBGRunPreparationService import PreparedRun, RunConfiguration  # noqa: E402
from JBGSQLSessions import SQLCredentials  # noqa: E402
from Storage.ArtifactContract import StorageScope  # noqa: E402

SPEC = {"spec_version": 1}


class FakeStdin:
    def __init__(self):
        self.data, self.closed = b"", False

    def write(self, data):
        self.data += data

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, command, **options):
        self.command, self.options, self.code = command, options, None
        self.stdin = FakeStdin()

    def poll(self):
        return self.code


class FakeLauncher:
    def __init__(self):
        self.processes = []

    def __call__(self, command, **options):
        process = FakeProcess(command, **options)
        self.processes.append(process)
        return process


def _store(tmp_path, name="jobs.sqlite"):
    store = RunJobStore.sqlite(tmp_path / name)
    store.create_tables()
    return store


def _dispatcher(store, tmp_path, launcher, **options):
    options.setdefault("heartbeat_seconds", 1)
    options.setdefault("stale_after", timedelta(minutes=5))
    return RunDispatcher(store, store_url=f"sqlite:///{tmp_path / 'jobs.sqlite'}",
                         log_dir=tmp_path / "logs", launcher=launcher, **options)


def _sessions(**owners):
    # 285f: the dispatcher also passes the job's connection target; the real
    # registry checks it against the signed-in server (see 284c/285f tests).
    return SimpleNamespace(
        credentials_for_owner=lambda owner, connection=None: owners.get(owner))


def _age_heartbeat(store, run, minutes):
    old = store.get(run).heartbeat_at - timedelta(minutes=minutes)
    with store._engine.begin() as connection:
        connection.execute(update(store._jobs).where(store._jobs.c.run_id == run.key)
                           .values(heartbeat_at=old))


@pytest.fixture
def store(tmp_path):
    jobs = _store(tmp_path)
    yield jobs
    jobs.close()


# --- Store additions ---------------------------------------------------------------------

def test_awaiting_credentials_lifecycle(store):
    sql = store.create("owner", DataSourceKind.SQL, SPEC)
    assert store.mark_awaiting_credentials(sql)
    assert store.get(sql).status is RunStatus.AWAITING_CREDENTIALS
    assert not store.claim(sql, "worker")                    # not claimable while waiting
    assert [r.run for r in store.awaiting_credentials()] == [sql]
    assert store.requeue(sql) and store.get(sql).status is RunStatus.PENDING
    store.mark_awaiting_credentials(sql)
    assert store.request_cancel(sql) is RunStatus.CANCELLED   # queued: cancelled at once
    with pytest.raises(ValueError):
        store.mark_awaiting_credentials(store.create("owner", DataSourceKind.CSV, SPEC))


def test_fail_unstarted_only_for_pending_jobs(store):
    pending = store.create("owner", DataSourceKind.CSV, SPEC)
    assert store.fail_unstarted(pending, "WorkerStartFailed")
    assert store.get(pending).status is RunStatus.FAILED
    running = store.create("owner", DataSourceKind.CSV, SPEC)
    store.claim(running, "worker")
    assert not store.fail_unstarted(running, "WorkerStartFailed")
    assert [r.run for r in store.running()] == [running]


# --- Dispatcher ---------------------------------------------------------------------------

def test_starts_workers_up_to_the_limit_without_secrets_in_arguments(store, tmp_path):
    first = store.create("owner", DataSourceKind.CSV, SPEC)
    second = store.create("owner", DataSourceKind.CSV, SPEC)
    launcher = FakeLauncher()
    dispatcher = _dispatcher(store, tmp_path, launcher, artifacts=tmp_path / "artifacts")
    assert dispatcher.poll_once() == (first,)
    assert dispatcher.poll_once() == ()             # started, not claimed: not started again
    process = launcher.processes[0]
    assert process.command[1:3] == ["-m", "JBGRunWorker"] and first.key in process.command
    assert process.stdin.data == b"\n" and process.stdin.closed
    env = process.options["env"]
    assert env["PYTHONIOENCODING"] == "utf-8" and env["PYTHONHASHSEED"] == "0"
    assert process.options.get("creationflags") or process.options.get("start_new_session")
    assert store.claim(first, "worker")              # behave like the real worker
    store.finish(first, "worker", RunStatus.SUCCEEDED)
    process.code = 0
    assert dispatcher.poll_once() == (second,)
    assert (tmp_path / "logs" / f"run_{first.key}.log").exists()
    dispatcher.stop()


def test_sql_credentials_travel_only_through_stdin(store, tmp_path):
    run = store.create("owner", DataSourceKind.SQL, SPEC)
    login = SQLCredentials.sql_login("analyst", "S3cret!")
    launcher = FakeLauncher()
    dispatcher = _dispatcher(store, tmp_path, launcher, sessions=_sessions(owner=login))
    assert dispatcher.poll_once() == (run,)
    process = launcher.processes[0]
    assert json.loads(process.stdin.data) == {"windows": False, "username": "analyst",
                                              "password": "S3cret!"}
    assert "S3cret!" not in " ".join(process.command)
    assert "S3cret!" not in json.dumps(process.options["env"])
    assert "S3cret!" not in json.dumps(store.get(run).spec)
    dispatcher.stop()


def test_sql_job_waits_for_sign_in_then_runs(store, tmp_path):
    run = store.create("owner", DataSourceKind.SQL, SPEC)
    owners, launcher = {}, FakeLauncher()
    dispatcher = _dispatcher(store, tmp_path, launcher,
                             sessions=SimpleNamespace(
                                 credentials_for_owner=lambda owner, connection=None: owners.get(owner)))
    assert dispatcher.poll_once() == ()
    assert store.get(run).status is RunStatus.AWAITING_CREDENTIALS
    owners["owner"] = SQLCredentials.windows_login()              # the user signs in again
    assert dispatcher.poll_once() == (run,)
    assert json.loads(launcher.processes[0].stdin.data)["windows"] is True
    dispatcher.stop()


def test_restart_interrupts_dead_jobs_and_respects_live_ones(tmp_path):
    before = _store(tmp_path)
    dead = before.create("owner", DataSourceKind.CSV, SPEC)
    alive = before.create("owner", DataSourceKind.CSV, SPEC)
    queued = before.create("owner", DataSourceKind.CSV, SPEC)
    before.claim(dead, "old-worker-1")
    before.claim(alive, "old-worker-2")
    _age_heartbeat(before, dead, minutes=10)
    before.close()                                                # the server stops

    after = _store(tmp_path)                                      # a new server starts
    launcher = FakeLauncher()
    dispatcher = _dispatcher(after, tmp_path, launcher, max_workers=1)
    assert dispatcher.poll_once() == ()           # the surviving worker still uses the slot
    assert after.get(dead).status is RunStatus.INTERRUPTED
    assert after.get(alive).status is RunStatus.RUNNING
    after.finish(alive, "old-worker-2", RunStatus.SUCCEEDED)
    assert dispatcher.poll_once() == (queued,)
    dispatcher.stop()
    after.close()


@pytest.mark.parametrize("code, claimed, signed_out, expected", [
    # The worker got no credentials and the user has signed out meanwhile.
    (3, False, True, RunStatus.AWAITING_CREDENTIALS),
    (1, False, False, RunStatus.FAILED),          # crashed before claiming
    (1, True, False, RunStatus.RUNNING),          # crashed after claiming: heartbeat decides
    (2, False, False, RunStatus.PENDING),         # someone else claimed or it was cancelled
])
def test_reaping_worker_exit_codes(store, tmp_path, code, claimed, signed_out, expected):
    run = store.create("owner", DataSourceKind.SQL, SPEC)
    owners, launcher = {"owner": SQLCredentials.windows_login()}, FakeLauncher()
    dispatcher = _dispatcher(store, tmp_path, launcher,
                             sessions=SimpleNamespace(
                                 credentials_for_owner=lambda owner, connection=None: owners.get(owner)))
    dispatcher.poll_once()
    if claimed:
        store.claim(run, "worker")
    if signed_out:
        owners.clear()
    launcher.processes[0].code = code
    with patch.object(dispatcher, "_launch"):      # observe reaping only
        dispatcher.poll_once()
    assert store.get(run).status is expected
    if code == 1 and not claimed:
        assert store.get(run).error_type == "WorkerStartFailed"
    dispatcher.stop()


def test_failed_process_start_does_not_loop(store, tmp_path):
    run = store.create("owner", DataSourceKind.CSV, SPEC)
    dispatcher = _dispatcher(store, tmp_path, Mock(side_effect=OSError("no python")))
    dispatcher.poll_once()
    assert store.get(run).status is RunStatus.FAILED
    assert dispatcher.poll_once() == ()


def test_background_loop_starts_promptly_on_wake(store, tmp_path):
    launcher = FakeLauncher()
    dispatcher = _dispatcher(store, tmp_path, launcher, poll_seconds=60)
    dispatcher.start()
    try:
        time.sleep(0.2)
        run = store.create("owner", DataSourceKind.CSV, SPEC)
        dispatcher.wake()
        deadline = time.time() + 5
        while not launcher.processes and time.time() < deadline:
            time.sleep(0.05)
        assert launcher.processes and run.key in launcher.processes[0].command
    finally:
        dispatcher.stop()
    assert dispatcher.active_runs() == ()


def test_dispatcher_validates_its_settings(store, tmp_path):
    with pytest.raises(ValueError):
        _dispatcher(store, tmp_path, FakeLauncher(), max_workers=0)
    with pytest.raises(ValueError):
        _dispatcher(store, tmp_path, FakeLauncher(), heartbeat_seconds=30,
                    stale_after=timedelta(seconds=60))


def test_worker_environment_and_detach_options():
    env = worker_environment({"PYTHONPATH": "other"})
    assert env["PYTHONPATH"].split(os.pathsep)[1] == "other"
    assert env["PYTHONUTF8"] == "1"
    options = _detach_options()
    if os.name == "nt":
        import subprocess
        assert options["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
        assert options["creationflags"] & subprocess.CREATE_NO_WINDOW
    else:
        assert options == {"start_new_session": True}


# --- Durable run service ---------------------------------------------------------------------

def _principal(subject="developer"):
    return AuthenticatedPrincipal(AuthenticationMode.LOCAL, subject)


def _csv_prepared(scope, key="artifact-1"):
    # Shaped exactly as RunPreparationService.prepare returns it.
    from Config import Config
    return PreparedRun(dataset=DatasetReference(DataSourceKind.CSV, key),
                       resolved_dataset=SimpleNamespace(scope=scope),
                       configuration=RunConfiguration(DataSourceKind.CSV, Config()))


def test_submit_stores_what_was_authorized_and_wakes_the_dispatcher(store):
    scope, woken = StorageScope(str(uuid4()), str(uuid4())), []
    service = DurableRunService(store, scope_for=lambda p: scope,
                                on_submit=lambda: woken.append(1))
    me = _principal()
    response = service.submit(me, _csv_prepared(scope), configuration_id="roles-1")
    assert response["status"] == "pending" and response["has_error"] is False
    record = store.get(RunReference(response["run_id"]))
    assert record.owner == me.principal_key
    assert record.spec["csv"] == {"tenant_id": scope.tenant_id, "project_id": scope.project_id,
                                  "artifact_id": "artifact-1", "configuration_id": "roles-1"}
    assert woken == [1]


def test_submit_refuses_a_foreign_scope_and_unauthorized_sql(store):
    scope = StorageScope(str(uuid4()), str(uuid4()))
    service = DurableRunService(store, scope_for=lambda p: scope)
    foreign = _csv_prepared(StorageScope(str(uuid4()), str(uuid4())))
    with pytest.raises(PermissionError, match="dataset unavailable"):
        service.submit(_principal(), foreign, configuration_id="roles-1")
    from Config import Config
    sql = PreparedRun(dataset=DatasetReference(DataSourceKind.SQL, "[Db].[dbo].[Cases]"),
                      resolved_dataset=object(),
                      configuration=RunConfiguration(DataSourceKind.SQL, Config()))
    with pytest.raises(PermissionError, match="SQL dataset authorization required"):
        service.submit(_principal(), sql)
    assert store.list_for_owner(_principal().principal_key) == ()


def test_reads_are_owner_bound_and_hide_messages(store):
    scope = StorageScope(str(uuid4()), str(uuid4()))
    service = DurableRunService(store, scope_for=lambda p: scope)
    me, other = _principal("me").principal_key, _principal("other").principal_key
    run = RunReference(service.submit(_principal("me"), _csv_prepared(scope),
                                      configuration_id="roles-1")["run_id"])
    store.claim(run, "worker")
    store.record_progress(run, "secret pipeline name", 0.5)
    assert service.progress(me, run)[0] == {"run_id": run.key, "percent": 0.5,
                                            "timestamp": service.progress(me, run)[0]["timestamp"]}
    assert "secret" not in json.dumps(service.progress(me, run))
    for read in (service.snapshot, service.progress, service.result, service.result_details,
                 service.cancel):
        with pytest.raises(PermissionError, match="run unavailable"):
            read(other, run)
    assert service.result_details(me, run)["available"] is False
    store.finish(run, "worker", RunStatus.SUCCEEDED,
                 result={"schema_version": 1, "kind": "classification",
                         "mispredicted_count": 2, "class_count": 3})
    details = service.result_details(me, run)
    assert details["available"] and details["summary"]["class_count"] == 3
    assert [r["run_id"] for r in service.list_runs(me)] == [run.key]
    assert service.list_runs(other) == []


def test_cancel_is_owner_bound(store):
    scope = StorageScope(str(uuid4()), str(uuid4()))
    service = DurableRunService(store, scope_for=lambda p: scope)
    me = _principal().principal_key
    run = RunReference(service.submit(_principal(), _csv_prepared(scope),
                                      configuration_id="roles-1")["run_id"])
    assert service.cancel(me, run)["status"] == "cancelled"


# --- Native: dispatcher starts a real worker process ---------------------------------------------

# Import the module, not the class, so pytest does not collect 281c's tests here.
import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402


class CSVDispatchedRun285c(revision_281c.CSVApplicationEndToEnd281c):
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def test_submitted_csv_run_is_executed_by_a_dispatched_worker(self):
        from JBGRunComposition import compose_csv_run
        model_path = self._configure_training()
        self.config.mail.notification_email = ""        # never send real email from tests
        store = RunJobStore.sqlite(self.root / "jobs.sqlite")
        self.addCleanup(store.close)                    # cleanups run last-in first-out
        store.create_tables()
        dispatcher = RunDispatcher(
            store, store_url=f"sqlite:///{self.root / 'jobs.sqlite'}",
            log_dir=self.root / "run-logs", artifacts=self.root / "artifacts",
            script_path=self.root, heartbeat_seconds=5, stale_after=timedelta(minutes=10))
        self.addCleanup(dispatcher.stop)
        service = DurableRunService(store, scope_for=lambda p: self.scope,
                                    on_submit=dispatcher.wake)
        composed = compose_csv_run(self.repo, self.scope, self.roles.configuration_id,
                                   self.logger)
        from JBGRunPreparationService import RunConfiguration
        prepared = composed.preparation.prepare(
            DatasetReference(DataSourceKind.CSV, self.ref.artifact_id),
            RunConfiguration(DataSourceKind.CSV, self.config))
        me = _principal()
        run = RunReference(service.submit(me, prepared,
                                          configuration_id=self.roles.configuration_id)["run_id"])
        dispatcher.poll_once()
        deadline = time.time() + 900
        status = None
        while time.time() < deadline:
            dispatcher.poll_once()
            status = service.snapshot(me.principal_key, run)["status"]
            if status in ("succeeded", "failed", "cancelled", "interrupted"):
                break
            time.sleep(1)
        log = self.root / "run-logs" / f"run_{run.key}.log"
        tail = log.read_bytes().decode("utf-8", errors="replace")[-3000:] if log.exists() else ""
        self.assertEqual(status, "succeeded", tail)
        self.assertTrue(model_path.is_file(), "dispatched worker did not save a model")
        self.assertTrue(service.result_details(me.principal_key, run)["available"])
        log.read_bytes().decode("utf-8")                 # the worker log is valid UTF-8
        for _ in range(50):                              # let the dispatcher release the log
            dispatcher.poll_once()
            if not dispatcher.active_runs():
                break
            time.sleep(0.2)
        self.assertEqual(dispatcher.active_runs(), ())
