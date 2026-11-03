"""285b: one stored job, run by a worker (in-process and as a separate process).

Unit tests use a real Config and SQLite job store, with mocked engine parts.
The native class reuses the synthetic CSV fixture of revision 281c and is
skipped without the full ML dependencies.
"""
import io
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

pytest.importorskip("sqlalchemy")

from JBGApplicationContracts import DataSourceKind, RunReference, RunStatus  # noqa: E402
from JBGRunJobStore import RunJobStore  # noqa: E402
from JBGRunWorker import (EXIT_FINISHED, EXIT_NO_CREDENTIALS, EXIT_NOT_CLAIMED,  # noqa: E402
                          RunCancelled, StoreProgressLogger, _Heartbeat, build_job_spec,
                          read_credentials, run_job)
from Storage.ArtifactContract import StorageScope  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"


def _store(tmp_path):
    store = RunJobStore.sqlite(tmp_path / "jobs.sqlite")
    store.create_tables()
    return store


def _sql_config():
    from Config import Config
    config = Config()
    config.connection.odbc_driver = "SQL Server"
    config.connection.host = "sql.example.invalid"
    config.connection.data_catalog = "ExampleDb"
    config.connection.data_table = "dbo.Cases"
    config.connection.sql_username = "analyst"
    config.connection.sql_password = "S3cret!"
    config.mail.notification_email = ""
    return config


@contextmanager
def _sql_server():
    from SQLDataLayer import DataLayer
    used = []

    def can_connect(layer, verbose=False):
        c = layer.config.connection
        used.append((c.trusted_connection, c.sql_username, c.sql_password))
        return True

    with patch("pyodbc.drivers", return_value=["SQL Server"]), \
            patch.object(DataLayer, "can_connect", autospec=True, side_effect=can_connect):
        yield used


# --- Store lifetime -------------------------------------------------------------------------

def test_closed_store_releases_its_database_file(tmp_path):
    # Windows refuses to delete a file that is still open; Linux does not care.
    path = tmp_path / "jobs.sqlite"
    store = RunJobStore.sqlite(path)
    store.create_tables()
    run = store.create("owner", DataSourceKind.CSV, {"spec_version": 1})
    store.record_progress(run, "step")
    store.close()
    path.unlink()
    assert not path.exists()


# --- Command-line logger (used by the worker) ----------------------------------------------

def test_command_line_logger_prints_current_configuration(tmp_path):
    # 285b: JBGLogger lacked JBGStreamedLogger's data_source fix and raised KeyError.
    from JBGLogger import JBGLogger
    logger = JBGLogger(quiet=True, in_terminal=True, log_dir=tmp_path)
    logger.print_config_settings(_sql_config())


# --- Job specification and credentials ---------------------------------------------------

def test_sql_spec_holds_neither_username_nor_password(tmp_path):
    spec = build_job_spec(DataSourceKind.SQL, _sql_config())
    text = json.dumps(spec)
    assert "S3cret!" not in text and "analyst" not in text
    assert "csv" not in spec
    _store(tmp_path).create("owner", DataSourceKind.SQL, spec)   # accepted by the store


def test_csv_spec_requires_trusted_scope_and_identifiers():
    config, scope = _sql_config(), StorageScope(str(uuid4()), str(uuid4()))
    with pytest.raises(TypeError):
        build_job_spec(DataSourceKind.CSV, config, artifact_id="a", configuration_id="c")
    with pytest.raises(ValueError):
        build_job_spec(DataSourceKind.CSV, config, scope=scope, artifact_id="",
                       configuration_id="c")
    spec = build_job_spec(DataSourceKind.CSV, config, scope=scope, artifact_id="a",
                          configuration_id="c")
    assert spec["csv"] == {"tenant_id": scope.tenant_id, "project_id": scope.project_id,
                           "artifact_id": "a", "configuration_id": "c"}


def test_credentials_are_read_from_one_json_line():
    assert read_credentials(io.StringIO("")) is None
    assert read_credentials(io.StringIO('{"windows": true}\n')).windows
    login = read_credentials(io.StringIO('{"username": "analyst", "password": "pw"}\n'))
    assert (login.windows, login.username) == (False, "analyst")
    with pytest.raises(ValueError):
        read_credentials(io.StringIO('{"username": "analyst"}\n'))


# --- Claiming ---------------------------------------------------------------------------

def test_sql_job_without_credentials_stays_pending(tmp_path):
    store = _store(tmp_path)
    run = store.create("owner", DataSourceKind.SQL, build_job_spec(DataSourceKind.SQL,
                                                                  _sql_config()))
    assert run_job(store, run, "worker-1") == EXIT_NO_CREDENTIALS
    assert store.get(run).status is RunStatus.PENDING


def test_job_claimed_elsewhere_or_cancelled_is_not_run(tmp_path):
    store = _store(tmp_path)
    spec = build_job_spec(DataSourceKind.SQL, _sql_config())
    taken = store.create("owner", DataSourceKind.SQL, spec)
    store.claim(taken, "worker-0")
    cancelled = store.create("owner", DataSourceKind.SQL, spec)
    store.request_cancel(cancelled)
    from JBGSQLSessions import SQLCredentials
    login = SQLCredentials.sql_login("analyst", "pw")
    with patch("JBGRunWorker._compose") as compose:
        assert run_job(store, taken, "worker-1", credentials=login) == EXIT_NOT_CLAIMED
        assert run_job(store, cancelled, "worker-1", credentials=login) == EXIT_NOT_CLAIMED
    compose.assert_not_called()


# --- Progress, cancellation and heartbeat ---------------------------------------------------

def test_progress_is_forwarded_recorded_and_cancellation_escapes_except_exception(tmp_path):
    store = _store(tmp_path)
    run = store.create("owner", DataSourceKind.CSV, {"spec_version": 1})
    store.claim(run, "worker-1")
    engine_logger, heartbeat = Mock(), Mock(lost=False)
    observed = StoreProgressLogger(engine_logger, store, run, heartbeat)
    observed.print_progress("Starting up ...", 0.0)
    observed.print_info("forwarded")
    engine_logger.print_progress.assert_called_once_with(message="Starting up ...", percent=0.0)
    engine_logger.print_info.assert_called_once_with("forwarded")
    assert [e.message for e in store.progress(run)] == ["Starting up ..."]

    store.request_cancel(run)
    with pytest.raises(RunCancelled):
        try:
            observed.print_progress("Check and train algorithms for best model")
        except Exception:            # how the engine guards single candidates
            pytest.fail("cancellation was swallowed by except Exception")


def test_lost_ownership_and_store_errors_at_a_checkpoint(tmp_path):
    store = _store(tmp_path)
    run = store.create("owner", DataSourceKind.CSV, {"spec_version": 1})
    with pytest.raises(RunCancelled):
        StoreProgressLogger(Mock(), store, run, Mock(lost=True)).checkpoint()
    broken = Mock()
    broken.cancel_requested.side_effect = RuntimeError("database unavailable")
    StoreProgressLogger(Mock(), broken, run, Mock(lost=False)).checkpoint()   # no raise


def test_heartbeat_advances_and_detects_lost_ownership(tmp_path):
    from datetime import timedelta
    store = _store(tmp_path)
    run = store.create("owner", DataSourceKind.CSV, {"spec_version": 1})
    store.claim(run, "worker-1")
    first = store.get(run).heartbeat_at
    with _Heartbeat(store, run, "worker-1", 0.05) as heartbeat:
        time.sleep(0.3)
        assert store.get(run).heartbeat_at > first
        time.sleep(0.2)
        store.mark_stale_interrupted(timedelta(microseconds=1))
        time.sleep(0.3)
        assert heartbeat.lost


# --- Whole jobs, in-process --------------------------------------------------------------

def test_sql_job_runs_with_piped_credentials_and_stores_only_the_projection(tmp_path):
    from JBGSQLSessions import SQLCredentials
    store = _store(tmp_path)
    run = store.create("owner", DataSourceKind.SQL, build_job_spec(DataSourceKind.SQL,
                                                                  _sql_config()))
    with _sql_server() as used, patch("AutomaticClassifier.AutomaticClassifier") as engine:
        engine.return_value.run.return_value = {"mispredicted": None}
        code = run_job(store, run, "worker-1", logger=Mock(),
                       credentials=SQLCredentials.sql_login("analyst", "pw"))
    assert code == EXIT_FINISHED
    assert used and set(used) == {(False, "analyst", "pw")}
    record = store.get(run)
    assert record.status is RunStatus.SUCCEEDED
    assert record.result == {"schema_version": 1, "kind": "classification",
                             "mispredicted_count": None, "class_count": None}
    layer = engine.call_args.kwargs["datalayer"]
    assert layer.config.get_data_table(include_database=True) == "[ExampleDb].[dbo].[Cases]"


def test_failure_records_only_the_error_type(tmp_path):
    store = _store(tmp_path)
    scope = StorageScope(str(uuid4()), str(uuid4()))
    spec = build_job_spec(DataSourceKind.CSV, _sql_config(), scope=scope,
                          artifact_id=str(uuid4()), configuration_id=str(uuid4()))
    run = store.create("owner", DataSourceKind.CSV, spec)
    code = run_job(store, run, "worker-1", logger=Mock(), artifacts=tmp_path / "artifacts")
    assert code == EXIT_FINISHED
    record = store.get(run)
    assert record.status is RunStatus.FAILED
    assert record.error_type in {"ValueError", "FileNotFoundError", "KeyError"}
    assert record.result is None


# --- Native engine -------------------------------------------------------------------------

# Import the module, not the class, so pytest does not collect 281c's tests here.
import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402


class CSVRunWorker285b(revision_281c.CSVApplicationEndToEnd281c):
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def _job(self):
        model_path = self._configure_training()
        self.config.mail.notification_email = ""        # never send real email from tests
        store = RunJobStore.sqlite(self.root / "jobs.sqlite")
        # Cleanups run last-in first-out: close before the fixture deletes the
        # temporary folder, which Windows refuses while the file is open.
        self.addCleanup(store.close)
        store.create_tables()
        spec = build_job_spec(DataSourceKind.CSV, self.config, scope=self.scope,
                              artifact_id=self.ref.artifact_id,
                              configuration_id=self.roles.configuration_id)
        return store, store.create("owner", DataSourceKind.CSV, spec), model_path

    def test_worker_process_trains_a_registered_csv_job(self):
        store, run, model_path = self._job()
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH")]))
        env["PYTHONHASHSEED"] = "0"
        completed = subprocess.run(
            [sys.executable, "-m", "JBGRunWorker", "--store-url",
             f"sqlite:///{self.root / 'jobs.sqlite'}", "--run", run.key,
             "--worker-id", "worker-process", "--artifacts", str(self.root / "artifacts"),
             "--script-path", str(self.root), "--log-dir", str(self.root / "logs"),
             "--heartbeat-seconds", "5"],
            input="", capture_output=True, text=True, timeout=900, env=env, cwd=str(SRC))
        self.assertEqual(completed.returncode, EXIT_FINISHED, completed.stderr[-3000:])
        record = store.get(run)
        self.assertEqual(record.status, RunStatus.SUCCEEDED, completed.stderr[-3000:])
        self.assertEqual(record.result["kind"], "classification")
        self.assertTrue(model_path.is_file(), "worker process did not save a model")
        messages = [event.message for event in store.progress(run)]
        self.assertEqual(messages[0], "Starting up ...")
        self.assertIn("Check and train algorithms for best model", messages)

    def test_cancellation_stops_a_running_job_at_the_next_checkpoint(self):
        store, run, model_path = self._job()
        cancel_at = "Check and train algorithms for best model"
        engine_logger = self.logger

        class CancellingLogger:
            # Request cancellation the moment training starts, as a user would.
            def print_progress(self, message=None, percent=None):
                if message == cancel_at:
                    store.request_cancel(run)
                return engine_logger.print_progress(message=message, percent=percent)

            def __getattr__(self, name):
                return getattr(engine_logger, name)

        code = run_job(store, run, "worker-1", logger=CancellingLogger(),
                       artifacts=self.root / "artifacts", script_path=self.root)
        self.assertEqual(code, EXIT_FINISHED)
        self.assertEqual(store.get(run).status, RunStatus.CANCELLED)
        self.assertFalse(model_path.exists(), "a cancelled run must not save a model")
        messages = [event.message for event in store.progress(run)]
        self.assertEqual(messages[-1], cancel_at)
