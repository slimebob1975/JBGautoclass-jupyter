"""285d: durable runs through the HTTP API (Phase 0 gate test).

Unit tests use SQLite stores in temporary folders and a fake worker launcher.
The native class submits a CSV run through the API, restarts the app and
checks that the run completes in a real worker process (skipped without the
full ML dependencies). An SQL Server connection is never needed.
"""
import json
import time
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from JBGApplicationContracts import RunReference, RunStatus  # noqa: E402
from JBGRunComposition import local_csv_storage  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings, metadata_store_settings, open_metadata_store  # noqa: E402
from JBGRunDispatcher import RunDispatcher  # noqa: E402
from JBGSavedConfigurations import SavedConfigurations, UnknownConfiguration  # noqa: E402


class FakeStdin:
    def __init__(self):
        self.data = b""

    def write(self, data):
        self.data += data

    def close(self):
        pass


class FakeProcess:
    def __init__(self, command, **options):
        self.command, self.options, self.stdin = command, options, FakeStdin()

    def poll(self):
        return None                                  # never finishes, never claims


class FakeLauncher:
    def __init__(self):
        self.processes = []

    def __call__(self, command, **options):
        self.processes.append(FakeProcess(command, **options))
        return self.processes[-1]


def _sql_config(name="demo"):
    from Config import Config
    config = Config()
    config.name = name
    config.connection.odbc_driver = "SQL Server"
    config.connection.host = "sql.example.invalid"
    config.connection.data_catalog = "ExampleDb"
    config.connection.data_table = "dbo.Cases"
    config.connection.class_column = "class"
    config.mail.notification_email = ""
    return config


@contextmanager
def _sql_server():
    # Windows authentication asks the OS for the user name (os.getlogin), which
    # fails without a login terminal (e.g. containers); fix it for the test.
    from SQLDataLayer import DataLayer
    with patch("pyodbc.drivers", return_value=["SQL Server"]), \
            patch.object(DataLayer, "can_connect", autospec=True, return_value=True), \
            patch("os.getlogin", return_value="developer"):
        yield


def _sqlite(tmp_path):
    return MetadataStoreSettings(sqlite_path=tmp_path / "run_jobs" / "jobs.sqlite")


def _app(tmp_path, launcher, subject="developer", **options):
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    return create_server_app(
        settings=_sqlite(tmp_path), subject=subject,
        # Never Voila's real CSV storage in the repository root.
        configurations=SavedConfigurations(
            config_dir=config_dir,
            csv_storage=lambda project: local_csv_storage(project, base=tmp_path)),
        run_logs=tmp_path / "run-logs", csv_artifacts=tmp_path / "artifacts",
        dispatcher_options={"launcher": launcher, "poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5), **options})


def _save_sql_configuration(tmp_path, name="autoclassconfig_demo_sql"):
    (tmp_path / "config").mkdir(exist_ok=True)
    _sql_config().save_to_file(filepath=tmp_path / "config" / f"{name}.json")
    return name


# --- Metadata store settings --------------------------------------------------------------

def test_settings_default_to_a_local_sqlite_file(tmp_path):
    settings = metadata_store_settings(tmp_path, environment={})
    assert not settings.uses_sql_server
    assert settings.sqlite_path == tmp_path / "output" / "run_jobs" / "jobs.sqlite"
    store = open_metadata_store(settings, create_tables=True)
    store.close()
    assert settings.sqlite_path.is_file()


def test_sql_server_settings_build_a_url_without_showing_the_password(tmp_path):
    settings = metadata_store_settings(tmp_path, environment={
        "JBG_METADATA_DB_HOST": "sql.example.invalid", "JBG_METADATA_DB_NAME": "ExampleDb",
        "JBG_METADATA_DB_SCHEMA": "example_schema", "JBG_METADATA_DB_USERNAME": "svc",
        "JBG_METADATA_DB_PASSWORD": "S3cret!", "DEFAULT_ODBC_DRIVER": "ODBC Driver 18 for SQL Server"})
    url = settings.url()
    assert settings.uses_sql_server and settings.schema == "example_schema"
    assert url.drivername == "mssql+pyodbc" and url.database == "ExampleDb"
    assert url.query["driver"] == "ODBC Driver 18 for SQL Server"
    assert url.query["TrustServerCertificate"] == "yes" and url.query["Encrypt"] == "yes"
    assert "S3cret!" not in str(url) and "S3cret!" not in repr(settings)
    windows = metadata_store_settings(tmp_path, environment={
        "JBG_METADATA_DB_HOST": "h", "JBG_METADATA_DB_NAME": "d", "JBG_METADATA_DB_DRIVER": "x"})
    assert windows.url().query["Trusted_Connection"] == "yes"


def test_settings_come_from_the_env_file_and_the_environment_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('JBG_METADATA_DB_HOST="from-file"\n'
                                   'JBG_METADATA_DB_NAME="file-db"\n', encoding="utf-8")
    monkeypatch.setenv("JBG_METADATA_DB_NAME", "env-db")
    settings = metadata_store_settings(tmp_path)
    assert (settings.host, settings.database) == ("from-file", "env-db")


def test_workers_read_sql_server_settings_themselves(tmp_path):
    settings = MetadataStoreSettings(host="h", database="d", username="svc",
                                     password="S3cret!", driver="x", sqlite_path=tmp_path / "x")
    store = open_metadata_store(MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
                                create_tables=True)
    launcher = FakeLauncher()
    from JBGApplicationContracts import DataSourceKind
    run = store.create("owner", DataSourceKind.CSV, {"spec_version": 1})
    dispatcher = RunDispatcher(store, store_from_env=True, store_schema="s",
                               log_dir=tmp_path / "logs", launcher=launcher,
                               heartbeat_seconds=1)
    dispatcher.poll_once()
    command = " ".join(launcher.processes[0].command)
    assert "--store-from-env" in command and "--store-url" not in command
    assert settings.password not in command and "--store-schema" not in command
    with pytest.raises(ValueError):
        RunDispatcher(store, log_dir=tmp_path, launcher=launcher)          # neither
    with pytest.raises(ValueError):
        RunDispatcher(store, store_url="sqlite://", store_from_env=True, log_dir=tmp_path)
    dispatcher.stop()
    store.close()
    assert run


def test_worker_takes_exactly_one_store_source():
    from JBGRunWorker import main
    with pytest.raises(SystemExit):
        main(["--run", "r", "--worker-id", "w"])
    with pytest.raises(SystemExit):
        main(["--store-url", "sqlite://", "--store-from-env", "--run", "r", "--worker-id", "w"])


# --- Configuration catalog ----------------------------------------------------------------

def test_sql_catalog_offers_only_listed_names(tmp_path):
    name = _save_sql_configuration(tmp_path)
    (tmp_path / "config" / "notes.json").write_text("{}", encoding="utf-8")
    catalog = SavedConfigurations(config_dir=tmp_path / "config")
    assert catalog.sql_configurations() == [{"name": name, "project": "demo"}]
    assert catalog.sql_configurations("other") == []
    assert catalog.load_sql(name).get_data_table(include_database=True) == "[ExampleDb].[dbo].[Cases]"
    for bad in ("notes", "../config/" + name, name + ".json", "", None):
        with pytest.raises(UnknownConfiguration):
            catalog.load_sql(bad)


def test_invalid_saved_sql_configuration_is_rejected_before_session_or_job(tmp_path):
    """A legacy N/A table must fail synchronously, not in the worker."""
    name = "autoclassconfig_invalid_sql"
    config = _sql_config()
    config.connection.data_catalog = ""
    config.connection.data_table = "N/A"
    (tmp_path / "config").mkdir(exist_ok=True)
    config.save_to_file(filepath=tmp_path / "config" / f"{name}.json")
    launcher = FakeLauncher()
    with TestClient(_app(tmp_path, launcher)) as client:
        # Structural rejection happens before authentication or SQL I/O.
        bad = client.post("/api/runs", json={"source": "sql", "configuration": name})
        assert bad.status_code == 422
        assert bad.json() == {"detail": "SQL configuration invalid: data_catalog is missing or a placeholder"}
        assert "sql.example.invalid" not in bad.text
        assert client.get("/api/runs").json() == []
        assert launcher.processes == []


def test_saved_sql_preflight_rejects_placeholder_table_and_missing_class(tmp_path):
    from JBGRunSubmission import InvalidSavedSQLConfiguration, validate_saved_sql_run_config
    config = _sql_config()
    validate_saved_sql_run_config(config)
    config.connection.data_table = " N/A "
    with pytest.raises(InvalidSavedSQLConfiguration, match="data_table"):
        validate_saved_sql_run_config(config)
    config.connection.data_table = "dbo.Cases"
    config.mode.train = True
    config.connection.class_column = ""
    with pytest.raises(InvalidSavedSQLConfiguration, match="class_column"):
        validate_saved_sql_run_config(config)
    config.connection.class_column = "class"
    validate_saved_sql_run_config(config)


# --- HTTP API -------------------------------------------------------------------------------

def test_sql_run_requires_sign_in_and_credentials_go_to_the_worker_pipe(tmp_path):
    name, launcher = _save_sql_configuration(tmp_path), FakeLauncher()
    with _sql_server(), TestClient(_app(tmp_path, launcher)) as client:
        response = client.post("/api/runs", json={"source": "sql", "configuration": name})
        assert (response.status_code, response.json()) == (409, {"detail": "SQL sign-in required"})
        signed = client.post("/api/sql/session",
                             json={"method": "sql", "username": "analyst", "password": "S3cret!", "configuration": name})
        assert signed.json() == {"signed_in": True, "windows": False}
        created = client.post("/api/runs", json={"source": "sql", "configuration": name})
        assert created.status_code == 201 and created.json()["status"] == "pending"
        run_id = created.json()["run_id"]
        client.app.state.jbg["dispatcher"].poll_once()
        assert [r["run_id"] for r in client.get("/api/runs").json()] == [run_id]
        stdin = launcher.processes[0].stdin.data
        assert json.loads(stdin)["password"] == "S3cret!"
        assert "S3cret!" not in json.dumps(launcher.processes[0].command)
        store = client.app.state.jbg["store"]
        assert "S3cret!" not in json.dumps(store.get(RunReference(run_id)).spec)
        assert client.post(f"/api/runs/{run_id}/cancel").json()["status"] in (
            "cancelled", "running")


def test_requests_never_echo_secrets_or_accept_paths(tmp_path):
    name = _save_sql_configuration(tmp_path)
    with TestClient(_app(tmp_path, FakeLauncher())) as client:
        # Rejected by our own check (no username) ...
        bad = client.post("/api/sql/session", json={"method": "sql", "password": "S3cret!", "configuration": name})
        assert bad.status_code == 422 and "S3cret!" not in bad.text
        assert client.post("/api/sql/session", json={"method": "windows"}).status_code == 422
        assert client.post("/api/sql/session", json={"method": "windows", "configuration": "../x"}).status_code == 404
        # ... and by FastAPI's validation, whose default response echoes the input.
        for body in ({"method": "sql", "username": "a", "password": ["S3cret!"], "configuration": name},
                     {"method": "kerberos", "username": "a", "password": "S3cret!", "configuration": name}):
            bad = client.post("/api/sql/session", json=body)
            assert bad.status_code == 422 and "S3cret!" not in bad.text, bad.text
            assert bad.json() == {"detail": "invalid request"}
        for body in ({"source": "sql", "configuration": "../../secret"},
                     {"source": "csv", "project": "demo", "configuration_id": "unknown"}):
            response = client.post("/api/runs", json=body)
            assert response.status_code == 404, response.text
        assert client.post("/api/runs", json={"source": "sql"}).status_code == 422
        assert client.post("/api/runs", json={"source": "file", "path": "/x"}).status_code == 422


def test_failed_sql_sign_in_reveals_nothing(tmp_path):
    name = _save_sql_configuration(tmp_path)
    from SQLDataLayer import DataLayer
    with patch("pyodbc.drivers", return_value=["SQL Server"]), \
            patch.object(DataLayer, "can_connect", autospec=True, return_value=False), \
            TestClient(_app(tmp_path, FakeLauncher())) as client:
        response = client.post("/api/sql/session",
                               json={"method": "sql", "username": "analyst", "password": "x", "configuration": name})
        assert (response.status_code, response.json()) == (401, {"detail": "SQL sign-in failed"})
        assert client.get("/api/sql/session").json() == {"signed_in": False, "windows": False}


def test_other_users_cannot_see_or_cancel_my_run(tmp_path):
    name = _save_sql_configuration(tmp_path)
    with _sql_server(), TestClient(_app(tmp_path, FakeLauncher())) as mine:
        mine.post("/api/sql/session", json={"method": "windows", "configuration": name})
        run_id = mine.post("/api/runs", json={"source": "sql", "configuration": name}).json()["run_id"]
    with TestClient(_app(tmp_path, FakeLauncher(), subject="bob")) as theirs:
        for response in (theirs.get(f"/api/runs/{run_id}"),
                         theirs.post(f"/api/runs/{run_id}/cancel")):
            assert (response.status_code, response.json()) == (404, {"detail": "run unavailable"})
        assert theirs.get("/api/runs").json() == []


def test_queued_sql_run_waits_for_credentials_after_a_restart(tmp_path):
    name = _save_sql_configuration(tmp_path)
    with _sql_server(), TestClient(_app(tmp_path, FakeLauncher())) as before:
        before.post("/api/sql/session", json={"method": "windows", "configuration": name})
        run_id = before.post("/api/runs", json={"source": "sql",
                                                "configuration": name}).json()["run_id"]
    launcher = FakeLauncher()                       # the server restarts; sessions are gone
    with _sql_server(), TestClient(_app(tmp_path, launcher)) as after:
        dispatcher = after.app.state.jbg["dispatcher"]
        dispatcher.poll_once()
        assert after.get(f"/api/runs/{run_id}").json()["status"] == "awaiting_credentials"
        assert launcher.processes == []
        after.post("/api/sql/session", json={"method": "windows", "configuration": name})       # sign in again
        dispatcher.poll_once()
        assert len(launcher.processes) == 1 and run_id in launcher.processes[0].command
        assert json.loads(launcher.processes[0].stdin.data)["windows"] is True


# --- Native: the Phase 0 gate run ----------------------------------------------------------------

# Import the module, not the class, so pytest does not collect 281c's tests here.
import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402


class CSVRunThroughAPI285d(revision_281c.CSVApplicationEndToEnd281c):
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def _release_worker_files(self):
        """Wait until the worker process has let go of its files, then empty the folder.

        The worker was started by the first (stopped) server instance, so no
        dispatcher can wait for its exit. Windows refuses to delete files that
        the exiting worker still holds open (its log, the SQLite store).
        """
        import shutil
        deadline = time.time() + 120
        for child in list(self.root.iterdir()):
            while child.exists():
                try:
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
                except PermissionError:
                    if time.time() > deadline:
                        raise
                    time.sleep(0.5)

    def test_csv_run_started_through_the_api_survives_a_server_restart(self):
        from Storage.CSVGUITrainingConfig import prepare_csv_gui_training_config
        # Cleanups run last-in first-out: this one runs before the fixture
        # deletes the temporary folder.
        self.addCleanup(self._release_worker_files)
        self._configure_training()
        self.config.mail.notification_email = ""
        csv_models = self.root / "csv-models"
        # What Voila does at training start: a record holding roles and settings.
        _, voila_record = prepare_csv_gui_training_config(
            self.config, self.repo, self.scope, self.ref.artifact_id,
            id_column="row_id", class_column="target", data_columns=("f1", "f2", "f3", "f4"),
            csv_format={"delimiter": ",", "quotechar": '"', "encoding": "utf-8"},
            model_root=csv_models)
        (self.root / "config").mkdir()

        def app():
            return create_server_app(
                settings=MetadataStoreSettings(sqlite_path=self.root / "jobs.sqlite"),
                configurations=SavedConfigurations(
                    config_dir=self.root / "config",
                    csv_storage=lambda project: (self.repo, self.scope)),
                csv_artifacts=self.root / "artifacts", run_logs=self.root / "run-logs",
                csv_model_root=csv_models,
                dispatcher_options={"script_path": self.root, "heartbeat_seconds": 5,
                                    "stale_after": timedelta(minutes=10), "poll_seconds": 1})

        with TestClient(app()) as first:
            listing = first.get("/api/configurations", params={"project": "demo"}).json()
            self.assertIn(voila_record.configuration_id,
                          [item["configuration_id"] for item in listing["csv"]])
            created = first.post("/api/runs", json={
                "source": "csv", "project": "demo",
                "configuration_id": voila_record.configuration_id})
            self.assertEqual(created.status_code, 201, created.text)
            run_id = created.json()["run_id"]
        # The first server has stopped; its worker (if started) keeps running.
        with TestClient(app()) as second:
            deadline, status = time.time() + 900, None
            while time.time() < deadline:
                status = second.get(f"/api/runs/{run_id}").json()["status"]
                if status in ("succeeded", "failed", "cancelled", "interrupted"):
                    break
                time.sleep(1)
            log = self.root / "run-logs" / f"run_{run_id}.log"
            tail = log.read_bytes().decode("utf-8", "replace")[-3000:] if log.exists() else ""
            self.assertEqual(status, "succeeded", tail)
            details = second.get(f"/api/runs/{run_id}/result").json()
            self.assertTrue(details["available"])
            self.assertEqual(details["summary"]["kind"], "classification")
            dispatcher = second.app.state.jbg["dispatcher"]
            for _ in range(50):                       # release the log before cleanup
                dispatcher.poll_once()
                if not dispatcher.active_runs():
                    break
                time.sleep(0.2)
        self.assertTrue(list(csv_models.rglob("*.sav")), "no model saved under the CSV model root")


def test_sql_sign_in_uses_selected_saved_connection(tmp_path):
    from SQLDataLayer import DataLayer
    name = _save_sql_configuration(tmp_path)
    observed = []

    def probe(layer, verbose=False):
        observed.append((layer.config.connection.host, layer.config.connection.odbc_driver,
                         layer.config.connection.data_catalog))
        return True

    with patch("pyodbc.drivers", return_value=["SQL Server"]), \
            patch.object(DataLayer, "can_connect", autospec=True, side_effect=probe), \
            TestClient(_app(tmp_path, FakeLauncher())) as client:
        response = client.post("/api/sql/session", json={"method": "windows", "configuration": name})
        assert response.status_code == 200, response.text
        assert observed == [("sql.example.invalid", "SQL Server", "ExampleDb")]


def test_sql_session_is_limited_to_verified_connection(tmp_path):
    name = _save_sql_configuration(tmp_path)
    second = "autoclassconfig_other_server"
    config = _sql_config()
    config.connection.host = "other.example.invalid"
    config.save_to_file(filepath=tmp_path / "config" / f"{second}.json")
    with _sql_server(), TestClient(_app(tmp_path, FakeLauncher())) as client:
        assert client.post("/api/sql/session", json={"method": "windows", "configuration": name}).status_code == 200
        response = client.post("/api/runs", json={"source": "sql", "configuration": second})
        assert (response.status_code, response.json()) == (409, {"detail": "SQL sign-in required"})
        assert client.post("/api/sql/session", json={"method": "windows", "configuration": second}).status_code == 200
        assert client.post("/api/runs", json={"source": "sql", "configuration": second}).status_code == 201
