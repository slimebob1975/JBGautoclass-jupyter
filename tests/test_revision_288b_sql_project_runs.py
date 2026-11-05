"""288b: runs from SQL projects; SQL sign-in per server (decision 2).

A real Config and DataLayer are used; only the ODBC driver list, the
connection test and the OS user name are replaced. No SQL Server is needed.
"""
import json
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import patch

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from JBGApplicationContracts import RunReference  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGSQLServer import SQLServerSettings, sql_server_settings  # noqa: E402

SERVER = SQLServerSettings(host="sql.example.invalid", driver="SQL Server",
                           probe_catalog="ExampleDb", class_catalog="ResultsDb",
                           class_table="dbo.Predictions")
ROLES = {"connection": {"id_column": "id", "class_column": "species",
                        "data_numerical_columns": ["sepal", "petal"]}}


class _Launcher:
    def __init__(self):
        self.started = []

    def __call__(self, command, **options):
        record = {"command": command, "stdin": b""}
        self.started.append(record)

        class Stdin:
            def write(self, data):
                record["stdin"] += data

            def close(self):
                pass
        return type("P", (), {"stdin": Stdin(), "poll": lambda s: None})()


@contextmanager
def _sql_server(accept=True):
    from SQLDataLayer import DataLayer
    attempts = []

    def can_connect(layer, verbose=False):
        c = layer.config.connection
        attempts.append((c.host, c.data_catalog, c.trusted_connection, c.sql_username))
        return accept

    with patch("pyodbc.drivers", return_value=["SQL Server"]), \
            patch.object(DataLayer, "can_connect", autospec=True, side_effect=can_connect), \
            patch("os.getlogin", return_value="developer"):
        yield attempts


def _client(tmp_path, launcher=None, server=SERVER, subject="developer"):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=server,
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5),
                            "launcher": launcher or _Launcher()}))


def _setup(client, roles=ROLES, catalog="DataDb"):
    project = client.post("/api/projects", json={"name": "Ärenden", "source": "sql"}).json()["project_id"]
    dataset = client.post(f"/api/projects/{project}/datasets", json={
        "name": "Ärenden 2025", "role": "training", "catalog": catalog,
        "table": "dbo.Cases"}).json()["dataset_id"]
    client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
    # 293a: the roles belong to the dataset (decision A).
    from jbg_test_roles import store_roles
    store_roles(client, project, dataset, roles)
    configuration = client.post(f"/api/projects/{project}/configurations",
                                json={"name": "Standard", "settings": roles})
    assert configuration.status_code == 201, configuration.text
    return project, dataset, configuration.json()["configuration_id"]


def _sign_in(client, **body):
    return client.post("/api/sql/session", json={"method": "sql", "username": "analyst",
                                                 "password": "S3cret!", **body})


# --- Sign-in per server ------------------------------------------------------------------

def test_sign_in_uses_the_servers_own_sql_settings(tmp_path):
    with _sql_server() as attempts, _client(tmp_path) as client:
        response = _sign_in(client)
        assert response.json() == {"signed_in": True, "windows": False}
    assert attempts == [("sql.example.invalid", "ExampleDb", False, "analyst")]


def test_sign_in_without_sql_settings_is_refused(tmp_path):
    with _sql_server() as attempts, _client(tmp_path, server=SQLServerSettings()) as client:
        response = _sign_in(client)
        assert (response.status_code, response.json()) == (
            409, {"detail": "no SQL Server is configured on this server"})
    assert attempts == []


def test_server_settings_come_from_the_voila_keys(tmp_path):
    settings = sql_server_settings(tmp_path, environment={
        "DEFAULT_HOST": "h", "DEFAULT_ODBC_DRIVER": "d", "DEFAULT_DATA_CATALOG": "c",
        "DEFAULT_CLASSIFICATION_CATALOG": "rc", "DEFAULT_CLASSIFICATION_TABLE": "rt"})
    assert settings == SQLServerSettings("h", "d", "c", "rc", "rt") and settings.configured
    assert not sql_server_settings(tmp_path, environment={}).configured


# --- Runs ------------------------------------------------------------------------------------

def test_sql_project_run_needs_the_sign_in_and_then_starts(tmp_path):
    launcher = _Launcher()
    with _sql_server(), _client(tmp_path, launcher) as client:
        project, dataset, configuration = _setup(client)
        url = f"/api/projects/{project}/runs"
        refused = client.post(url, json={"configuration_id": configuration})
        assert (refused.status_code, refused.json()) == (409, {"detail": "SQL sign-in required"})
        _sign_in(client)
        created = client.post(url, json={"configuration_id": configuration})
        assert created.status_code == 201, created.text
        run_id = created.json()["run_id"]
        spec = client.app.state.jbg["store"].get(RunReference(run_id)).spec
        client.app.state.jbg["dispatcher"].poll_once()
    connection = spec["configuration"]["connection"]
    assert (connection["host"], connection["odbc_driver"]) == ("sql.example.invalid", "SQL Server")
    assert (connection["data_catalog"], connection["data_table"]) == ("DataDb", "dbo.Cases")
    assert (connection["class_catalog"], connection["class_table"]) == ("ResultsDb", "dbo.Predictions")
    assert connection["sql_username"] == "" and connection["sql_password"] == ""
    run = spec["project_run"]
    assert (run["project_id"], run["dataset_id"], run["configuration_version"]) == (project, dataset, 1)
    assert spec["configuration"]["io"] == dict(spec["configuration"]["io"],
                                                model_path=str(tmp_path / "models" / project),
                                                model_name=run["model_id"])
    assert "S3cret!" not in json.dumps(spec)
    assert json.loads(launcher.started[0]["stdin"]) == {"windows": False, "username": "analyst",
                                                        "password": "S3cret!"}


def test_one_sign_in_covers_every_catalog_on_the_server(tmp_path):
    with _sql_server(), _client(tmp_path) as client:
        _sign_in(client)
        project, _, configuration = _setup(client, catalog="AnotherDb")
        created = client.post(f"/api/projects/{project}/runs", json={"configuration_id": configuration})
        assert created.status_code == 201, created.text


def test_signing_in_to_another_server_does_not_count(tmp_path):
    with _sql_server(), _client(tmp_path) as client:
        project, _, configuration = _setup(client)
        # A sign-in bound to another host (e.g. through a Voilà configuration).
        sessions = client.app.state.jbg["sessions"]
        from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
        from JBGSQLSessions import SQLCredentials
        other = SERVER.probe_config()
        other.connection.host = "other.example.invalid"
        sessions.sign_in(AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer"),
                         SQLCredentials.sql_login("analyst", "pw"), other, None)
        refused = client.post(f"/api/projects/{project}/runs", json={"configuration_id": configuration})
        assert (refused.status_code, refused.json()) == (409, {"detail": "SQL sign-in required"})


def test_a_configuration_without_class_column_cannot_start(tmp_path):
    roles = {"connection": {"class_column": "", "data_numerical_columns": ["a"]}}
    with _sql_server(), _client(tmp_path) as client:
        _sign_in(client)
        project, _, configuration = _setup(client, roles=roles)
        response = client.post(f"/api/projects/{project}/runs", json={"configuration_id": configuration})
        assert (response.status_code, response.json()) == (
            422, {"detail": "SQL configuration invalid: class_column is missing or a placeholder"})
        assert client.get("/api/runs").json() == []


def test_prediction_only_configurations_cannot_be_saved_for_sql_either(tmp_path):
    with _sql_server(), _client(tmp_path) as client:
        project = client.post("/api/projects", json={"name": "S", "source": "sql"}).json()["project_id"]
        response = client.post(f"/api/projects/{project}/configurations", json={
            "name": "P", "settings": {"mode": {"train": False, "predict": True}}})
        assert (response.status_code, response.json()) == (422, {"detail": "configuration is invalid"})
