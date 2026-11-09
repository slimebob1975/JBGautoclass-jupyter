"""295c: predicting with a saved model (SQL); tables without a class column."""
import json
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from Config import Config

SERVER_ARGS = dict(host="sql.example.invalid", driver="SQL Server", probe_catalog="Db")


# --- Engine: a prediction table may lack the class column -------------------------------------

def _layer(train, predict, columns):
    from SQLDataLayer import DataLayer
    config = Config()
    config.connection.data_catalog, config.connection.data_table = "Db", "dbo.Nov"
    config.connection.id_column, config.connection.class_column = "id", "species"
    config.connection.data_numerical_columns = ["sepal"]
    config.mode.train, config.mode.predict = train, predict
    layer = DataLayer.__new__(DataLayer)
    layer.config, layer.logger = config, SimpleNamespace()
    calls = []
    layer.get_table_columns = lambda db, table: calls.append(table) or columns
    return layer, calls


def test_prediction_without_a_class_column_predicts_every_row():
    layer, calls = _layer(False, True, {"id": "int", "sepal": "float"})
    query = layer.get_data_query_with_order(10, True, True)
    assert "NULL AS [species]" in query and "WHERE" not in query and "ORDER BY [species]" not in query
    assert layer.get_class_distribution_query() == "SELECT NULL, COUNT(*) FROM [Db].[dbo].[Nov]"
    # Looked up once per run, by the plain name INFORMATION_SCHEMA can match.
    assert calls == ["dbo.Nov"]


@pytest.mark.parametrize("train, predict, columns", [
    (False, True, {"id": "int", "sepal": "float", "species": "nvarchar"}),   # column present
    (True, False, {"id": "int", "sepal": "float"}),                          # training
    (True, True, {"id": "int", "sepal": "float"}),                           # train and predict
])
def test_everything_else_keeps_the_class_column_queries(train, predict, columns):
    layer, _ = _layer(train, predict, columns)
    assert "GROUP BY [species]" in layer.get_class_distribution_query()
    assert "NULL AS" not in layer.get_data_query_with_order(10, True, True)


def test_a_prediction_run_registers_no_model(tmp_path):
    from JBGRunWorker import register_trained_model
    model = tmp_path / "m.sav"
    model.write_bytes(b"model")
    run_config = SimpleNamespace(get_model_filename=lambda: str(model))
    record = SimpleNamespace(spec={"project_run": {"project_id": "p", "kind": "prediction"}})
    assert register_trained_model(None, record, run_config) is False


def test_translations_reload_when_the_catalogue_changes():
    import JBGWebUI
    po = JBGWebUI.WEB / "i18n" / "sv.po"
    backup = Path(tempfile.mkdtemp()) / "sv.po"
    shutil.copy2(po, backup)
    try:
        assert JBGWebUI.translator("sv")("A text added after start") == "A text added after start"
        with po.open("a", encoding="utf-8") as stream:
            stream.write('\nmsgid "A text added after start"\nmsgstr "En text tillagd efter start"\n')
        later = time.time() + 5
        os.utime(po, (later, later))
        assert JBGWebUI.translator("sv")("A text added after start") == "En text tillagd efter start"
    finally:
        shutil.copy2(backup, po)
        os.utime(po, (time.time() + 10, time.time() + 10))
        JBGWebUI.translator("sv")


# --- Prediction runs through the API ----------------------------------------------------------------

pytest.importorskip("sqlalchemy")


@contextmanager
def _sql_signed_in():
    from SQLDataLayer import DataLayer
    with patch("pyodbc.drivers", return_value=["SQL Server"]), \
            patch.object(DataLayer, "can_connect", autospec=True, return_value=True), \
            patch("os.getlogin", return_value="developer"):
        yield


def _client(tmp_path, subject="developer"):
    pytest.importorskip("fastapi")
    pytest.importorskip("python_multipart")
    from fastapi.testclient import TestClient
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    from JBGSQLServer import SQLServerSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs", sql_server=SQLServerSettings(**SERVER_ARGS),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _trained_sql_model(client, tmp_path, *, state="available", with_file=True):
    """A SQL project with a training dataset, a finished training run and its model."""
    from JBGApplicationContracts import DataSourceKind, RunStatus
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    from JBGModelCatalog import ModelState
    owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
    project = client.post("/api/projects", json={"name": "S", "source": "sql"}).json()["project_id"]
    url = f"/api/projects/{project}/datasets"
    training = client.post(url, json={"name": "T", "role": "training", "catalog": "Db",
                                      "table": "dbo.Train"}).json()["dataset_id"]
    prediction = client.post(url, json={"name": "Nov", "role": "prediction", "catalog": "Db",
                                        "table": "dbo.Nov"}).json()["dataset_id"]
    config = Config()
    config.name = project
    config.connection.data_catalog, config.connection.data_table = "Db", "dbo.Train"
    config.connection.id_column, config.connection.class_column = "id", "species"
    config.connection.data_numerical_columns = ["sepal"]
    folder = tmp_path / "models" / project
    folder.mkdir(parents=True)
    from uuid import uuid4
    model_key = str(uuid4())                   # training runs name the file by a UUID
    config.io.model_path, config.io.model_name = str(folder), model_key
    if with_file:
        Path(config.get_model_filename()).write_bytes(b"model")
    store = client.app.state.jbg["store"]
    run = store.create(owner, DataSourceKind.SQL, {
        "spec_version": 1, "configuration": config._config_to_json(username=""),
        "project_run": {"project_id": project, "dataset_id": training,
                        "configuration_name": "LR", "configuration_version": 3}})
    store.claim(run, "w")
    store.finish(run, "w", RunStatus.SUCCEEDED, result={"schema_version": 1})
    catalog = client.app.state.jbg["models"]
    entry = catalog.register(owner, project, "LR v3", model_key, dataset_id=training, run_id=run.key)
    if state == "available":
        catalog.set_state(owner, project, entry.model_id, ModelState.AVAILABLE)
    return project, entry.model_id, prediction


def _sign_in(client):
    return client.post("/api/sql/session", json={"method": "sql", "username": "analyst",
                                                  "password": "secret"})


def test_a_prediction_run_reuses_the_training_configuration(tmp_path):
    from JBGApplicationContracts import RunReference
    with _sql_signed_in(), _client(tmp_path) as client:
        project, model_id, prediction = _trained_sql_model(client, tmp_path)
        assert _sign_in(client).status_code in (200, 201)
        started = client.post(f"/api/projects/{project}/predictions",
                              json={"model_id": model_id, "dataset_id": prediction})
        assert started.status_code == 201, started.text
        spec = client.app.state.jbg["store"].get(RunReference(started.json()["run_id"])).spec
        from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
        owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
        trained_key = client.app.state.jbg["models"].get_owned(owner, project, model_id).artifact_key
    mode, connection, io = (spec["configuration"][k] for k in ("mode", "connection", "io"))
    assert (mode["train"], mode["predict"], mode["mispredicted"]) == (False, True, False)
    assert connection["data_table"] == "dbo.Nov" and connection["class_column"] == "species"
    assert io["model_name"] == trained_key                        # exactly the trained model
    assert Path(io["model_path"]).name == project
    assert spec["project_run"]["kind"] == "prediction" and spec["project_run"]["model_id"] == model_id


def test_without_a_dataset_the_training_dataset_is_predicted(tmp_path):
    from JBGApplicationContracts import RunReference
    with _sql_signed_in(), _client(tmp_path) as client:
        project, model_id, _ = _trained_sql_model(client, tmp_path)
        _sign_in(client)
        started = client.post(f"/api/projects/{project}/predictions", json={"model_id": model_id})
        spec = client.app.state.jbg["store"].get(RunReference(started.json()["run_id"])).spec
    assert spec["configuration"]["connection"]["data_table"] == "dbo.Train"


@pytest.mark.parametrize("case, detail", [
    ("registered", "the model is not available"),
    ("no file", "the model file is missing"),
    ("not signed in", "SQL sign-in required"),
])
def test_predictions_are_refused_clearly(tmp_path, case, detail):
    with _sql_signed_in(), _client(tmp_path) as client:
        project, model_id, _ = _trained_sql_model(
            client, tmp_path, state="registered" if case == "registered" else "available",
            with_file=case != "no file")
        if case != "not signed in":
            _sign_in(client)
        response = client.post(f"/api/projects/{project}/predictions", json={"model_id": model_id})
    assert (response.status_code, response.json()) == (409, {"detail": detail})


def test_csv_projects_and_other_users_are_refused(tmp_path):
    with _sql_signed_in(), _client(tmp_path) as client:
        project, model_id, _ = _trained_sql_model(client, tmp_path)
        csv = client.post("/api/projects", json={"name": "C", "source": "csv"}).json()["project_id"]
        refused = client.post(f"/api/projects/{csv}/predictions", json={"model_id": model_id})
        assert refused.json() == {"detail": "predicting with a saved model is not available for CSV projects yet"}
    with _sql_signed_in(), _client(tmp_path, subject="bob") as theirs:
        response = theirs.post(f"/api/projects/{project}/predictions", json={"model_id": model_id})
        assert response.status_code == 404


def test_the_run_tab_offers_prediction_with_the_chosen_model(tmp_path):
    with _sql_signed_in(), _client(tmp_path) as client:
        project, model_id, prediction = _trained_sql_model(client, tmp_path)
        client.post(f"/ui/projects/{project}/open", follow_redirects=False)
        page = client.get("/ui/run").text
        assert "Prediktera med en sparad modell" in page and "Välj först en modell" in page
        client.post(f"/ui/models/{model_id}/choose", follow_redirects=False)
        _sign_in(client)
        page = client.get("/ui/run").text
        assert "LR v3" in page and f'<option value="{prediction}">Nov</option>' in page
        started = client.post("/ui/run/predict", data={"dataset_id": prediction}, follow_redirects=True).text
        assert "Prediktionen har startat." in started and ">Prediktion</span>" in started


def test_csv_projects_show_no_prediction_card_yet(tmp_path):
    with _client(tmp_path) as client:
        location = client.post("/ui/projects", data={"name": "C", "source": "csv"},
                               follow_redirects=False).headers["location"]
        project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
        client.post(f"/ui/projects/{project}/open", follow_redirects=False)
        assert "Prediktera med en sparad modell" not in client.get("/ui/run").text
