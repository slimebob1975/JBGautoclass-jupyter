"""298i (noted 2026-11-12): an SQL run with a uniqueidentifier ID column
fetched every row and then failed ("cannot convert float NaN to integer"):
the engine's SQL path orders, splits and writes back rows by whole-number IDs.
Guarded in three layers: suggestions pick only whole-number SQL columns as ID,
Column roles warns before saving, validation's ID check requires a
whole-number type in SQL, and runs (training and prediction) are refused
before a worker starts. CSV IDs are opaque text and unaffected."""
import re
from datetime import timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from JBGDatasetRoles import INTEGER_SQL_TYPES, suggest_roles  # noqa: E402
from JBGProjectRuns import IdColumnNotInteger, RunRequestRefused  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402

SERVER = SQLServerSettings(host="sql.example.invalid", driver="SQL Server", probe_catalog="Db")


# --- Suggestions ------------------------------------------------------------------------------

def _column(name, kind, distinct, missing=0):
    return {"name": name, "type": kind, "distinct": distinct, "missing": missing}


def test_sql_suggests_only_a_whole_number_id():
    columns = [_column("DatasetRadId", "uniqueidentifier", 10), _column("Nr", "int", 10),
               _column("Kind", "nvarchar", 3)]
    assert suggest_roles(columns, 10, source="sql")["DatasetRadId"] == "ignored"
    assert suggest_roles(columns, 10, source="sql")["Nr"] == "id"
    only_text = [_column("Key", "nvarchar", 10), _column("Kind", "nvarchar", 3)]
    assert "id" not in suggest_roles(only_text, 10, source="sql").values()


def test_csv_ids_stay_opaque_text():
    columns = [_column("key", "text", 10), _column("x", "number", 4)]
    assert suggest_roles(columns, 10)["key"] == "id"                  # default source: csv
    assert suggest_roles(columns, 10, source="csv")["key"] == "id"


def test_the_whole_number_types():
    assert INTEGER_SQL_TYPES == {"bigint", "int", "smallint", "tinyint"}


# --- Runs ---------------------------------------------------------------------------------------

class Layer:
    def __init__(self, types):
        self.types = types

    def get_table_columns(self, catalog, table):
        return dict(self.types)


def _compose(types, monkeypatch):
    import JBGProjectRuns
    import JBGRunComposition
    monkeypatch.setattr(JBGRunComposition, "compose_sql_run_on_layer", lambda layer, logger: "composed")
    service = SimpleNamespace(_sessions=SimpleNamespace(
        data_layer_for=lambda principal, config, logger: Layer(types)))
    config = SimpleNamespace(connection=SimpleNamespace(data_catalog="Db", data_table="dbo.T",
                                                        id_column="DatasetRadId"))
    return JBGProjectRuns.ProjectRunService._compose_checked_sql(service, object(), config)


@pytest.mark.parametrize("kind", ["uniqueidentifier", "nvarchar", "decimal", "float"])
def test_a_run_with_another_id_type_is_refused_before_it_starts(kind, monkeypatch):
    with pytest.raises(IdColumnNotInteger) as refused:
        _compose({"DatasetRadId": kind, "x": "int"}, monkeypatch)
    assert isinstance(refused.value, RunRequestRefused)                 # 409 on the API
    assert "whole-number" in str(refused.value)


@pytest.mark.parametrize("kind", ["int", "BIGINT", "smallint", "tinyint"])
def test_a_run_with_a_whole_number_id_goes_ahead(kind, monkeypatch):
    assert _compose({"DatasetRadId": kind}, monkeypatch) == "composed"


def test_an_unreadable_column_list_is_left_to_the_run(monkeypatch):
    # The run's own checks report an unreachable table; this guard does not guess.
    assert _compose({}, monkeypatch) == "composed"


# --- The tabs -----------------------------------------------------------------------------------

def _client(tmp_path):
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SERVER,
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _sql_project(client):
    location = client.post("/ui/projects", data={"name": "S", "source": "sql"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    return project


@pytest.mark.parametrize("path, message", [
    ("/ui/run", "ID-kolumnen i ett SQL-dataset måste innehålla heltal"),
])
def test_the_training_tab_says_why(tmp_path, monkeypatch, path, message):
    import JBGProjectRuns

    def refuse(self, *args, **kwargs):
        raise IdColumnNotInteger("the ID column is not a whole-number column")
    monkeypatch.setattr(JBGProjectRuns.ProjectRunService, "submit", refuse)
    with _client(tmp_path) as client:
        _sql_project(client)
        client.cookies.set("jbg_configuration", "c")
        started = client.post(path, follow_redirects=False)
        page = client.get(started.headers["location"]).text
    assert "message=id_not_integer" in started.headers["location"]
    assert message in page and "(Kolumnroller)" in page


def test_the_prediction_tab_says_why(tmp_path, monkeypatch):
    import JBGProjectRuns

    def refuse(self, *args, **kwargs):
        raise IdColumnNotInteger("the ID column is not a whole-number column")
    monkeypatch.setattr(JBGProjectRuns.ProjectRunService, "submit_prediction", refuse)
    with _client(tmp_path) as client:
        _sql_project(client)
        client.cookies.set("jbg_model", "m")                         # the model on show
        started = client.post("/ui/prediction", data={"dataset_id": "d"}, follow_redirects=False)
    assert "message=id_not_integer" in started.headers["location"]


def test_column_roles_warn_before_saving(tmp_path, monkeypatch):
    import JBGDatasetInspection
    profile = {"columns": [_column("DatasetRadId", "uniqueidentifier", 4),
                           _column("Nr", "int", 4), _column("Size", "float", 3),
                           _column("Kind", "nvarchar", 2)],
               "basis": {"method": "exact", "total_rows": 4, "rows_used": 4}}
    monkeypatch.setattr(JBGDatasetInspection.DatasetInspectionService, "profile",
                        lambda self, principal, project_id, dataset_id: profile)
    with _client(tmp_path) as client:
        project = _sql_project(client)
        sessions = client.app.state.jbg["sessions"]
        sessions.is_signed_in = lambda principal: True
        types = {"DatasetRadId": "uniqueidentifier", "Nr": "int", "Size": "float",
                 "Kind": "nvarchar"}

        class Tables(Layer):
            def count_data_rows(self, catalog, table):
                return 4
        sessions.data_layer_for = lambda principal, config, logger: Tables(types)
        created = client.post(f"/api/projects/{project}/datasets", json={
            "name": "T", "role": "training", "catalog": "Db", "table": "dbo.T"})
        assert created.status_code == 201, created.text
        dataset = created.json()["dataset_id"]
        client.put(f"/api/projects/{project}/datasets/{dataset}/roles", json={"columns": {
            "DatasetRadId": "id", "Size": "numeric", "Kind": "class"}})
        page = client.get(f"/ui/dataset?dataset={dataset}&view=roles").text
        saved = client.put(f"/api/projects/{project}/datasets/{dataset}/roles", json={"columns": {
            "Nr": "id", "Size": "numeric", "Kind": "class"}})
        assert saved.status_code == 200, saved.text
        fixed = client.get(f"/ui/dataset?dataset={dataset}&view=roles").text
    row = page.split('name="role:DatasetRadId"')[1].split("</tr>")[0]
    assert "En ID-kolumn i SQL måste innehålla heltal" in row
    assert "En ID-kolumn i SQL måste innehålla heltal" not in fixed


# --- Validation ---------------------------------------------------------------------------------

def test_validation_requires_a_whole_number_sql_id(tmp_path):
    from JBGDatasetInspection import _SQLFacts

    class Facts(_SQLFacts):
        def __init__(self, types):
            self._types, self.columns, self.rows = types, list(types), 4

    assert Facts({"id": "uniqueidentifier"}).id_type_ok("id") is False
    assert Facts({"id": "int"}).id_type_ok("id") is True
    from JBGDatasetInspection import CHECKS
    failed = dict((code, failed) for code, _, failed in CHECKS)["id_column"]
    assert "whole-number column in an SQL table" in failed
