"""291a: Dataset tab of the web interface."""
import csv
import io
import re
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import patch

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

import JBGDatasetService  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402

SERVER = SQLServerSettings(host="sql.example.invalid", driver="SQL Server", probe_catalog="Db")


def _csv(rows=None):
    out = io.StringIO()
    writer = csv.writer(out, delimiter=";")
    writer.writerow(["id", "sepal", "species"])
    writer.writerows(rows or [[1, 5.1, "a"], [2, 4.7, "b"], [3, "", "a"], [4, 6.0, "<script>x</script>"]])
    return out.getvalue().encode("utf-8")


def _client(tmp_path, subject="developer", server=SERVER):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=server,
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _open_project(client, source):
    location = client.post("/ui/projects", data={"name": source.upper(), "source": source},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    return project


def _upload(client, name="Iris", data=None, role="training"):
    return client.post("/ui/dataset/csv", data={"name": name, "role": role},
                       files={"file": ("iris.csv", data or _csv(), "text/csv")},
                       follow_redirects=False)


def _dataset_id(response):
    return re.search(r"dataset=([0-9a-f-]{36})", response.headers["location"]).group(1)


@contextmanager
def _sql_server():
    from SQLDataLayer import DataLayer
    with patch("pyodbc.drivers", return_value=["SQL Server"]), \
            patch.object(DataLayer, "can_connect", autospec=True, return_value=True), \
            patch("os.getlogin", return_value="developer"):
        yield


# --- Without a project -------------------------------------------------------------------

def test_the_tab_asks_for_a_project_first(tmp_path):
    with _client(tmp_path) as client:
        page = client.get("/ui/dataset").text
        assert "Öppna först ett projekt under fliken Projekt." in page
        assert '<a href="/ui/dataset" class="on"' in page


# --- CSV ---------------------------------------------------------------------------------------

def test_upload_activate_and_the_three_views(tmp_path):
    with _client(tmp_path) as client:
        _open_project(client, "csv")
        uploaded = _upload(client)
        assert "message=added" in uploaded.headers["location"]
        dataset = _dataset_id(uploaded)
        activated = client.post(f"/ui/dataset/{dataset}/activate", follow_redirects=True).text
        assert "Datasetet är aktiverat för träning." in activated
        assert re.search(r"<b>Iris</b>\s*</span>", activated)                   # context bar
        assert "&lt;script&gt;x&lt;/script&gt;" in activated and "<script>x" not in activated
        assert "Tal" in activated                                                 # CSV type word
        profile = client.get(f"/ui/dataset?dataset={dataset}&view=profile").text
        assert "Exakt, alla rader." in profile and "25.0 %" in profile           # 1 of 4 missing
        classes = client.get(f"/ui/dataset?dataset={dataset}&view=classes&column=species").text
        assert re.search(r"<td>a</td><td class=\"num\">2</td>", classes)


def test_validation_shows_the_check_list_and_updates_the_status(tmp_path):
    with _client(tmp_path) as client:
        project = _open_project(client, "csv")
        dataset = _dataset_id(_upload(client))
        page = client.post(f"/ui/dataset/{dataset}/validate").text
        for text in ("Datakällan kan nås.", "Data kunde läsas.", "Datasetet innehåller rader."):
            assert text in page
        assert client.get(f"/api/projects/{project}/datasets/{dataset}").json()[
            "validation_state"] == "valid"
        # 293a: roles belong to the dataset; deliberately wrong ones, stored
        # directly, show up as failed checks.
        from jbg_test_roles import store_roles
        store_roles(client, project, dataset, {"connection": {
            "id_column": "id", "class_column": "nope", "data_numerical_columns": ["sepal"]}})
        page = client.post(f"/ui/dataset/{dataset}/validate").text
        assert "Klasskolumnen saknas" in page and "ID-kolumnen finns och är unik." in page


def test_upload_errors_are_fixed_messages(tmp_path, monkeypatch):
    with _client(tmp_path) as client:
        _open_project(client, "csv")
        # Ragged rows (the 286b example): refused by the structure check.
        assert "message=bad_csv" in _upload(client, data=b"a,b\n1,2,3\n").headers["location"]
        monkeypatch.setattr(JBGDatasetService, "MAX_UPLOAD_BYTES", 10)
        assert "message=too_large" in _upload(client).headers["location"]
        monkeypatch.undo()
        _upload(client)
        duplicate = _upload(client, name="IRIS")
        page = client.get(duplicate.headers["location"]).text
        assert "Det finns redan ett dataset med det namnet." in page


# --- SQL -----------------------------------------------------------------------------------------

def test_sql_sign_in_register_and_preview(tmp_path):
    class Layer:
        def get_table_columns(self, catalog, table):
            return {"id": "int", "species": "nvarchar"}

        def count_data_rows(self, catalog, table):
            return 2

        def get_data_list_from_query(self, query):
            return [(1, "a"), (2, "b")]

    with _sql_server(), _client(tmp_path) as client:
        _open_project(client, "sql")
        assert "Inte inloggad" in client.get("/ui/dataset").text
        registered = client.post("/ui/dataset/sql", data={"name": "Ärenden", "role": "training",
                                                          "catalog": "Db", "table": "dbo.Cases"},
                                 follow_redirects=False)
        dataset = _dataset_id(registered)
        page = client.get(f"/ui/dataset?dataset={dataset}").text
        assert "Logga in på SQL Server för att läsa datasetet." in page        # not signed in
        signed = client.post("/ui/sql/session", data={"method": "sql", "username": "analyst",
                                                      "password": "S3cret!",
                                                      "back": f"/ui/dataset?dataset={dataset}"},
                             follow_redirects=False)
        assert signed.headers["location"] == f"/ui/dataset?dataset={dataset}&message=signed_in"
        client.app.state.jbg["sessions"].data_layer_for = lambda principal, config, logger: Layer()
        page = client.get(signed.headers["location"]).text
        assert "Inloggad på SQL Server." in page and "<td>b</td>" in page
        assert "S3cret!" not in page


def test_bad_table_names_and_failed_sign_in(tmp_path):
    from SQLDataLayer import DataLayer
    with _client(tmp_path) as client:
        _open_project(client, "sql")
        bad = client.post("/ui/dataset/sql", data={"name": "X", "role": "training",
                                                   "catalog": "Db", "table": "dbo.X'; DROP"},
                          follow_redirects=False)
        assert "message=bad_sql" in bad.headers["location"]
        with patch("pyodbc.drivers", return_value=["SQL Server"]), \
                patch.object(DataLayer, "can_connect", autospec=True, return_value=False):
            failed = client.post("/ui/sql/session", data={"method": "sql", "username": "a",
                                                          "password": "S3cret!", "back": "https://evil.example/"},
                                 follow_redirects=False)
        assert failed.headers["location"] == "/ui/dataset?message=signin_failed"
        page = client.get(failed.headers["location"]).text
        assert "SQL-inloggningen misslyckades." in page and "S3cret!" not in page


def test_without_sql_settings_the_tab_says_so(tmp_path):
    with _client(tmp_path, server=SQLServerSettings()) as client:
        _open_project(client, "sql")
        assert "Ingen SQL Server är konfigurerad på den här servern." in client.get("/ui/dataset").text


# --- Ownership and protection --------------------------------------------------------------------------

def test_another_users_project_and_datasets_stay_hidden(tmp_path):
    with _client(tmp_path) as mine:
        project = _open_project(mine, "csv")
        dataset = _dataset_id(_upload(mine))
    with _client(tmp_path, subject="bob") as theirs:
        theirs.cookies.set("jbg_project", project, path="/ui")
        page = theirs.get(f"/ui/dataset?dataset={dataset}").text
        assert "Öppna först ett projekt" in page and "Iris" not in page
        refused = theirs.post(f"/ui/dataset/{dataset}/activate", follow_redirects=False)
        assert refused.headers["location"] == "/ui/dataset"


def test_cross_site_forms_are_refused(tmp_path):
    with _client(tmp_path) as client:
        _open_project(client, "csv")
        response = client.post("/ui/dataset/csv", data={"name": "X", "role": "training"},
                               files={"file": ("x.csv", _csv(), "text/csv")},
                               headers={"origin": "http://evil.example"})
        assert response.status_code == 403
