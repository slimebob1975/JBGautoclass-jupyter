"""293b: Dataset tab with SQL lists, the roles table and compatibility."""
import re
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402

SERVER = SQLServerSettings(host="sql.example.invalid", driver="SQL Server", probe_catalog="Db")
CSV = b"id;sepal;note;comment;species\n1;5.1;x;long text one;a\n2;7.0;y;long text two;b\n3;6.3;x;long text three;a\n4;4.9;y;long text four;b\n"


def _client(tmp_path):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SERVER,
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _open(client, source):
    location = client.post("/ui/projects", data={"name": source, "source": source},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    return project


def _csv_training(client):
    project = _open(client, "csv")
    uploaded = client.post("/ui/dataset/csv", data={"name": "Train", "role": "training"},
                           files={"file": ("t.csv", CSV, "text/csv")}, follow_redirects=False)
    dataset = re.search(r"dataset=([0-9a-f-]{36})", uploaded.headers["location"]).group(1)
    client.post(f"/ui/dataset/{dataset}/activate", follow_redirects=False)
    return project, dataset


def _roles_form(roles, limit="30"):
    return {**{f"role:{column}": role for column, role in roles.items()}, "categorical_limit": limit}


GOOD = {"id": "id", "sepal": "numeric", "note": "text", "comment": "text", "species": "class"}


# --- Roles table ---------------------------------------------------------------------------

def test_roles_are_set_through_the_table(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _csv_training(client)
        page = client.get(f"/ui/dataset?dataset={dataset}&view=roles").text
        assert "Inga kolumnroller ännu." in page and 'name="role:species"' in page
        saved = client.post(f"/ui/dataset/{dataset}/roles", data=_roles_form(GOOD, "3"),
                            follow_redirects=False)
        assert "message=roles_saved" in saved.headers["location"]
        stored = client.get(f"/api/projects/{project}/datasets/{dataset}/roles").json()
        assert stored["columns"] == GOOD and stored["categorical_limit"] == 3
        page = client.get(saved.headers["location"]).text
        assert "Kolumnrollerna är sparade." in page
        # note has 2 distinct values (<= 3): categorised; comment has 4: free text.
        rows = dict(re.findall(r"<tr><td>(\w+)</td>.*?<td class=\"muted\">\s*([^<]+?)\s*</td></tr>",
                               page, re.S))
        assert rows["note"].startswith("Kategoriseras (högst gränsen)")
        assert rows["comment"].startswith("Fritext") and rows["id"] == "–"


def test_categorical_role_is_always_categorised_and_ignored_is_derived(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _csv_training(client)
        roles = dict(GOOD, comment="categorical", note="ignored")
        client.post(f"/ui/dataset/{dataset}/roles", data=_roles_form(roles), follow_redirects=False)
        stored = client.get(f"/api/projects/{project}/datasets/{dataset}/roles").json()["columns"]
        assert "note" not in stored and stored["comment"] == "categorical"     # ignored = no role
        page = client.get(f"/ui/dataset?dataset={dataset}&view=roles").text
        assert "Kategoriseras (alltid)" in page and "Används inte" in page


@pytest.mark.parametrize("roles, limit, message", [
    ({"id": "id", "sepal": "numeric"}, "30", "Exakt en kolumn måste vara klasskolumn."),
    ({"sepal": "numeric", "species": "class"}, "30", "Exakt en kolumn måste vara ID-kolumn."),
    ({"id": "id", "species": "class"}, "30", "Minst en kolumn måste vara en variabel."),
    (GOOD, "1", "Gränsen för kategorisering måste vara ett heltal mellan 2 och 1000."),
    (GOOD, "lots", "Gränsen för kategorisering måste vara ett heltal mellan 2 och 1000."),
])
def test_refused_roles_keep_the_choices(tmp_path, roles, limit, message):
    with _client(tmp_path) as client:
        project, dataset = _csv_training(client)
        response = client.post(f"/ui/dataset/{dataset}/roles", data=_roles_form(roles, limit))
        assert response.status_code == 422 and message in response.text
        column, role = next(iter(roles.items()))                          # a kept choice
        assert re.search(rf'name="role:{column}"[^>]*>.*?<option value="{role}" selected',
                         response.text, re.S)
        assert client.get(f"/api/projects/{project}/datasets/{dataset}/roles").json()["columns"] is None


# --- Prediction datasets ---------------------------------------------------------------------

def test_prediction_uploads_name_the_missing_columns(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _csv_training(client)
        client.post(f"/ui/dataset/{dataset}/roles", data=_roles_form(GOOD), follow_redirects=False)
        refused = client.post("/ui/dataset/csv", data={"name": "Nov", "role": "prediction"},
                              files={"file": ("n.csv", b"id;sepal;note\n9;5.0;x\n", "text/csv")})
        assert refused.status_code == 409
        assert "Saknade kolumner: comment" in refused.text
        fine = client.post("/ui/dataset/csv", data={"name": "Dec", "role": "prediction"},
                           files={"file": ("d.csv", b"id;sepal;note;comment\n9;5.0;x;t\n", "text/csv")},
                           follow_redirects=False)
        assert "message=added" in fine.headers["location"]
        prediction = re.search(r"dataset=([0-9a-f-]{36})", fine.headers["location"]).group(1)
        page = client.get(f"/ui/dataset?dataset={prediction}&view=roles").text
        assert "Prediktionsdataset använder träningsdatasetets kolumnroller." in page


# --- SQL lists ----------------------------------------------------------------------------------

class Layer:
    def __init__(self, columns):
        self.columns = columns

    def get_catalogs_as_options(self):
        return ["", "Db", "Other"]

    def get_tables_as_options(self):
        return ["", "dbo.Train", "dbo.Nov", "dbo.Bad"]

    def get_table_columns(self, catalog, table):
        return {c: t for (s, n, c, t) in self.columns if f"{s}.{n}" == table}

    def count_data_rows(self, catalog, table):
        return 4

    def get_data_list_from_query(self, query):
        assert "INFORMATION_SCHEMA.COLUMNS" in query and query.count("FROM [Db]") == 1
        return list(self.columns)


COLUMNS = [("dbo", "Train", "id", "int"), ("dbo", "Train", "sepal", "float"),
           ("dbo", "Train", "species", "nvarchar"),
           ("dbo", "Nov", "id", "int"), ("dbo", "Nov", "sepal", "float"),
           ("dbo", "Bad", "id", "int")]


def test_sql_lists_and_only_compatible_prediction_tables(tmp_path):
    with _client(tmp_path) as client:
        project = _open(client, "sql")
        sessions = client.app.state.jbg["sessions"]
        sessions.is_signed_in = lambda principal: True
        sessions.data_layer_for = lambda principal, config, logger: Layer(COLUMNS)
        page = client.get("/ui/dataset").text
        assert '<option value="Db" selected>' in page                    # the .env catalog
        page = client.get("/ui/dataset?catalog=Db&add_role=training").text
        assert all(f'<option value="dbo.{t}">' in page for t in ("Train", "Nov", "Bad"))
        registered = client.post("/ui/dataset/sql", data={"name": "T", "role": "training",
                                                          "catalog": "Db", "table": "dbo.Train"},
                                 follow_redirects=False)
        training = re.search(r"dataset=([0-9a-f-]{36})", registered.headers["location"]).group(1)
        client.post(f"/ui/dataset/{training}/activate", follow_redirects=False)
        client.post(f"/ui/dataset/{training}/roles", data=_roles_form(
            {"id": "id", "sepal": "numeric", "species": "class"}), follow_redirects=False)
        page = client.get("/ui/dataset?catalog=Db&add_role=prediction").text
        assert '<option value="dbo.Nov">' in page and '<option value="dbo.Train">' in page
        assert '<option value="dbo.Bad">' not in page
        assert "Bara tabeller som är kompatibla med träningsdatasetet visas." in page


def test_a_failing_list_leaves_the_tab_usable(tmp_path):
    class Broken:
        def get_catalogs_as_options(self):
            raise RuntimeError("[driver] detail that must not show")

    with _client(tmp_path) as client:
        _open(client, "sql")
        sessions = client.app.state.jbg["sessions"]
        sessions.is_signed_in = lambda principal: True
        sessions.data_layer_for = lambda principal, config, logger: Broken()
        page = client.get("/ui/dataset")
        assert page.status_code == 200 and "driver" not in page.text
        assert "Logga in på SQL Server för att välja katalog och tabell." in page.text


def test_every_new_text_has_a_swedish_translation():
    import JBGWebUI
    from JBGWebUIDatasets import MESSAGES, ROLE_CHOICES, ROLE_ERRORS
    texts = {t for _, t in MESSAGES.values()} | {label for _, label in ROLE_CHOICES} | set(ROLE_ERRORS)
    assert not sorted(t for t in texts if t not in JBGWebUI._CATALOGS["sv"])
