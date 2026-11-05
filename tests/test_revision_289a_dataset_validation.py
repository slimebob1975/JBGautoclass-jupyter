"""289a: dataset validation and preview, the same for SQL and CSV."""
import csv
import datetime
import decimal
import io
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")

from fastapi.testclient import TestClient  # noqa: E402

from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402

MISSING = "00000000-0000-4000-8000-000000000000"
HEADER = ["id", "sepal", "petal", "species", "note"]
ROWS = [[1, 5.1, 1.4, "a", "x"], [2, 7.0, 4.7, "b", "y"], [3, 6.3, 6.0, "a", "z"],
        [4, 4.9, 1.4, "b", "w"]]
SQL_TYPES = {"id": "int", "sepal": "float", "petal": "float", "species": "nvarchar",
             "note": "nvarchar"}
SERVER = SQLServerSettings(host="sql.example.invalid", driver="SQL Server", probe_catalog="Db")
GOOD = {"id_column": "id", "class_column": "species",
        "data_numerical_columns": ["sepal", "petal"], "data_text_columns": ["note"]}


def _csv_bytes(rows=ROWS):
    out = io.StringIO()
    writer = csv.writer(out, delimiter=";")
    writer.writerow(HEADER)
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


class FakeLayer:
    """Answers the DataLayer queries the service uses, from ROWS."""

    def __init__(self, rows=ROWS, types=SQL_TYPES):
        self.rows, self.types, self.queries = rows, types, []

    def get_table_columns(self, catalog, table):
        return dict(self.types)

    def count_data_rows(self, catalog, table):
        return len(self.rows)

    def _values(self, column):
        return [row[HEADER.index(column)] for row in self.rows]

    def get_column_uniqueness_stats(self, catalog, table, column):
        values = self._values(column)
        present = [v for v in values if v not in (None, "")]
        return {"total": len(values), "nonnull": len(present), "distinct": len(set(present))}

    def get_column_cardinalities(self, catalog, table, columns):
        return {c: len({v for v in self._values(c) if v not in (None, "")}) for c in columns}

    def get_data_list_from_query(self, query):
        self.queries.append(query)
        return [tuple(row) for row in self.rows[:50]]


def _client(tmp_path, subject="developer"):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SERVER,
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _project(client, source, roles=GOOD, rows=ROWS, role="training"):
    project = client.post("/api/projects", json={"name": source.upper(), "source": source}).json()["project_id"]
    if source == "csv":
        dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "D", "role": role},
                              files={"file": ("d.csv", _csv_bytes(rows), "text/csv")}).json()["dataset_id"]
    else:
        dataset = client.post(f"/api/projects/{project}/datasets", json={
            "name": "D", "role": role, "catalog": "Db", "table": "dbo.Iris"}).json()["dataset_id"]
    configuration = client.post(f"/api/projects/{project}/configurations",
                                json={"name": "C", "settings": {"connection": roles}}).json()
    return project, dataset, configuration["configuration_id"]


def _validate(client, project, dataset, layer=None, configuration=None):
    sessions = client.app.state.jbg["sessions"]
    original = sessions.data_layer_for
    if layer is not None:
        sessions.data_layer_for = lambda principal, config, logger: layer
    try:
        if configuration:
            # 293a: the roles belong to the dataset; take them from the
            # configuration this test was written with.
            from jbg_test_roles import store_roles
            settings = client.get(f"/api/projects/{project}/configurations/{configuration}"
                                  ).json()["settings"]
            store_roles(client, project, dataset, settings)
        return client.post(f"/api/projects/{project}/datasets/{dataset}/validate")
    finally:
        sessions.data_layer_for = original


def _codes(response):
    return [(c["code"], c["status"], c["message"]) for c in response.json()["checks"]]


# --- The same check list for both sources ------------------------------------------------

def test_dataset_only_validation(tmp_path):
    with _client(tmp_path) as client:
        project, dataset, _ = _project(client, "csv")
        response = _validate(client, project, dataset)
        assert response.status_code == 200, response.text
        assert [c[:2] for c in _codes(response)] == [
            ("source_reachable", "passed"), ("source_readable", "passed"), ("has_rows", "passed")]
        assert response.json()["validation_state"] == "valid"
        assert client.get(f"/api/projects/{project}/datasets/{dataset}").json()[
            "validation_state"] == "valid"


@pytest.mark.parametrize("change, failing", [
    ({}, []),
    # Not unique as ID, and then also the same column as the class.
    ({"id_column": "species"}, ["id_column", "class_column"]),
    ({"class_column": "missing"}, ["class_column"]),
    ({"class_column": "id", "id_column": "id"}, ["class_column"]),
    # 293a: one role per column makes "ID also a feature" impossible; a
    # missing feature column still fails the same check.
    ({"data_numerical_columns": ["sepal", "nope"]}, ["feature_columns"]),
    ({"data_numerical_columns": ["sepal", "note"], "data_text_columns": []}, ["feature_types"]),
])
def test_csv_and_sql_give_the_same_check_list(tmp_path, change, failing):
    roles = dict(GOOD, **change)
    results = {}
    for source in ("csv", "sql"):
        with _client(tmp_path / source) as client:
            project, dataset, configuration = _project(client, source, roles=roles)
            response = _validate(client, project, dataset, FakeLayer() if source == "sql" else None,
                                 configuration)
            assert response.status_code == 200, response.text
            results[source] = (_codes(response), response.json()["validation_state"])
    assert results["csv"] == results["sql"]
    codes, state = results["csv"]
    assert [c[0] for c in codes] == ["source_reachable", "source_readable", "has_rows",
                                     "id_column", "class_column", "feature_columns", "feature_types"]
    failed = [c[0] for c in codes if c[1] == "failed"]
    assert failed == failing
    assert state == ("invalid" if failing else "valid")


def test_prediction_datasets_need_no_classes(tmp_path):
    rows = [row[:3] + [None, row[4]] for row in ROWS]
    with _client(tmp_path) as client:
        # 293a: prediction datasets have no roles of their own; without a
        # training dataset with roles only the basic checks apply.
        project, dataset, _ = _project(client, "sql", role="prediction")
        response = _validate(client, project, dataset, FakeLayer(rows=rows))
        assert [c["code"] for c in response.json()["checks"]] == [
            "source_reachable", "source_readable", "has_rows"]
        assert response.json()["validation_state"] == "valid", response.json()


# --- SQL access and failures ------------------------------------------------------------------

def test_sql_needs_the_sign_in_and_settings(tmp_path):
    with _client(tmp_path) as client:
        project, dataset, _ = _project(client, "sql")
        response = _validate(client, project, dataset)               # real registry, no sign-in
        assert (response.status_code, response.json()) == (409, {"detail": "SQL sign-in required"})
        assert client.get(f"/api/projects/{project}/datasets/{dataset}").json()[
            "validation_state"] == "unverified"


def test_unreachable_and_unreadable_sources(tmp_path):
    with _client(tmp_path) as client:
        project, dataset, _ = _project(client, "sql")
        missing_table = FakeLayer(types={})
        codes = _codes(_validate(client, project, dataset, missing_table))
        assert [c[:2] for c in codes] == [("source_reachable", "passed"),
                                          ("source_readable", "failed"), ("has_rows", "skipped")]

        class Broken:
            def __call__(self, *a):
                raise ValueError("server said: secret detail")
        sessions = client.app.state.jbg["sessions"]
        sessions.data_layer_for = Broken()
        response = client.post(f"/api/projects/{project}/datasets/{dataset}/validate", json={})
        assert [c[:2] for c in _codes(response)] == [("source_reachable", "failed"),
                                                      ("source_readable", "skipped"),
                                                      ("has_rows", "skipped")]
        assert "secret" not in response.text
        assert response.json()["validation_state"] == "invalid"


# --- Preview ------------------------------------------------------------------------------------

def test_csv_preview(tmp_path):
    with _client(tmp_path) as client:
        project, dataset, _ = _project(client, "csv")
        preview = client.get(f"/api/projects/{project}/datasets/{dataset}/preview").json()
    assert preview["row_count"] == 4 and len(preview["rows"]) == 4
    assert [c["name"] for c in preview["columns"]] == HEADER
    assert {c["name"]: c["type"] for c in preview["columns"]}["note"] == "text"
    assert preview["rows"][0] == ["1", "5.1", "1.4", "a", "x"]


def test_sql_preview_quotes_real_columns_and_converts_values(tmp_path):
    odd = {"id": "int", "we]ird": "decimal", "when": "date", "blob": "varbinary"}
    rows = [(1, decimal.Decimal("2.50"), datetime.date(2026, 11, 5), b"\x00\x01")]
    layer = FakeLayer(rows=rows, types=odd)
    with _client(tmp_path) as client:
        project, dataset, _ = _project(client, "sql")
        sessions = client.app.state.jbg["sessions"]
        sessions.data_layer_for = lambda principal, config, logger: layer
        preview = client.get(f"/api/projects/{project}/datasets/{dataset}/preview").json()
    assert preview["rows"] == [[1, 2.5, "2026-11-05", None]]
    assert layer.queries == ["SELECT TOP(50) [id], [we]]ird], [when], [blob] FROM [Db].[dbo].[Iris]"]


# --- Ownership ------------------------------------------------------------------------------------

def test_other_users_and_unknown_references(tmp_path):
    with _client(tmp_path) as mine:
        project, dataset, _ = _project(mine, "csv")
        assert mine.post(f"/api/projects/{project}/datasets/{MISSING}/validate", json={}).status_code == 404
    with _client(tmp_path, subject="bob") as theirs:
        for response in (theirs.post(f"/api/projects/{project}/datasets/{dataset}/validate", json={}),
                         theirs.get(f"/api/projects/{project}/datasets/{dataset}/preview")):
            assert (response.status_code, response.json()) == (404, {"detail": "project unavailable"})
