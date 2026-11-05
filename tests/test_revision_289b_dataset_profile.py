"""289b: column profile and class distribution; plain SQL names."""
import csv
import io
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")

from fastapi.testclient import TestClient  # noqa: E402

from JBGDatasetInspection import MAX_CLASSES, QUERY_SECONDS  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402

HEADER = ["id", "sepal", "species", "note"]
ROWS = [[1, 5.1, "a", "x"], [2, None, "b", ""], [3, 6.3, "a", "z"], [4, 4.9, None, "z"],
        [5, 5.0, "a", "x"]]
TYPES = {"id": "int", "sepal": "float", "species": "nvarchar", "note": "varchar"}
SERVER = SQLServerSettings(host="sql.example.invalid", driver="SQL Server", probe_catalog="Db")


def _csv_bytes(rows=ROWS, header=HEADER):
    out = io.StringIO()
    writer = csv.writer(out, delimiter=";")
    writer.writerow(header)
    writer.writerows([["" if v is None else v for v in row] for row in rows])
    return out.getvalue().encode("utf-8")


class FakeHelper:
    def __init__(self, layer):
        self.layer, self.Connection = layer, None

    def connect(self):
        return type("Connection", (), {"timeout": 0})()

    def disconnect(self):
        pass

    def get_data_from_query(self, sql):
        layer = self.layer
        layer.queries.append(sql)
        layer.timeouts.append(self.Connection.timeout)
        if layer.timeout:
            raise RuntimeError("[HYT00] [Microsoft][ODBC Driver] Query timeout expired")
        missing = lambda v: v is None or v == ""
        if "INFORMATION_SCHEMA.TABLES" in sql:
            return [(layer.table_type,)]
        index = HEADER.index("species")
        if "GROUP BY" in sql:
            counts = {}
            for row in layer.rows:
                if not missing(row[index]):
                    counts[str(row[index])] = counts.get(str(row[index]), 0) + 1
            ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            return ordered[:MAX_CLASSES + 1]
        if "COUNT(DISTINCT" in sql or sql.count("SUM(") > 1:
            result = [len(layer.rows)]
            for i, column in enumerate(HEADER):
                values = [row[i] for row in layer.rows]
                result.append(sum(1 for v in values if missing(v)))
                result.append(len({v for v in values if not missing(v)}))
            return [tuple(result)]
        return [(len(layer.rows), sum(1 for row in layer.rows if missing(row[index])))]


class FakeLayer:
    def __init__(self, rows=ROWS, total=None, table_type="BASE TABLE", timeout=False):
        self.rows, self.total, self.table_type, self.timeout = rows, total, table_type, timeout
        self.queries, self.timeouts = [], []

    def get_table_columns(self, catalog, table):
        return dict(TYPES)

    def count_data_rows(self, catalog, table):
        return self.total if self.total is not None else len(self.rows)

    def get_connection(self):
        return FakeHelper(self)


def _client(tmp_path, subject="developer"):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SERVER,
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _dataset(client, source, data=None):
    project = client.post("/api/projects", json={"name": source, "source": source}).json()["project_id"]
    if source == "csv":
        response = client.post(f"/api/projects/{project}/datasets/csv",
                               data={"name": "D", "role": "training"},
                               files={"file": ("d.csv", data or _csv_bytes(), "text/csv")})
    else:
        response = client.post(f"/api/projects/{project}/datasets", json={
            "name": "D", "role": "training", "catalog": "Db", "table": "dbo.Iris"})
    return f"/api/projects/{project}/datasets/{response.json()['dataset_id']}"


def _with_layer(client, layer):
    client.app.state.jbg["sessions"].data_layer_for = lambda principal, config, logger: layer


# --- Exact results, the same for CSV and SQL ---------------------------------------------

def test_profile_is_exact_and_the_same_for_csv_and_sql(tmp_path):
    with _client(tmp_path / "csv") as client:
        csv_profile = client.get(_dataset(client, "csv") + "/profile").json()
    layer = FakeLayer()
    with _client(tmp_path / "sql") as client:
        url = _dataset(client, "sql")
        _with_layer(client, layer)
        sql_profile = client.get(url + "/profile").json()
    shape = lambda p: [(c["name"], c["missing"], c["distinct"], c["missing_share"]) for c in p["columns"]]
    assert shape(csv_profile) == shape(sql_profile) == [
        ("id", 0, 5, 0.0), ("sepal", 1, 4, 0.2), ("species", 1, 2, 0.2), ("note", 1, 2, 0.2)]
    assert csv_profile["basis"]["method"] == sql_profile["basis"]["method"] == "exact"
    assert not any("TABLESAMPLE" in q for q in layer.queries)
    assert set(layer.timeouts) == {QUERY_SECONDS}


def test_class_distribution_is_the_same_for_csv_and_sql(tmp_path):
    with _client(tmp_path / "csv") as client:
        from_csv = client.get(_dataset(client, "csv") + "/class-distribution",
                              params={"column": "species"}).json()
    with _client(tmp_path / "sql") as client:
        url = _dataset(client, "sql")
        _with_layer(client, FakeLayer())
        from_sql = client.get(url + "/class-distribution", params={"column": "species"}).json()
    for result in (from_csv, from_sql):
        assert result["classes"] == [{"value": "a", "count": 3, "share": 0.6},
                                     {"value": "b", "count": 1, "share": 0.2}]
        assert (result["missing"], result["more_classes"]) == (1, False)


# --- Large tables, views and time limits -------------------------------------------------------

def test_large_tables_are_sampled_and_marked_as_estimated(tmp_path):
    layer = FakeLayer(total=5_000_000)
    with _client(tmp_path) as client:
        url = _dataset(client, "sql")
        _with_layer(client, layer)
        profile = client.get(url + "/profile").json()
    assert profile["basis"] == {"method": "tablesample", "total_rows": 5_000_000,
                                "percent": 2.0, "rows_used": 5}
    sampled = [q for q in layer.queries if "COUNT(DISTINCT" in q]
    assert sampled and all("TABLESAMPLE SYSTEM (2.0000 PERCENT)" in q for q in sampled)


def test_large_views_are_computed_exactly(tmp_path):
    layer = FakeLayer(total=5_000_000, table_type="VIEW")
    with _client(tmp_path) as client:
        url = _dataset(client, "sql")
        _with_layer(client, layer)
        profile = client.get(url + "/profile").json()
    assert profile["basis"]["method"] == "exact"
    assert not any("TABLESAMPLE" in q for q in layer.queries)


def test_a_query_over_the_time_limit_gives_a_fixed_message(tmp_path):
    with _client(tmp_path) as client:
        url = _dataset(client, "sql")
        _with_layer(client, FakeLayer(timeout=True))
        for path in ("/profile", "/class-distribution?column=species"):
            response = client.get(url + path)
            assert (response.status_code, response.json()) == (
                422, {"detail": "the table is too large for a quick profile"})
            assert "ODBC" not in response.text


# --- Columns and class caps ------------------------------------------------------------------------

def test_unknown_columns_and_many_classes(tmp_path):
    rows = [[i, 1.0, f"class{i:03d}", "x"] for i in range(MAX_CLASSES + 20)]
    with _client(tmp_path) as client:
        url = _dataset(client, "csv", data=_csv_bytes(rows))
        assert client.get(url + "/class-distribution", params={"column": "nope"}).json() == {
            "detail": "column unavailable"}
        result = client.get(url + "/class-distribution", params={"column": "species"}).json()
    assert len(result["classes"]) == MAX_CLASSES and result["more_classes"] is True


# --- Plain SQL names (fix for 286b/289a) ----------------------------------------------------------

@pytest.mark.parametrize("catalog, table", [
    ("Db", "dbo.X'; DROP TABLE dbo.Y--"), ("Db", "[dbo].[X]"), ("Db", "Cases"),
    ("Db;", "dbo.X"), ("Db", "dbo.X/*"), ("Db", 'dbo."X"'), ("Db", "dbo. X"),
])
def test_sql_dataset_names_must_be_plain(tmp_path, catalog, table):
    with _client(tmp_path) as client:
        project = client.post("/api/projects", json={"name": "S", "source": "sql"}).json()["project_id"]
        response = client.post(f"/api/projects/{project}/datasets", json={
            "name": "D", "role": "training", "catalog": catalog, "table": table})
        assert response.status_code == 422
        assert client.get(f"/api/projects/{project}/datasets").json() == []


def test_profile_is_owner_bound(tmp_path):
    with _client(tmp_path) as mine:
        url = _dataset(mine, "csv")
    with _client(tmp_path, subject="bob") as theirs:
        for path in ("/profile", "/class-distribution?column=species"):
            response = theirs.get(url + path)
            assert (response.status_code, response.json()) == (404, {"detail": "project unavailable"})
