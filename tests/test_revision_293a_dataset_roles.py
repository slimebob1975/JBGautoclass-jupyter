"""293a: column roles belong to the training dataset (decision A);
prediction datasets must be compatible; SQL catalog and table lists."""
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")

from fastapi.testclient import TestClient  # noqa: E402

from JBGApplicationContracts import RunReference  # noqa: E402
from JBGDatasetRoles import DatasetRoles, IncompatibleDataset, check_compatible  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402

SERVER = SQLServerSettings(host="sql.example.invalid", driver="SQL Server", probe_catalog="Db")
TRAIN = b"id;sepal;petal;note;species\n1;5.1;1.4;x;a\n2;7.0;4.7;y;b\n3;6.3;6.0;z;a\n4;4.9;1.4;w;b\n"
ROLES = {"id": "id", "sepal": "numeric", "petal": "numeric", "note": "categorical",
         "species": "class"}


def _client(tmp_path, subject="developer"):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SERVER,
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _csv_project(client, *, roles=ROLES):
    project = client.post("/api/projects", json={"name": "P", "source": "csv"}).json()["project_id"]
    dataset = _upload(client, project, TRAIN, "Train", "training").json()["dataset_id"]
    client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
    if roles is not None:
        assert client.put(f"/api/projects/{project}/datasets/{dataset}/roles",
                          json={"columns": roles}).status_code == 200
    return project, dataset


def _upload(client, project, data, name, role):
    return client.post(f"/api/projects/{project}/datasets/csv", data={"name": name, "role": role},
                       files={"file": (f"{name}.csv", data, "text/csv")})


def _stored_files(tmp_path, project):
    from Storage.CSVDatasetCatalog import list_csv_dataset_labels
    from JBGProjectStorage import project_csv_storage
    repository, scope = project_csv_storage(project, root=tmp_path / "csv")
    return len(list_csv_dataset_labels(repository, scope))


# --- The roles model --------------------------------------------------------------------

def test_one_role_per_column_and_the_engine_mapping():
    from Config import Config
    roles = DatasetRoles.from_mapping(ROLES)
    config = Config()
    roles.apply_to(config)
    assert (config.connection.id_column, config.connection.class_column) == ("id", "species")
    assert config.connection.data_numerical_columns == ["sepal", "petal"]
    assert config.connection.data_text_columns == ["note"]            # categorical is text
    assert config.mode.category_text_columns == ["note"]


@pytest.mark.parametrize("mapping, message", [
    ({"id": "id", "sepal": "numeric", "nope": "class"}, "does not exist"),
    ({"sepal": "numeric", "species": "class"}, "exactly one column must be the ID"),
    ({"id": "id", "sepal": "id", "species": "class", "petal": "numeric"}, "exactly one column must be the ID"),
    ({"id": "id", "sepal": "numeric"}, "exactly one column must be the class"),
    ({"id": "id", "species": "class", "sepal": "ignored"}, "at least one column must be a feature"),
])
def test_role_rules(mapping, message):
    with pytest.raises(ValueError, match=message):
        DatasetRoles.from_mapping(mapping).check(["id", "sepal", "petal", "note", "species"])
    with pytest.raises(ValueError, match="unknown column role"):
        DatasetRoles.from_mapping({"id": "label"})


def test_compatibility_rule():
    roles = DatasetRoles.from_mapping(ROLES)
    training = {"id": "number", "sepal": "number", "petal": "number", "note": "text", "species": "text"}
    check_compatible(roles, training, {"id": "number", "sepal": "number", "petal": "number",
                                       "note": "text", "extra": "text"})      # no class: fine
    with pytest.raises(IncompatibleDataset) as missing:
        check_compatible(roles, training, {"id": "number", "sepal": "number", "note": "text"})
    assert missing.value.missing == ("petal",)
    with pytest.raises(IncompatibleDataset) as kind:
        check_compatible(roles, training, {"id": "number", "sepal": "text", "petal": "number",
                                           "note": "number"})
    assert kind.value.different == ("sepal",)                # text features may change kind


# --- Roles API -----------------------------------------------------------------------------

def test_roles_are_checked_stored_and_reset_validation(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _csv_project(client, roles=None)
        url = f"/api/projects/{project}/datasets/{dataset}"
        # 293a2 added the categorical limit to the response.
        assert client.get(url + "/roles").json() == {"dataset_id": dataset, "columns": None,
                                                     "categorical_limit": None}
        client.post(url + "/validate")
        assert client.get(url).json()["validation_state"] == "valid"
        bad = client.put(url + "/roles", json={"columns": {"id": "id", "nope": "class",
                                                           "sepal": "numeric"}})
        assert (bad.status_code, bad.json()) == (
            422, {"detail": "a column with a role does not exist in the dataset"})
        assert client.put(url + "/roles", json={"columns": ROLES}).json()["columns"] == ROLES
        assert client.get(url).json()["validation_state"] == "unverified"
        codes = [c["code"] for c in client.post(url + "/validate").json()["checks"]]
        assert codes[3:] == ["id_column", "class_column", "feature_columns", "feature_types"]


def test_prediction_datasets_have_no_roles_and_others_cannot_set_them(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _csv_project(client)
        prediction = _upload(client, project, TRAIN, "Nov", "prediction").json()["dataset_id"]
        refused = client.put(f"/api/projects/{project}/datasets/{prediction}/roles",
                             json={"columns": ROLES})
        assert refused.json() == {"detail": "prediction datasets use the roles of the training dataset"}
    with _client(tmp_path, subject="bob") as theirs:
        response = theirs.put(f"/api/projects/{project}/datasets/{dataset}/roles",
                              json={"columns": ROLES})
        assert (response.status_code, response.json()) == (404, {"detail": "project unavailable"})


# --- Runs use the dataset's roles ------------------------------------------------------------

def test_runs_need_dataset_roles_and_use_them(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _csv_project(client, roles=None)
        configuration = client.post(f"/api/projects/{project}/configurations", json={
            "name": "C", "settings": {"connection": {"class_column": "note", "id_column": "sepal"}}}
        ).json()["configuration_id"]
        url = f"/api/projects/{project}/runs"
        refused = client.post(url, json={"configuration_id": configuration})
        assert refused.json() == {"detail": "the dataset has no column roles; set them on the Dataset tab first"}
        client.put(f"/api/projects/{project}/datasets/{dataset}/roles", json={"columns": ROLES})
        run = client.post(url, json={"configuration_id": configuration})
        assert run.status_code == 201, run.text
        connection = client.app.state.jbg["store"].get(RunReference(run.json()["run_id"])).spec[
            "configuration"]["connection"]
        assert (connection["id_column"], connection["class_column"]) == ("id", "species")


# --- Compatibility of prediction datasets ----------------------------------------------------------

def test_csv_prediction_uploads_must_be_compatible(tmp_path):
    with _client(tmp_path) as client:
        project, _ = _csv_project(client)
        no_class = b"id;sepal;petal;note\n9;5.0;1.3;q\n"
        assert _upload(client, project, no_class, "Nov", "prediction").status_code == 201
        before = _stored_files(tmp_path, project)
        missing = _upload(client, project, b"id;sepal;note\n9;5.0;q\n", "Dec", "prediction")
        assert missing.status_code == 409
        assert missing.json()["detail"]["missing_columns"] == ["petal"]
        assert _stored_files(tmp_path, project) == before           # the file was removed
        wrong_kind = _upload(client, project, b"id;sepal;petal;note\n9;high;1.3;q\n", "Jan", "prediction")
        assert wrong_kind.json()["detail"]["different_kind"] == ["sepal"]


def test_validation_of_a_prediction_dataset_checks_compatibility(tmp_path):
    with _client(tmp_path) as client:
        project, training = _csv_project(client, roles=None)
        prediction = _upload(client, project, b"id;sepal;note\n9;5.0;q\n", "Nov", "prediction").json()["dataset_id"]
        # Registered before the training roles existed; validation finds it.
        client.put(f"/api/projects/{project}/datasets/{training}/roles", json={"columns": ROLES})
        result = client.post(f"/api/projects/{project}/datasets/{prediction}/validate").json()
        assert [(c["code"], c["status"]) for c in result["checks"]][3] == ("compatible_structure", "failed")
        assert result["validation_state"] == "invalid" and result["compatibility_checked"]


class FakeLayer:
    def __init__(self, tables):
        self.tables, self.table = tables, None

    def get_table_columns(self, catalog, table):
        return dict(self.tables.get(table, {}))

    def count_data_rows(self, catalog, table):
        return 3

    def get_catalogs_as_options(self):
        return ["", "Db", "Other"]

    def get_tables_as_options(self):
        return ["", "dbo.Train", "dbo.Nov"]


def _sql_layers(client, tables):
    client.app.state.jbg["sessions"].data_layer_for = (
        lambda principal, config, logger: FakeLayer(tables))


def test_sql_prediction_datasets_must_be_compatible(tmp_path):
    train = {"id": "int", "sepal": "float", "petal": "float", "note": "nvarchar", "species": "nvarchar"}
    with _client(tmp_path) as client:
        project = client.post("/api/projects", json={"name": "S", "source": "sql"}).json()["project_id"]
        url = f"/api/projects/{project}/datasets"
        training = client.post(url, json={"name": "T", "role": "training", "catalog": "Db",
                                          "table": "dbo.Train"}).json()["dataset_id"]
        client.post(f"{url}/{training}/activate")
        _sql_layers(client, {"dbo.Train": train, "dbo.Nov": {k: v for k, v in train.items()
                                                              if k != "species"},
                             "dbo.Bad": {"id": "int", "sepal": "nvarchar"}})
        assert client.put(f"{url}/{training}/roles", json={"columns": ROLES}).status_code == 200
        good = client.post(url, json={"name": "Nov", "role": "prediction", "catalog": "Db",
                                      "table": "dbo.Nov"})
        assert good.status_code == 201, good.text
        bad = client.post(url, json={"name": "Bad", "role": "prediction", "catalog": "Db",
                                     "table": "dbo.Bad"})
        assert bad.status_code == 409 and set(bad.json()["detail"]["missing_columns"]) == {"petal", "note"}


def test_sql_prediction_registration_needs_the_sign_in(tmp_path):
    with _client(tmp_path) as client:
        project = client.post("/api/projects", json={"name": "S", "source": "sql"}).json()["project_id"]
        url = f"/api/projects/{project}/datasets"
        training = client.post(url, json={"name": "T", "role": "training", "catalog": "Db",
                                          "table": "dbo.Train"}).json()["dataset_id"]
        client.post(f"{url}/{training}/activate")
        from jbg_test_roles import store_roles
        store_roles(client, project, training, {"connection": {
            "id_column": "id", "class_column": "species", "data_numerical_columns": ["sepal"]}})
        response = client.post(url, json={"name": "Nov", "role": "prediction", "catalog": "Db",
                                          "table": "dbo.Nov"})
        assert (response.status_code, response.json()) == (409, {"detail": "SQL sign-in required"})


# --- SQL catalog and table lists ------------------------------------------------------------------

def test_catalog_and_table_lists(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/api/sql/catalogs").json() == {"detail": "SQL sign-in required"}
        _sql_layers(client, {})
        assert client.get("/api/sql/catalogs").json() == {"catalogs": ["Db", "Other"], "default": "Db"}
        assert client.get("/api/sql/tables", params={"catalog": "Db"}).json() == {
            "catalog": "Db", "tables": ["dbo.Train", "dbo.Nov"]}
        unsafe = client.get("/api/sql/tables", params={"catalog": "Db'; DROP"})
        assert unsafe.status_code == 422


# --- Dataset tab messages ----------------------------------------------------------------------

def test_the_dataset_tab_explains_refusals(tmp_path):
    import re
    with _client(tmp_path) as client:
        project, _ = _csv_project(client)
        client.post(f"/ui/projects/{project}/open", follow_redirects=False)
        response = client.post("/ui/dataset/csv", data={"name": "Dec", "role": "prediction"},
                               files={"file": ("d.csv", b"id;sepal;note\n9;5.0;q\n", "text/csv")},
                               follow_redirects=True)
        assert "Datasetet är inte kompatibelt med träningsdatasetet." in response.text
        sql = client.post("/api/projects", json={"name": "S", "source": "sql"}).json()["project_id"]
        client.post(f"/ui/projects/{sql}/open", follow_redirects=False)
        training = client.post(f"/api/projects/{sql}/datasets", json={
            "name": "T", "role": "training", "catalog": "Db", "table": "dbo.Train"}).json()["dataset_id"]
        client.post(f"/api/projects/{sql}/datasets/{training}/activate")
        from jbg_test_roles import store_roles
        store_roles(client, sql, training, {"connection": {
            "id_column": "id", "class_column": "species", "data_numerical_columns": ["sepal"]}})
        page = client.post("/ui/dataset/sql", data={"name": "Nov", "role": "prediction",
                                                    "catalog": "Db", "table": "dbo.Nov"},
                           follow_redirects=True).text
        assert "Logga in på SQL Server först." in page
