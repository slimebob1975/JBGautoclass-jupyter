"""286c: configuration service and endpoints with server-owned fields."""
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from JBGConfigurationService import SERVER_OWNED  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402

MISSING = "00000000-0000-4000-8000-000000000000"


def _client(tmp_path, subject="developer"):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_artifacts=tmp_path / "csv" / "objects",
        csv_root=tmp_path / "csv",
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _project(client, name="Iris", source="sql"):
    return client.post("/api/projects", json={"name": name, "source": source}).json()["project_id"]


def _owner(client):
    from JBGIdentity import AuthenticationMode, AuthenticatedPrincipal
    return AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key


# 293c: column roles belong to the dataset; the roles here must be ignored.
ROLES = {"connection": {"class_column": "species", "id_column": "id",
                        "data_numerical_columns": ["sepal", "petal"]},
         "mode": {"max_iterations": 777},
         "debug": {"data_limit": 100}}


def test_created_configuration_shows_only_user_settings(tmp_path):
    with _client(tmp_path) as client:
        project = _project(client)
        response = client.post(f"/api/projects/{project}/configurations",
                               json={"name": "Standard", "settings": ROLES})
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["version"] == 1 and body["name"] == "Standard"
        settings = body["settings"]
        assert set(settings) == {"connection", "mode", "io", "debug", "mail"}
        assert settings["mode"]["max_iterations"] == 777
        assert "class_column" not in settings["connection"]            # 293c: dataset-owned
        assert settings["debug"]["data_limit"] == 100
        for section, keys in SERVER_OWNED.items():
            assert not keys & set(settings[section]), section


def test_server_owned_fields_are_ignored(tmp_path):
    hostile = {"connection": {"host": "evil.example.invalid", "sql_password": "S3cret!",
                              "data_table": "dbo.Other", "class_column": "species"},
               "io": {"model_path": "../../outside", "model_name": "x", "verbose": False},
               "mail": {"smtp_server": "smtp.evil.example.invalid"},
               "data_source": {"kind": "csv"}, "name": "renamed", "schema_version": 1}
    with _client(tmp_path) as client:
        project = _project(client)
        created = client.post(f"/api/projects/{project}/configurations",
                              json={"name": "C", "settings": hostile})
        assert created.status_code == 201, created.text
        assert "S3cret!" not in created.text and "evil" not in created.text
        stored = client.app.state.jbg["configuration_service"].stored_settings(
            _owner(client), project, created.json()["configuration_id"])
    assert stored["connection"]["host"] != "evil.example.invalid"
    assert stored["connection"]["sql_password"] == "" and stored["connection"]["sql_username"] == ""
    assert stored["connection"]["data_table"] != "dbo.Other"
    assert stored["io"]["model_path"] != "../../outside" and stored["io"]["verbose"] is False
    assert stored["mail"]["smtp_server"] != "smtp.evil.example.invalid"
    assert stored["data_source"]["kind"] == "sql" and stored["name"] == project
    assert stored["connection"]["class_column"] != "species"           # 293c: ignored


def test_invalid_settings_are_refused_without_echo(tmp_path):
    with _client(tmp_path) as client:
        project = _project(client)
        url = f"/api/projects/{project}/configurations"
        cases = [({"mode": {"no_such_setting": 1}}, "unknown configuration setting"),
                 ({"extras": {}}, "unknown configuration section"),
                 ({"mode": "train"}, "configuration sections must be JSON objects"),
                 ({"debug": {"data_limit": "lots-of-rows-XYZ"}}, "configuration is invalid")]
        for index, (settings, message) in enumerate(cases):
            response = client.post(url, json={"name": f"C{index}", "settings": settings})
            assert (response.status_code, response.json()) == (422, {"detail": message})
            assert "XYZ" not in response.text
        assert client.get(url).json() == []                     # nothing was stored


def test_saving_needs_the_current_version(tmp_path):
    with _client(tmp_path) as client:
        project = _project(client)
        url = f"/api/projects/{project}/configurations"
        created = client.post(url, json={"name": "C", "settings": ROLES}).json()
        item = f"{url}/{created['configuration_id']}"
        saved = client.put(item, json={"settings": {"debug": {"data_limit": 500}},
                                       "expected_version": 1})
        assert saved.status_code == 200 and saved.json()["version"] == 2
        assert saved.json()["settings"]["debug"]["data_limit"] == 500
        assert saved.json()["settings"]["mode"]["max_iterations"] == 777            # kept
        stale = client.put(item, json={"settings": {"debug": {"data_limit": 1}},
                                       "expected_version": 1})
        assert (stale.status_code, stale.json()) == (
            409, {"detail": "configuration changed; reload before saving"})
        assert client.get(item, params={"version": 1}).json()["settings"]["debug"]["data_limit"] == 100
        assert client.get(item).json()["version"] == 2
        assert client.get(item, params={"version": 9}).status_code == 404


def test_csv_configurations_are_stored_without_a_dataset(tmp_path):
    with _client(tmp_path) as client:
        project = _project(client, source="csv")
        created = client.post(f"/api/projects/{project}/configurations",
                              json={"name": "C", "settings": ROLES})
        assert created.status_code == 201, created.text
        stored = client.app.state.jbg["configuration_service"].stored_settings(
            _owner(client), project, created.json()["configuration_id"])
    assert stored["data_source"]["kind"] == "csv"
    assert stored["data_source"]["artifact_id"] == "" and stored["data_source"]["dataset_id"] == ""


def test_names_and_ownership(tmp_path):
    with _client(tmp_path) as mine:
        project = _project(mine)
        url = f"/api/projects/{project}/configurations"
        configuration = mine.post(url, json={"name": "Standard"}).json()["configuration_id"]
        duplicate = mine.post(url, json={"name": "STANDARD"})
        assert (duplicate.status_code, duplicate.json()) == (
            409, {"detail": "configuration name already exists"})
        assert mine.get(f"{url}/{MISSING}").status_code == 404
        assert mine.delete(f"{url}/{configuration}").status_code == 405
    with _client(tmp_path, subject="bob") as theirs:
        for response in (theirs.get(url), theirs.get(f"{url}/{configuration}"),
                         theirs.post(url, json={"name": "Bob's"}),
                         theirs.put(f"{url}/{configuration}",
                                    json={"settings": {}, "expected_version": 1})):
            assert (response.status_code, response.json()) == (404, {"detail": "project unavailable"})
