"""286b: dataset service and endpoints, CSV upload into project-ID storage."""
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")

from fastapi.testclient import TestClient  # noqa: E402

import JBGDatasetService  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGProjectStorage import detect_delimiter, project_csv_storage  # noqa: E402

MISSING = "00000000-0000-4000-8000-000000000000"
IRIS = ("id;sepal;petal;species\n1;5.1;1.4;setosa\n2;7.0;4.7;versicolor\n"
        "3;6.3;6.0;virginica\n4;4.9;1.4;setosa\n").encode("utf-8")


def _client(tmp_path, subject="developer"):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_artifacts=tmp_path / "csv" / "objects",
        csv_root=tmp_path / "csv",
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _project(client, name, source):
    response = client.post("/api/projects", json={"name": name, "source": source})
    assert response.status_code == 201, response.text
    return response.json()["project_id"]


def _upload(client, project_id, name="Iris", role="training", data=IRIS,
            filename="iris.csv", **form):
    return client.post(f"/api/projects/{project_id}/datasets/csv",
                       data={"name": name, "role": role, **form},
                       files={"file": (filename, data, "text/csv")})


def _stored(tmp_path, project_id):
    from Storage.CSVDatasetCatalog import list_csv_dataset_labels
    repository, scope = project_csv_storage(project_id, root=tmp_path / "csv")
    return list_csv_dataset_labels(repository, scope)


# --- SQL datasets ----------------------------------------------------------------------

def test_sql_datasets_are_registered_and_one_is_active_for_training(tmp_path):
    with _client(tmp_path) as client:
        project = _project(client, "Ärenden", "sql")
        url = f"/api/projects/{project}/datasets"
        train = client.post(url, json={"name": "Ärenden 2025", "role": "training",
                                       "catalog": "ExampleDb", "table": "dbo.Cases2025"})
        assert train.status_code == 201
        body = train.json()
        assert (body["source"], body["validation_state"], body["active_training"]) == (
            "sql", "unverified", False)
        assert (body["catalog"], body["table"]) == ("ExampleDb", "dbo.Cases2025")
        mixed = client.post(url, json={"name": "Mixed", "role": "mixed",
                                       "catalog": "ExampleDb", "table": "dbo.All"}).json()
        predict = client.post(url, json={"name": "Nov", "role": "prediction",
                                         "catalog": "ExampleDb", "table": "dbo.Nov"}).json()
        client.post(f"{url}/{body['dataset_id']}/activate")
        client.post(f"{url}/{mixed['dataset_id']}/activate")            # switches atomically
        active = [d["name"] for d in client.get(url).json() if d["active_training"]]
        assert active == ["Mixed"]
        refused = client.post(f"{url}/{predict['dataset_id']}/activate")
        assert refused.status_code == 422 and "prediction-only" in refused.json()["detail"]
        placeholder = client.post(url, json={"name": "Bad", "role": "training",
                                             "catalog": "N/A", "table": "dbo.X"})
        assert placeholder.status_code == 422


def test_source_type_of_the_project_decides(tmp_path):
    with _client(tmp_path) as client:
        csv_project, sql_project = _project(client, "CSV", "csv"), _project(client, "SQL", "sql")
        sql_into_csv = client.post(f"/api/projects/{csv_project}/datasets", json={
            "name": "X", "role": "training", "catalog": "Db", "table": "dbo.T"})
        assert (sql_into_csv.status_code, sql_into_csv.json()) == (
            409, {"detail": "this is a CSV project"})
        csv_into_sql = _upload(client, sql_project)
        assert (csv_into_sql.status_code, csv_into_sql.json()) == (
            409, {"detail": "this is a SQL project"})


# --- CSV upload ------------------------------------------------------------------------

def test_csv_upload_is_stored_under_the_project_id(tmp_path):
    with _client(tmp_path) as client:
        project = _project(client, "Iris", "csv")
        response = _upload(client, project)
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["upload"] == {"rows": 4, "columns": ["id", "sepal", "petal", "species"],
                                  "delimiter": ";", "encoding": "utf-8-sig"}
        assert "registration_id" not in body and "catalog" not in body
        client.patch(f"/api/projects/{project}", json={"name": "Iris renamed"})
    labels = _stored(tmp_path, project)
    assert len(labels) == 1 and "iris.csv" in labels[0][0]
    # Voila's name-based storage for the same name sees nothing.
    from Storage.FileArtifactRepository import FileArtifactRepository
    from Storage.LocalCSVScope import local_csv_scope
    from Storage.CSVDatasetCatalog import list_csv_dataset_labels
    voila_scope = local_csv_scope(tmp_path / "csv", "Iris")
    assert list_csv_dataset_labels(FileArtifactRepository(tmp_path / "csv" / "objects"),
                                   voila_scope) == ()


def test_bad_or_oversized_uploads_leave_nothing_behind(tmp_path, monkeypatch):
    with _client(tmp_path) as client:
        project = _project(client, "Iris", "csv")
        ragged = _upload(client, project, data=b"a,b\n1,2,3\n")
        assert (ragged.status_code, ragged.json()) == (422, {"detail": "Inconsistent CSV row shape"})
        monkeypatch.setattr(JBGDatasetService, "MAX_UPLOAD_BYTES", 50)
        large = _upload(client, project)
        assert (large.status_code, large.json()) == (413, {"detail": "CSV file exceeds 25 MB"})
        monkeypatch.undo()
        assert _upload(client, project, name="Iris").status_code == 201
        duplicate = _upload(client, project, name="IRIS")
        assert (duplicate.status_code, duplicate.json()) == (
            409, {"detail": "dataset name already exists"})
    assert len(_stored(tmp_path, project)) == 1                  # the duplicate was erased


def test_client_file_names_are_display_text_only(tmp_path):
    with _client(tmp_path) as client:
        project = _project(client, "Iris", "csv")
        assert _upload(client, project, filename="..\\..\\secret/iris.csv").status_code == 201
    label = _stored(tmp_path, project)[0][0]
    assert "/" not in label and "\\" not in label and ".." not in label
    assert not (tmp_path.parent / "secret").exists()


def test_explicit_format_overrides_detection(tmp_path):
    with _client(tmp_path) as client:
        project = _project(client, "Iris", "csv")
        data = IRIS.replace(b";", b"|")
        assert _upload(client, project, data=data, delimiter="|").json()["upload"]["delimiter"] == "|"
        bad = _upload(client, project, name="Other", delimiter=":")
        assert bad.status_code == 422


# --- Ownership -----------------------------------------------------------------------------

def test_other_users_cannot_see_or_add_to_my_project(tmp_path):
    with _client(tmp_path) as mine:
        project = _project(mine, "Mine", "csv")
        dataset = _upload(mine, project).json()["dataset_id"]
    with _client(tmp_path, subject="bob") as theirs:
        for response in (theirs.get(f"/api/projects/{project}/datasets"),
                         theirs.get(f"/api/projects/{project}/datasets/{dataset}"),
                         theirs.post(f"/api/projects/{project}/datasets/{dataset}/activate"),
                         _upload(theirs, project, name="Bob's")):
            assert (response.status_code, response.json()) == (404, {"detail": "project unavailable"})
    with _client(tmp_path) as mine:
        assert mine.get(f"/api/projects/{project}/datasets/{MISSING}").status_code == 404
        assert mine.delete(f"/api/projects/{project}/datasets/{dataset}").status_code == 405
    assert len(_stored(tmp_path, project)) == 1


# --- Helpers ---------------------------------------------------------------------------------

def test_delimiter_detection_and_project_storage_identity(tmp_path):
    assert detect_delimiter(b"a;b;c\n1;2;3\n") == ";"
    assert detect_delimiter(b"a\tb\n1\t2\n") == "\t"
    assert detect_delimiter(b"single\nvalue\n") == ","
    project = "3f1c2b8e-0d4a-4e7f-9a51-2c6d8e9b0a17"
    _, first = project_csv_storage(project, root=tmp_path)
    _, again = project_csv_storage(project, root=tmp_path)
    assert first == again and first.project_id == project
    for bad in ("not-a-uuid", MISSING.replace("4000", "1000"), None):
        with pytest.raises(ValueError):
            project_csv_storage(bad, root=tmp_path)
