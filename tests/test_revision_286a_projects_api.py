"""286a: project service and endpoints on the 285i repository."""
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings, open_project_repository  # noqa: E402
from JBGProjectService import ProjectConflict, ProjectService, ProjectUnavailable  # noqa: E402

MISSING = "00000000-0000-4000-8000-000000000000"


@pytest.fixture
def service(tmp_path):
    repository = open_project_repository(
        MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), create_tables=True)
    yield ProjectService(repository)
    repository.close()


def _client(tmp_path, subject="developer"):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_artifacts=tmp_path / "artifacts",
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


# --- Service ------------------------------------------------------------------------------

def test_projects_are_created_listed_and_renamed(service):
    created = service.create("me", name="Ärenden", source="sql", description="Fem ärendetyper")
    assert set(created) == {"project_id", "name", "description", "source",
                            "created_at", "updated_at"}            # never the owner
    assert (created["name"], created["source"]) == ("Ärenden", "sql")
    renamed = service.update("me", created["project_id"], name="Ärenden 2026")
    assert renamed["project_id"] == created["project_id"]          # the stable identity
    assert renamed["name"] == "Ärenden 2026" and renamed["source"] == "sql"
    assert [p["project_id"] for p in service.list("me")] == [created["project_id"]]


def test_names_are_unique_per_owner_ignoring_case(service):
    service.create("me", name="Iris", source="csv")
    with pytest.raises(ProjectConflict):
        service.create("me", name="iris", source="sql")
    other = service.create("me", name="Other", source="csv")
    with pytest.raises(ProjectConflict):
        service.update("me", other["project_id"], name="IRIS")
    assert service.create("someone-else", name="Iris", source="csv")["name"] == "Iris"


def test_unknown_and_foreign_projects_look_the_same(service):
    mine = service.create("me", name="Mine", source="csv")
    for owner, project_id in (("someone-else", mine["project_id"]), ("me", MISSING),
                              ("me", "")):
        with pytest.raises(ProjectUnavailable):
            service.get(owner, project_id)
        with pytest.raises(ProjectUnavailable):
            service.update(owner, project_id, name="Taken over")
    assert service.list("someone-else") == []


def test_invalid_input_is_refused(service):
    with pytest.raises(ValueError):
        service.create("me", name="X", source="file")
    with pytest.raises(ValueError):
        service.create("me", name="   ", source="sql")
    with pytest.raises(ValueError):
        service.create("me", name="x" * 201, source="sql")


# --- HTTP API -------------------------------------------------------------------------------

def test_project_endpoints(tmp_path):
    with _client(tmp_path) as client:
        created = client.post("/api/projects", json={"name": "Iris", "source": "csv",
                                                     "description": "Demo"})
        assert created.status_code == 201
        project_id = created.json()["project_id"]
        assert client.get("/api/projects").json()[0]["project_id"] == project_id
        assert client.get(f"/api/projects/{project_id}").json()["name"] == "Iris"
        patched = client.patch(f"/api/projects/{project_id}",
                               json={"name": "Iris 2", "source": "sql"})
        assert patched.status_code == 200
        assert (patched.json()["name"], patched.json()["source"]) == ("Iris 2", "csv")
        duplicate = client.post("/api/projects", json={"name": "IRIS 2", "source": "sql"})
        assert (duplicate.status_code, duplicate.json()) == (
            409, {"detail": "project name already exists"})


def test_project_endpoints_refuse_bad_input_without_echo(tmp_path):
    with _client(tmp_path) as client:
        for body in ({"name": "X", "source": "file"}, {"name": "X"},
                     {"name": ["secret-ish"], "source": "sql"}):
            response = client.post("/api/projects", json=body)
            assert (response.status_code, response.json()) == (422, {"detail": "invalid request"})
        empty = client.post("/api/projects", json={"name": "  ", "source": "sql"})
        assert empty.status_code == 422 and "characters" in empty.json()["detail"]
        assert client.get(f"/api/projects/{MISSING}").status_code == 404
        assert client.delete(f"/api/projects/{MISSING}").status_code == 405   # no deletion yet


def test_other_users_cannot_see_or_change_my_project(tmp_path):
    with _client(tmp_path) as mine:
        project_id = mine.post("/api/projects", json={"name": "Mine", "source": "sql"}
                               ).json()["project_id"]
    with _client(tmp_path, subject="bob") as theirs:
        assert theirs.get("/api/projects").json() == []
        for response in (theirs.get(f"/api/projects/{project_id}"),
                         theirs.patch(f"/api/projects/{project_id}", json={"name": "Bob's"})):
            assert (response.status_code, response.json()) == (404, {"detail": "project unavailable"})
    with _client(tmp_path) as mine:
        assert mine.get(f"/api/projects/{project_id}").json()["name"] == "Mine"
