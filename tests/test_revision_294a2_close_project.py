"""294a2: close the open project; switching forgets the selected configuration."""
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


def _client(tmp_path):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _project(client, name):
    location = client.post("/ui/projects", data={"name": name, "source": "csv"},
                           follow_redirects=False).headers["location"]
    return re.search(r"selected=([0-9a-f-]{36})", location).group(1)


def _open(client, project):
    return client.post(f"/ui/projects/{project}/open", follow_redirects=False)


def _select_configuration(client):
    created = client.post("/ui/configuration", data={"name": "C"}, follow_redirects=False)
    return re.search(r"configuration=([0-9a-f-]{36})", created.headers["location"]).group(1)


def _deleted(response, cookie):
    return any(h.startswith(f"{cookie}=") and "Max-Age=0" in h
               for h in response.headers.get_list("set-cookie"))


def test_closing_forgets_project_and_configuration_on_any_tab(tmp_path):
    with _client(tmp_path) as client:
        project = _project(client, "A")
        _open(client, project)
        _select_configuration(client)
        page = client.get("/ui/run").text
        assert 'action="/ui/close-project"' in page                       # the chip's close button
        closed = client.post("/ui/close-project", data={"back": "/ui/run"}, follow_redirects=False)
        assert closed.headers["location"] == "/ui/run?message=closed"
        assert _deleted(closed, "jbg_project") and _deleted(closed, "jbg_configuration")
        page = client.get(closed.headers["location"]).text
        assert "Projektet är stängt." in page and "Inget valt" in page
        assert 'action="/ui/close-project"' not in page                   # nothing left to close


def test_the_projects_tab_offers_close_for_the_open_project(tmp_path):
    with _client(tmp_path) as client:
        project = _project(client, "A")
        other = _project(client, "B")
        _open(client, project)
        page = client.get(f"/ui/projects?selected={project}").text
        assert re.search(r'<form method="post" action="/ui/close-project">\s*<button class="btn sub">Stäng projekt', page)
        page = client.get(f"/ui/projects?selected={other}").text
        assert '<button class="btn act ready">Öppna projekt</button>' in page


def test_switching_project_forgets_the_configuration_reopening_keeps_it(tmp_path):
    with _client(tmp_path) as client:
        first, second = _project(client, "A"), _project(client, "B")
        _open(client, first)
        _select_configuration(client)
        assert not _deleted(_open(client, first), "jbg_configuration")    # same project
        assert _deleted(_open(client, second), "jbg_configuration")       # another project


@pytest.mark.parametrize("back", ["https://evil.example/", "//evil.example/ui/", "/docs"])
def test_closing_never_redirects_elsewhere(tmp_path, back):
    with _client(tmp_path) as client:
        response = client.post("/ui/close-project", data={"back": back}, follow_redirects=False)
        assert response.headers["location"] == "/ui/projects?message=closed"


def test_closing_from_another_site_is_refused(tmp_path):
    with _client(tmp_path) as client:
        _open(client, _project(client, "A"))
        response = client.post("/ui/close-project", headers={"origin": "http://evil.example"})
        assert response.status_code == 403


def test_the_new_texts_are_translated():
    import JBGWebUI
    assert {"Close project", "Project closed."} <= set(JBGWebUI._CATALOGS["sv"])
