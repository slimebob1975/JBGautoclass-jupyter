"""295b2: choosing a configuration for runs without saving it."""
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

CSV = b"id;sepal;species\n1;5.1;a\n2;7.0;b\n3;6.3;a\n4;4.9;b\n"


def _client(tmp_path):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs", sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _project_with_older_configuration(client):
    """An open project with a configuration that was created elsewhere (API),
    so it is not chosen yet: the situation from 2026-11-09."""
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "D", "role": "training"},
                          files={"file": ("d.csv", CSV, "text/csv")}).json()["dataset_id"]
    client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
    client.put(f"/api/projects/{project}/datasets/{dataset}/roles",
               json={"columns": {"id": "id", "sepal": "numeric", "species": "class"}})
    configuration = client.post(f"/api/projects/{project}/configurations",
                                json={"name": "From Voila"}).json()["configuration_id"]
    return project, configuration


def test_opening_a_configuration_offers_choosing_it(tmp_path):
    with _client(tmp_path) as client:
        _, configuration = _project_with_older_configuration(client)
        page = client.get(f"/ui/configuration?configuration={configuration}").text
        assert f'action="/ui/configuration/{configuration}/select"' in page
        assert "Välj för körning" in page and "Vald för körning" not in page
        assert "Ingen konfiguration är vald" in client.get("/ui/run").text      # only opened


def test_choosing_does_not_create_a_version_and_unblocks_the_run(tmp_path):
    with _client(tmp_path) as client:
        project, configuration = _project_with_older_configuration(client)
        chosen = client.post(f"/ui/configuration/{configuration}/select", follow_redirects=False)
        assert any(h.startswith(f"jbg_configuration={configuration}")
                   for h in chosen.headers.get_list("set-cookie"))
        page = client.get(chosen.headers["location"]).text
        assert "Konfigurationen är vald." in page and "Vald för körning" in page
        assert f'action="/ui/configuration/{configuration}/select"' not in page
        assert re.search(r"<b>From Voila</b></a>\s*<span class=\"badge b-info\">Vald</span>", page)
        assert client.get(f"/api/projects/{project}/configurations/{configuration}").json()["version"] == 1
        run = client.get("/ui/run").text
        assert "En konfiguration är vald." in run and "Konfigurationen tränar en modell." in run
        assert '<button class="btn act ready">Starta körning</button>' in run


def test_the_new_texts_are_translated():
    import JBGWebUI
    assert {"Choose for runs", "Chosen for runs"} <= set(JBGWebUI._CATALOGS["sv"])
