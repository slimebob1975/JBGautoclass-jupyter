"""298d (noted 2026-11-11 on an existing configuration): jbg.js and jbg.css
carry a content version in their address, so browsers never keep an old
script (the 298c formula and target classes did not react); only the lists
of sampling methods are shown (the single, older fields are server-owned,
the engine's fallback); the group column is a choice among the active
training dataset's features, and new configurations have none."""
import hashlib
import re
from datetime import timedelta

import pytest

from JBGWebUIConfigurations import effective_sampling

# --- Helpers -----------------------------------------------------------------------------------

def test_an_empty_sampling_list_shows_the_fallback():
    stored = {"mode": {"oversampler": "NOG", "undersampler": "RUS"}}
    shown = effective_sampling({"mode": {"oversamplers": None, "undersamplers": ["NUG", "TOM"]}}, stored)
    assert shown["mode"]["oversamplers"] == ["NOG"]
    assert shown["mode"]["undersamplers"] == ["NUG", "TOM"]                 # a choice stands
    assert effective_sampling({"mode": {"oversamplers": []}}, None)["mode"]["oversamplers"] == []


def test_static_addresses_carry_the_content_hash():
    from JBGWebUI import WEB, static_version
    for name in ("jbg.js", "jbg.css"):
        content = (WEB / "static" / name).read_bytes()
        assert static_version(name) == hashlib.sha256(content).hexdigest()[:12]
    assert static_version("missing.js") == ""


def test_the_engine_default_is_unchanged():
    from Config import Config
    assert Config().mode.validation_group_column == "iafId"               # Voila unchanged


# --- Through the API and the tab ----------------------------------------------------------------

CSV = "id;x;grp;label\n1;1.0;a;katt\n2;2.0;a;hund\n3;3.0;b;katt\n4;4.0;b;älg\n".encode()


def _client(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("python_multipart")
    pytest.importorskip("jinja2")
    from fastapi.testclient import TestClient
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    from JBGSQLServer import SQLServerSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _project(client, roles=True, mode=None):
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "D", "role": "training"},
                          files={"file": ("d.csv", CSV, "text/csv")}).json()["dataset_id"]
    client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
    if roles:
        client.put(f"/api/projects/{project}/datasets/{dataset}/roles", json={"columns": {
            "id": "id", "x": "numeric", "grp": "categorical", "label": "class"}})
    configuration = client.post(f"/api/projects/{project}/configurations", json={
        "name": "C", "settings": {"mode": mode or {}}}).json()
    return project, configuration


def _page(client, configuration):
    return client.get(f"/ui/configuration?configuration={configuration['configuration_id']}").text


def test_pages_load_versioned_static_files(tmp_path):
    from JBGWebUI import static_version
    with _client(tmp_path) as client:
        page = client.get("/ui/projects").text
    assert f'href="/ui/static/jbg.css?v={static_version("jbg.css")}"' in page
    assert f'src="/ui/static/jbg.js?v={static_version("jbg.js")}"' in page


def test_new_configurations_have_no_group_column(tmp_path):
    # 298g: server-owned since the group column became a dataset column role
    # (the 298d choice among the features was replaced).
    with _client(tmp_path) as client:
        project, configuration = _project(client)
        stored = client.app.state.jbg["configuration_service"].stored_settings(
            *_owner_and_ids(project, configuration))
    assert "validation_group_column" not in configuration["settings"]["mode"]
    assert stored["mode"]["validation_group_column"] == ""


def test_only_the_sampling_lists_are_shown_with_the_fallback_ticked(tmp_path):
    with _client(tmp_path) as client:
        _, configuration = _project(client)
        page = _page(client, configuration)
        schema = client.get("/api/config/schema", params={"source": "csv"}).json()
    keys = {field["key"] for field in schema["fields"]}
    assert "mode.oversampler" not in keys and "mode.undersampler" not in keys
    assert 'name="f:mode.oversampler"' not in page and 'name="f:mode.undersampler"' not in page
    assert re.search(r'name="f:mode.oversamplers" value="NOG" checked', page)
    assert re.search(r'name="f:mode.undersamplers" value="NUG" checked', page)


def test_the_single_sampling_fields_are_server_owned(tmp_path):
    with _client(tmp_path) as client:
        project, configuration = _project(client)
        url = f"/api/projects/{project}/configurations/{configuration['configuration_id']}"
        saved = client.put(url, json={"settings": {"mode": {"oversampler": "SMO"}},
                                      "expected_version": 1}).json()
        stored = client.app.state.jbg["configuration_service"].stored_settings(
            *_owner_and_ids(project, configuration))
    assert "oversampler" not in saved["settings"]["mode"]
    assert stored["mode"]["oversampler"] == "NOG"                          # unchanged


def _owner_and_ids(project, configuration):
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
    return owner, project, configuration["configuration_id"]
