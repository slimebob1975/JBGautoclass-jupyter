"""293b2: roles suggested from a configuration, validation above the views,
a complete classes table and decimal commas in Swedish."""
import re
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from JBGDatasetRoles import roles_from_configuration  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402

CSV = b"id;sepal;note;species\n1;5.1;x;a\n2;7.0;y;b\n3;6.3;x;a\n4;4.9;y;\n5;5.0;x;a\n"
OLD = {"connection": {"id_column": "id", "class_column": "species",
                      "data_numerical_columns": ["sepal", "gone"], "data_text_columns": ["note"]},
       "mode": {"category_text_columns": ["note"]}}


def _client(tmp_path):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _setup(client, data=CSV):
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    uploaded = client.post("/ui/dataset/csv", data={"name": "D", "role": "training"},
                           files={"file": ("d.csv", data, "text/csv")}, follow_redirects=False)
    dataset = re.search(r"dataset=([0-9a-f-]{36})", uploaded.headers["location"]).group(1)
    client.post(f"/ui/dataset/{dataset}/activate", follow_redirects=False)
    return project, dataset


def _store_old_configuration(client, project, name, roles_settings):
    """A configuration as stored before 293c, roles included (the service no
    longer accepts roles, so write it directly into the repository)."""
    from JBGConfigurationService import _server_template
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    settings = _server_template()
    settings["data_source"]["kind"] = "csv"
    settings["connection"].update(roles_settings["connection"])
    settings["mode"].update(roles_settings.get("mode", {}))
    owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
    repository = client.app.state.jbg["configuration_service"]._repository
    repository.create(owner, project, name, settings)


def _selected(page, column):
    match = re.search(rf'name="role:{column}"[^>]*>(.*?)</select>', page, re.S)
    return re.search(r'<option value="(\w+)" selected', match.group(1)).group(1)


# --- 1. Roles suggested from an older configuration ------------------------------------------

def test_conversion_keeps_only_existing_columns():
    assert roles_from_configuration(OLD, ["id", "sepal", "note", "species"]) == {
        "sepal": "numeric", "note": "categorical", "species": "class", "id": "id"}


def test_a_dataset_without_roles_gets_a_suggestion_that_is_not_saved(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _setup(client)
        _store_old_configuration(client, project, "From Voila", OLD)
        page = client.get(f"/ui/dataset?dataset={dataset}&view=roles").text
        assert "Förslag från konfigurationen “From Voila”." in page
        assert (_selected(page, "species"), _selected(page, "note")) == ("class", "categorical")
        url = f"/api/projects/{project}/datasets/{dataset}/roles"
        assert client.get(url).json()["columns"] is None                     # nothing saved
        client.put(url, json={"columns": {"id": "id", "sepal": "numeric", "species": "class"}})
        page = client.get(f"/ui/dataset?dataset={dataset}&view=roles").text
        assert "Förslag från" not in page and _selected(page, "note") == "ignored"


def test_without_a_configuration_with_roles_nothing_is_suggested(tmp_path):
    with _client(tmp_path) as client:
        _, dataset = _setup(client)
        page = client.get(f"/ui/dataset?dataset={dataset}&view=roles").text
        assert "Förslag från" not in page and "Inga kolumnroller ännu." in page


# --- 2. Validation above the views ---------------------------------------------------------------

def test_the_views_stay_available_after_validation(tmp_path):
    with _client(tmp_path) as client:
        _, dataset = _setup(client)
        page = client.post(f"/ui/dataset/{dataset}/validate").text
        assert "Datakällan kan nås." in page
        assert page.count('class="subtabs"') == 1 and "Förhandsgranskning" in page
        assert page.index("Datakällan kan nås.") < page.index('class="subtabs"')


# --- 3. A complete classes table, decimal commas -----------------------------------------------

def _rows(page):
    body = re.search(r'<table class="grid classes">.*?<tbody>(.*?)</tbody>', page, re.S).group(1)
    return [[re.sub(r"<[^>]+>", "", cell).strip() for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
            for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S)]


def test_unknown_and_total_rows(tmp_path):
    with _client(tmp_path) as client:
        _, dataset = _setup(client)
        page = client.get(f"/ui/dataset?dataset={dataset}&view=classes&column=species").text
        assert _rows(page) == [["a", "3", "60,0 %"], ["b", "1", "20,0 %"],
                               ["Okänd", "1", "20,0 %"], ["Totalt", "5", "100,0 %"]]
        english = client.get(f"/ui/dataset?dataset={dataset}&view=classes&column=species",
                             headers={"accept-language": "en"}).text
        assert _rows(english)[-2:] == [["Unknown", "1", "20.0 %"], ["In total", "5", "100.0 %"]]


def test_other_classes_keep_the_total(tmp_path):
    rows = "\n".join(f"{i};1.0;x;c{i:03d}" for i in range(105)).encode()
    with _client(tmp_path) as client:
        _, dataset = _setup(client, b"id;sepal;note;species\n" + rows + b"\n")
        page = client.get(f"/ui/dataset?dataset={dataset}&view=classes&column=species").text
        table = _rows(page)
        assert len(table) == 100 + 3                                   # classes + other + unknown + total
        assert table[-3][:2] == ["Övriga klasser", "5"] and table[-1][:2] == ["Totalt", "105"]
