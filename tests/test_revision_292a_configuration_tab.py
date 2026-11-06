"""292a: Configuration tab; forms generated from the 290a schema."""
import re
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.datastructures import FormData  # noqa: E402

from JBGConfigSchema import configuration_schema  # noqa: E402
from JBGConfigurationService import _server_template  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402
from JBGWebUIConfigurations import FIELD_ERRORS, MESSAGES, convert_form  # noqa: E402

CSV = b"id;sepal;petal;species\n1;5.1;1.4;a\n2;7.0;4.7;b\n3;6.3;6.0;a\n4;4.9;1.4;b\n"


def _schema(source="csv"):
    return configuration_schema(source, lambda text: text, _server_template())


def _form_for(values, schema, **overrides):
    """What a browser sends for the shown values (unticked boxes are absent)."""
    items = []
    for field in schema["fields"]:
        if field["kind"] == "parameters":
            continue
        section, name = field["key"].split(".", 1)
        value = values[section][name]
        key = "f:" + field["key"]
        if field["kind"] == "boolean":
            if value:
                items.append((key, "on"))
        elif field["kind"] in ("multi_choice", "columns"):
            items += [(key, v) for v in (value or [])]
        elif field["kind"] == "integer_list":
            items.append((key, ", ".join(str(v) for v in value or [])))
        else:
            items.append((key, "" if value is None else str(value)))
    items = [(k, v) for k, v in items if k not in overrides]
    for key, value in overrides.items():
        items += [(key, v) for v in (value if isinstance(value, list) else [value])]
    return items


# --- Form conversion ------------------------------------------------------------------------

def test_converting_the_default_form_gives_back_the_defaults():
    schema = _schema()
    defaults = {f["key"]: f["default"] for f in schema["fields"]}
    from JBGConfigurationService import user_view
    values = user_view(_server_template())
    settings, errors = convert_form(schema, FormData(_form_for(values, schema)))
    assert errors == {}
    for field in schema["fields"]:
        if field["kind"] == "parameters":
            continue
        section, name = field["key"].split(".", 1)
        assert settings[section][name] == defaults[field["key"]], field["key"]


def test_kinds_are_converted_and_checked():
    schema = _schema()
    from JBGConfigurationService import user_view
    values = user_view(_server_template())
    form = FormData(_form_for(values, schema, **{
        "f:mode.test_size": "0,25",                     # decimal comma
        "f:debug.data_limit": "",                       # nullable: no value
        "f:mode.feature_search_sizes": "5, 10;20",
        "p:LRN:C": "0,5", "p:LRN:class_weight": "None", "p:KNN:n_neighbors": "",
    }))
    settings, errors = convert_form(schema, form)
    assert errors == {}
    assert settings["mode"]["test_size"] == 0.25 and settings["debug"]["data_limit"] is None
    assert settings["mode"]["feature_search_sizes"] == [5, 10, 20]
    assert "data_numerical_columns" not in settings.get("connection", {})      # 293c
    # The registry's own option is the text "None" (JBGModelSettings).
    assert settings["mode"]["model_parameters"] == {"LRN": {"C": 0.5, "class_weight": "None"}}
    assert settings["mode"]["train"] is True
    unticked = [(k, v) for k, v in form.multi_items() if k != "f:mode.train"]
    assert convert_form(schema, FormData(unticked))[0]["mode"]["train"] is False


@pytest.mark.parametrize("key, value, error", [
    ("f:debug.data_limit", "abc", "integer"), ("f:mode.test_size", "x", "number"),
    ("f:mode.test_size", "1.5", "range"), ("f:mode.max_iterations", "", "required"),
    ("f:mode.scoring", "nope", "choice"), ("f:mode.algorithm", ["LRN", "NOPE"], "choice"),
    ("f:mode.feature_search_sizes", "5, x", "list"), ("p:LRN:C", "-1", "range"),
    ("p:LRN:class_weight", "heavy", "choice"),
])
def test_bad_values_become_field_errors(key, value, error):
    schema = _schema()
    from JBGConfigurationService import user_view
    form = FormData(_form_for(user_view(_server_template()), schema, **{key: value}))
    assert convert_form(schema, form)[1] == {key.replace("f:", "", 1) if key.startswith("f:") else key: error}


# --- The tab ------------------------------------------------------------------------------------

def _as_form(items):
    """httpx takes form data as a mapping; repeated fields become lists."""
    form = {}
    for key, value in items:
        form.setdefault(key, []).append(value)
    return form


def _client(tmp_path, subject="developer"):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _project_with_dataset(client, with_dataset=True):
    location = client.post("/ui/projects", data={"name": "Iris", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    if with_dataset:
        uploaded = client.post("/ui/dataset/csv", data={"name": "Data", "role": "training"},
                               files={"file": ("d.csv", CSV, "text/csv")}, follow_redirects=False)
        dataset = re.search(r"dataset=([0-9a-f-]{36})", uploaded.headers["location"]).group(1)
        client.post(f"/ui/dataset/{dataset}/activate", follow_redirects=False)
    return project


def _create(client, name="Standard"):
    created = client.post("/ui/configuration", data={"name": name}, follow_redirects=False)
    return re.search(r"configuration=([0-9a-f-]{36})", created.headers["location"]).group(1)


def _current(client, project, configuration):
    return client.get(f"/api/projects/{project}/configurations/{configuration}").json()


def test_the_tab_needs_a_project(tmp_path):
    with _client(tmp_path) as client:
        assert "Öppna först ett projekt under fliken Projekt." in client.get("/ui/configuration").text


def test_create_and_save_through_the_form(tmp_path):
    with _client(tmp_path) as client:
        project = _project_with_dataset(client)
        configuration = _create(client)
        page = client.get(f"/ui/configuration?configuration={configuration}").text
        # 293c: column roles are set on the Dataset tab, not here.
        assert "Förbehandling" in page and 'name="f:connection.class_column"' not in page
        assert "Kolumnroller anges under fliken Dataset." in page
        current = _current(client, project, configuration)
        form = _form_for(current["settings"], _schema(), **{
            "f:mode.algorithm": ["LRN"], "expected_version": "1"})
        saved = client.post(f"/ui/configuration/{configuration}", data=_as_form(form), follow_redirects=False)
        assert saved.status_code == 303 and "message=saved" in saved.headers["location"]
        stored = _current(client, project, configuration)
        assert stored["version"] == 2 and stored["settings"]["mode"]["algorithm"] == ["LRN"]
        page = client.get(saved.headers["location"]).text
        assert "Inställningar för LRN" in page                          # now shown
        assert re.search(r"<b>Standard v2</b>", page)                   # context bar


def test_field_errors_keep_the_other_entries(tmp_path):
    with _client(tmp_path) as client:
        project = _project_with_dataset(client)
        configuration = _create(client)
        current = _current(client, project, configuration)
        form = _form_for(current["settings"], _schema(), **{
            "f:debug.data_limit": "abc", "f:mode.max_iterations": "777", "expected_version": "1"})
        response = client.post(f"/ui/configuration/{configuration}", data=_as_form(form))
        assert response.status_code == 422
        assert "Ange ett heltal." in response.text and 'value="777"' in response.text
        assert _current(client, project, configuration)["version"] == 1        # nothing saved


def test_a_refused_configuration_and_a_newer_version(tmp_path):
    with _client(tmp_path) as client:
        project = _project_with_dataset(client)
        configuration = _create(client)
        current = _current(client, project, configuration)
        # Valid for the form, refused by the Config loader: neither train nor predict.
        neither = [(k, v) for k, v in _form_for(current["settings"], _schema(),
                                               expected_version="1")
                   if k not in ("f:mode.train", "f:mode.predict")]
        refused = client.post(f"/ui/configuration/{configuration}", data=_as_form(neither))
        assert refused.status_code == 422 and "Konfigurationen godtogs inte." in refused.text
        client.put(f"/api/projects/{project}/configurations/{configuration}",
                   json={"settings": {"debug": {"data_limit": 5}}, "expected_version": 1})
        stale = client.post(f"/ui/configuration/{configuration}",
                            data=_as_form(_form_for(current["settings"], _schema(),
                                                    expected_version="1")))
        assert stale.status_code == 409 and "Någon har sparat en nyare version." in stale.text
        assert _current(client, project, configuration)["settings"]["debug"]["data_limit"] == 5


def test_no_column_fields_remain_in_the_form(tmp_path):
    # 293c replaces the 292a column lists: roles belong to the dataset.
    with _client(tmp_path) as client:
        _project_with_dataset(client, with_dataset=False)
        configuration = _create(client)
        page = client.get(f"/ui/configuration?configuration={configuration}").text
        for key in ("connection.id_column", "connection.class_column",
                    "connection.data_numerical_columns", "connection.data_text_columns",
                    "mode.category_text_columns", "mode.excluded_predictor_columns"):
            assert f'name="f:{key}"' not in page, key


def test_other_users_and_cross_site_forms(tmp_path):
    with _client(tmp_path) as mine:
        project = _project_with_dataset(mine)
        configuration = _create(mine)
        assert mine.post(f"/ui/configuration/{configuration}", data={"expected_version": "1"},
                         headers={"origin": "http://evil.example"}).status_code == 403
    with _client(tmp_path, subject="bob") as theirs:
        theirs.cookies.set("jbg_project", project, path="/ui")
        page = theirs.get(f"/ui/configuration?configuration={configuration}").text
        assert "Öppna först ett projekt" in page and "Standard" not in page
        assert theirs.post(f"/ui/configuration/{configuration}/select",
                           follow_redirects=False).headers["location"] == \
            "/ui/configuration?message=unavailable"


def test_every_configuration_tab_text_has_a_swedish_translation():
    import JBGWebUI
    texts = {text for _, text in MESSAGES.values()} | set(FIELD_ERRORS.values())
    assert not sorted(t for t in texts if t not in JBGWebUI._CATALOGS["sv"])
