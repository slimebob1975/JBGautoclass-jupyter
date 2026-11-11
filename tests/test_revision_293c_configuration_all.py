"""293c: "ALL" choices, roles out of the configuration, essentials first."""
import json
import re
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

import JBGMeta
from Config import Config

LISTS = {"algorithm": JBGMeta.Algorithm, "preprocessor": JBGMeta.Preprocess,
         "feature_selection": JBGMeta.Reduction, "oversamplers": JBGMeta.Oversampling,
         "undersamplers": JBGMeta.Undersampling}


def _load(payload):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "c.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return Config.load_config_from_json(path)


# --- The engine expands ALL ---------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(LISTS))
def test_all_means_every_code_including_none(key):
    payload = Config()._config_to_json(username="")
    payload["mode"][key] = ["ALL"]
    exported = _load(payload)._config_to_json(username="")["mode"][key]
    assert exported == [member.name for member in LISTS[key]]


def test_all_wins_over_other_codes_and_files_without_all_are_unchanged():
    payload = Config()._config_to_json(username="")
    payload["mode"]["algorithm"] = ["LRN", "ALL"]
    assert len(_load(payload).get_algorithm_abbreviations()) == len(list(JBGMeta.Algorithm))
    plain = Config()._config_to_json(username="")
    plain["mode"]["algorithm"] = ["LRN"]
    assert _load(plain).get_algorithm_abbreviations() == ["LRN"]


# --- Form rules on the server (work without JavaScript) --------------------------------------

def _schema():
    from JBGConfigSchema import configuration_schema
    from JBGConfigurationService import _server_template
    return configuration_schema("csv", lambda t: t, _server_template())


@pytest.mark.parametrize("ticked, expected", [
    (["ALL"], ["ALL"]),                                            # rule 1
    (["ALL", "LRN"], ["ALL"]),                                     # rule 5
    ([m.name for m in JBGMeta.Algorithm], ["ALL"]),                # rule 3
    (["LRN", "KNN"], ["LRN", "KNN"]),
])
def test_the_server_applies_the_all_rules(ticked, expected):
    from starlette.datastructures import FormData
    from JBGConfigurationService import user_view, _server_template
    from JBGWebUIConfigurations import convert_form
    schema = _schema()
    values = user_view(_server_template())
    items = [("f:mode.algorithm", code) for code in ticked]
    for field in schema["fields"]:
        if field["kind"] in ("parameters", "multi_choice", "boolean") or field["key"] == "mode.algorithm":
            continue
        section, name = field["key"].split(".", 1)
        value = values[section][name]
        items.append(("f:" + field["key"], ", ".join(map(str, value)) if isinstance(value, list)
                      else "" if value is None else str(value)))
    for field in schema["fields"]:
        if field["kind"] == "multi_choice" and field["key"] != "mode.algorithm":
            section, name = field["key"].split(".", 1)
            items += [("f:" + field["key"], v) for v in values[section][name] or []]
    settings, errors = convert_form(schema, FormData(items))
    assert errors == {} and settings["mode"]["algorithm"] == expected


def test_schema_offers_all_on_exactly_the_five_lists_and_no_roles():
    schema = _schema()
    assert {f["key"] for f in schema["fields"] if f.get("allows_all")} == {
        f"mode.{key}" for key in LISTS}
    keys = {f["key"] for f in schema["fields"]}
    for role in ("connection.id_column", "connection.class_column", "connection.data_text_columns",
                 "connection.data_numerical_columns", "mode.category_text_columns",
                 "mode.excluded_predictor_columns"):
        assert role not in keys
    assert next(f for f in schema["fields"] if f["key"] == "mode.validation_group_column")[
        "group"] == "splitting"                                   # 298b


# --- Through the API and the tab -----------------------------------------------------------------

def _client(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("python_multipart")
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


CSV = b"id;sepal;species\n1;5.1;a\n2;7.0;b\n3;6.3;a\n4;4.9;b\n"


def _project(client):
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "D", "role": "training"},
                          files={"file": ("d.csv", CSV, "text/csv")}).json()["dataset_id"]
    client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
    client.put(f"/api/projects/{project}/datasets/{dataset}/roles",
               json={"columns": {"id": "id", "sepal": "numeric", "species": "class"}})
    return project


def test_stored_configurations_keep_all_and_runs_record_the_codes(tmp_path):
    from JBGApplicationContracts import RunReference
    with _client(tmp_path) as client:
        project = _project(client)
        created = client.post(f"/api/projects/{project}/configurations", json={
            "name": "C", "settings": {"mode": {"algorithm": ["ALL"], "preprocessor": ["ALL"]}}}).json()
        assert created["settings"]["mode"]["algorithm"] == ["ALL"]
        assert created["settings"]["mode"]["preprocessor"] == ["ALL"]
        run = client.post(f"/api/projects/{project}/runs",
                          json={"configuration_id": created["configuration_id"]}).json()
        spec = client.app.state.jbg["store"].get(RunReference(run["run_id"])).spec
    assert len(spec["configuration"]["mode"]["algorithm"]) == len(list(JBGMeta.Algorithm))


def test_the_form_shows_all_essentials_first_and_the_combinations(tmp_path):
    with _client(tmp_path) as client:
        project = _project(client)
        configuration = client.post(f"/api/projects/{project}/configurations", json={
            "name": "C", "settings": {"mode": {"algorithm": ["ALL"], "preprocessor": ["STA", "MIX"]}}}
        ).json()["configuration_id"]
        page = client.get(f"/ui/configuration?configuration={configuration}").text
    algorithms = re.search(r'<fieldset class="choices" data-all-group><legend>Algoritmer</legend>(.*?)</fieldset>',
                           page, re.S).group(1)
    assert re.search(r'value="ALL" data-all checked', algorithms)
    assert algorithms.count(" checked>") == len(list(JBGMeta.Algorithm)) + 1   # every box shown ticked
    expected = len(list(JBGMeta.Algorithm)) * 2       # x 2 scalers x 1 reduction x 1 x 1
    assert re.search(rf"Kombinationer att jämföra: <b>{expected}</b>", page)
    # 298b: pipeline order replaced 293c's essentials first.
    assert "Fler inställningar" not in page
    assert '<script src="/ui/static/jbg.js" defer></script>' in page


def test_the_script_is_served_and_needs_no_library(tmp_path):
    with _client(tmp_path) as client:
        script = client.get("/ui/static/jbg.js")
    assert script.status_code == 200 and "data-all-group" in script.text
    assert "import " not in script.text and "require(" not in script.text
