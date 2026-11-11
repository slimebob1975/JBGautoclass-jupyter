"""298c (asked 2026-11-11): every Configuration section starts closed; the
Dark Numbers section offers the target class as a choice among the classes of
the active training dataset, and shows the formula of the chosen method and
alpha (from the engine's LaTeX, as MathML), as the Voila GUI does. The live
behaviour (jbg.js) was checked in headless Chromium; these tests cover what
the server renders."""
import html
import re
from datetime import timedelta

import pytest

from JBGDarkNumbers import DarkNumberCalculator
from JBGWebUIDarkNumbers import calculation_type, formula_cards, latex_to_mathml

# --- The formula -----------------------------------------------------------------------------

@pytest.mark.parametrize("kind", sorted(DarkNumberCalculator.FORMULA_LATEX))
def test_every_engine_formula_becomes_mathml(kind):
    mathml = latex_to_mathml(DarkNumberCalculator.FORMULA_LATEX[kind].replace("1/k", "1/3"))
    assert mathml.startswith('<math display="block"') and mathml.endswith("</math>")
    assert "<mi>TN</mi>" in mathml and "<mi>TP</mi>" in mathml
    assert ("<msubsup>" in mathml) == kind.startswith("non_linear")
    assert ("<mi>\u03b1</mi>" in mathml) == ("alpha" in kind)


@pytest.mark.parametrize("latex", [r"\beta", "{x", "x}", "^2", "x_a_b", "x # y"])
def test_anything_else_is_refused(latex):
    with pytest.raises(ValueError):
        latex_to_mathml(latex)


def test_the_cards_follow_the_engine():
    cards = formula_cards(lambda text: text)
    assert cards["types"] == {"LINEAR:NONE": "base", "LINEAR:SINGLE": "single_alpha",
                              "LINEAR:SEPARATED": "separated_alpha", "NON_LINEAR:NONE": "non_linear",
                              "NON_LINEAR:SINGLE": "non_linear_alpha"}
    assert set(cards["formulas"]) == set(DarkNumberCalculator.FORMULA_LATEX)
    assert all(card["mathml"] for card in cards["formulas"].values())
    assert calculation_type(cards, {"mode": {"dark_number_method": "NON_LINEAR",
                                             "dark_number_alpha": "SEPARATED"}}) == ""


# --- The tab ---------------------------------------------------------------------------------

CSV = "id;x;label\n1;1.0;katt\n2;2.0;hund\n3;3.0;katt\n4;4.0;älg\n".encode()


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
    dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "Djur", "role": "training"},
                          files={"file": ("d.csv", CSV, "text/csv")}).json()["dataset_id"]
    client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
    if roles:
        client.put(f"/api/projects/{project}/datasets/{dataset}/roles",
                   json={"columns": {"id": "id", "x": "numeric", "label": "class"}})
    configuration = client.post(f"/api/projects/{project}/configurations", json={
        "name": "C", "settings": {"mode": mode or {}}}).json()["configuration_id"]
    return project, dataset, configuration


def _section(page):
    start = page.index('id="section-dark_numbers"')
    return page[start:page.index("</details>", page.index("data-dark-formula", start))]


def _shown(section):
    return [kind for kind, hidden in re.findall(r'<div data-formula="(\w+)" (hidden)?>', section)
            if not hidden]


def test_every_section_starts_closed(tmp_path):
    with _client(tmp_path) as client:
        _, _, configuration = _project(client)
        page = client.get(f"/ui/configuration?configuration={configuration}").text
    sections = re.findall(r'<details class="group" id="section-\w+" ?(open)?>', page)
    assert len(sections) == 13 and not any(sections)


def test_the_target_class_is_chosen_among_the_training_datasets_classes(tmp_path):
    with _client(tmp_path) as client:
        project, dataset, configuration = _project(client)
        section = _section(client.get(f"/ui/configuration?configuration={configuration}").text)
        url = html.unescape(re.search(r'data-classes-url="([^"]+)"', section).group(1))
        classes = client.get(url).json()
    select = re.search(r'<select id="f-mode-dark_number_target"[^>]*>(.*?)</select>', section, re.S)
    assert re.findall(r'<option value="([^"]*)"', select.group(1)) == [""]   # read by the browser
    assert "Alla klasser" in select.group(1)
    assert url == f"/api/projects/{project}/datasets/{dataset}/class-distribution?column=label"
    assert [c["value"] for c in classes["classes"]] == ["katt", "hund", "älg"]
    assert "Klasser i kolumnen “label” i det aktiva träningsdatasetet “Djur”." in section


def test_a_saved_target_class_is_kept_and_named(tmp_path):
    with _client(tmp_path) as client:
        _, _, configuration = _project(client, mode={"dark_number_target": "hund"})
        section = _section(client.get(f"/ui/configuration?configuration={configuration}").text)
    assert '<option value="hund" selected>hund</option>' in section
    assert "Målklass: <span data-formula-target>hund</span>" in section


def test_without_a_class_column_the_dataset_tab_is_pointed_to(tmp_path):
    with _client(tmp_path) as client:
        _, _, configuration = _project(client, roles=False)
        section = _section(client.get(f"/ui/configuration?configuration={configuration}").text)
    assert "data-classes-url" not in section
    assert "Ange klasskolumnen för det aktiva träningsdatasetet under fliken Dataset" in section


@pytest.mark.parametrize("method, alpha, shown", [
    ("LINEAR", "SEPARATED", "separated_alpha"), ("LINEAR", "NONE", "base"),
    ("NON_LINEAR", "SINGLE", "non_linear_alpha")])
def test_the_saved_formula_is_shown(tmp_path, method, alpha, shown):
    with _client(tmp_path) as client:
        _, _, configuration = _project(client, mode={"dark_number_method": method,
                                                     "dark_number_alpha": alpha})
        section = _section(client.get(f"/ui/configuration?configuration={configuration}").text)
    assert _shown(section) == [shown]
    assert "data-formula-missing hidden" in section
    assert ("rot = 3" in section.split(f'data-formula="{shown}"')[1].split("</div>")[0]) == (
        method == "NON_LINEAR")


def test_an_estimate_that_is_off_fades_the_card(tmp_path):
    with _client(tmp_path) as client:
        _, _, configuration = _project(client, mode={"calculate_dark_numbers": False})
        section = _section(client.get(f"/ui/configuration?configuration={configuration}").text)
    assert 'class="formula-card off"' in section
    assert "<p class=\"formula-note\" data-formula-off >" in section


def test_an_unsupported_combination_is_refused_and_explained(tmp_path):
    from JBGConfigSchema import configuration_schema
    from JBGConfigurationService import _server_template, user_view
    with _client(tmp_path) as client:
        project, _, configuration = _project(client)
        schema = configuration_schema("csv", lambda text: text, _server_template())
        values = user_view(_server_template())
        form = []
        for field in schema["fields"]:
            if field["kind"] == "parameters":
                continue
            section, name = field["key"].split(".", 1)
            value, key = values[section][name], "f:" + field["key"]
            if field["kind"] == "boolean":
                form += [(key, "on")] if value else []
            elif field["kind"] == "multi_choice":
                form += [(key, v) for v in value or []]
            elif field["kind"] == "integer_list":
                form.append((key, ", ".join(str(v) for v in value or [])))
            else:
                form.append((key, "" if value is None else str(value)))
        form = [(k, v) for k, v in form if k not in ("f:mode.dark_number_method",
                                                    "f:mode.dark_number_alpha")]
        form += [("f:mode.dark_number_method", "NON_LINEAR"), ("f:mode.dark_number_alpha", "SEPARATED"),
                 ("expected_version", "1")]
        response = client.post(f"/ui/configuration/{configuration}", content=_encode(form),
                               headers={"content-type": "application/x-www-form-urlencoded"})
    assert response.status_code == 422
    section = _section(response.text)
    assert _shown(section) == []
    assert "data-formula-missing >" in section and "finns bara med metoden" in section


def _encode(items):
    from urllib.parse import urlencode
    return urlencode(items)


def test_english(tmp_path):
    with _client(tmp_path) as client:
        _, _, configuration = _project(client)
        client.post("/ui/language", data={"language": "en", "back": "/ui/projects"},
                    follow_redirects=False)
        section = _section(client.get(f"/ui/configuration?configuration={configuration}").text)
    for text in ("All classes", "Formula", "If the correction cannot be estimated",
                 "Classes of the column “label” in the active training dataset “Djur”."):
        assert text in section


def test_the_script_wires_the_card_and_the_classes(tmp_path):
    with _client(tmp_path) as client:
        script = client.get("/ui/static/jbg.js").text
    for name in ("wireDarkFormula", "wireClassChoice", "keepOpenSections", "jbg:classes"):
        assert name in script
