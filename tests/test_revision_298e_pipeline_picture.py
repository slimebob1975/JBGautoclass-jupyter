"""298e (requested and agreed 2026-11-11): the Configuration tab's left
panel shows the pipeline of the configuration on show, in run order, each
step marked used, several alternatives or not used, with the
cross-validated steps framed; every step links to its section. Swedish and
English. Checked in headless Chromium (layout, links, narrow screens)."""
import re
from datetime import timedelta

import pytest

from JBGDatasetRoles import DatasetRoles
from JBGWebUIPipeline import CROSS_VALIDATED, pipeline_steps

ORDER = ["dataset", "text_processing", "feature_selection", "splitting", "scaling", "oversampling",
         "undersampling", "reduction", "algorithms", "decision", "evaluation", "dark_numbers",
         "predictions"]


def _schema(source="csv", language="sv"):
    from JBGConfigSchema import configuration_schema
    from JBGConfigurationService import _server_template
    from JBGWebUI import translator
    return configuration_schema(source, translator(language), _server_template())


def _values(**mode):
    from JBGConfigurationService import _server_template, user_view
    values = user_view(_server_template())
    values["mode"].update({"oversamplers": ["NOG"], "undersamplers": ["NUG"]}, **mode)
    return values


def _steps(source="csv", language="sv", roles=None, dataset="Djur", **mode):
    from JBGWebUI import translator
    steps = pipeline_steps(_schema(source, language), _values(**mode), translate=translator(language),
                           language=language, dataset_name=dataset, roles=roles)
    return {s["key"]: s for s in steps}, [s["key"] for s in steps]


# --- The steps -------------------------------------------------------------------------------

def test_the_steps_follow_the_pipeline():
    _, order = _steps()
    assert order == ORDER
    assert list(CROSS_VALIDATED) == order[4:9]


def test_the_defaults():
    steps, _ = _steps()
    states = {key: s["state"] for key, s in steps.items()}
    assert states == {"dataset": "used", "text_processing": "used", "feature_selection": "used",
                      "splitting": "used", "scaling": "unused", "oversampling": "unused",
                      "undersampling": "unused", "reduction": "unused", "algorithms": "used",
                      "decision": "used", "evaluation": "unused", "dark_numbers": "used",
                      "predictions": "unused"}
    assert steps["dataset"]["subtitle"] == "Djur"
    assert steps["splitting"]["subtitle"] == "Testandel 20 %"
    assert steps["text_processing"]["subtitle"].startswith("Kategorisering, Unigrams")
    assert steps["predictions"]["subtitle"] == "På fliken Prediktion"           # CSV: own tab


@pytest.mark.parametrize("chosen, state, subtitle", [
    (["ALL"], "alternatives", "Alla, 6 alternativ"), (["STA", "MIX"], "alternatives", "2 alternativ"),
    (["STA"], "used", "Standard Scaler"), (["NOS"], "unused", "Används inte")])
def test_a_list_step(chosen, state, subtitle):
    steps, _ = _steps(preprocessor=chosen)
    assert (steps["scaling"]["state"], steps["scaling"]["subtitle"]) == (state, subtitle)


def test_several_algorithms_and_the_metric():
    steps, _ = _steps(algorithm=["LDA", "LRN"])
    assert steps["algorithms"]["state"] == "alternatives"
    assert steps["algorithms"]["subtitle"].startswith("2 alternativ, ")


def test_the_other_steps_follow_their_settings():
    steps, _ = _steps(source="sql", calculate_feature_importance=True, calculate_dark_numbers=False,
                      decision_policy="fixed", probability_threshold=0.7,
                      grouped_cross_validation=True, dark_number_target="hund")
    assert steps["evaluation"]["state"] == "used" and steps["dark_numbers"]["state"] == "unused"
    assert steps["decision"]["subtitle"] == "Fast sannolikhetströskel 0,7"
    assert steps["splitting"]["subtitle"] == "Testandel 20 %, grupper hålls isär"
    assert (steps["predictions"]["state"], steps["predictions"]["subtitle"]) == ("used", "Rader utan klass")
    dark, _ = _steps(dark_number_target="hund")
    assert dark["dark_numbers"]["subtitle"].endswith(", hund")


def test_text_processing_without_text_features():
    roles = DatasetRoles.from_mapping({"id": "id", "x": "numeric", "label": "class"})
    steps, _ = _steps(roles=roles)
    assert (steps["text_processing"]["state"], steps["text_processing"]["subtitle"]) == (
        "unused", "Inga textvariabler")


def test_without_an_active_dataset():
    steps, _ = _steps(dataset="")
    assert (steps["dataset"]["state"], steps["dataset"]["subtitle"]) == (
        "unused", "Inget aktivt träningsdataset")


def test_english():
    steps, _ = _steps(language="en", preprocessor=["ALL"])
    assert steps["scaling"]["subtitle"] == "All, 6 alternatives"
    assert steps["splitting"]["subtitle"] == "Test share 20 %"
    assert steps["predictions"]["subtitle"] == "On the Prediction tab"


# --- The tab ---------------------------------------------------------------------------------

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


def _project(client, mode=None):
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "Djur", "role": "training"},
                          files={"file": ("d.csv", b"id;x;label\n1;1;a\n2;2;b\n", "text/csv")}
                          ).json()["dataset_id"]
    client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
    return client.post(f"/api/projects/{project}/configurations", json={
        "name": "Bred", "settings": {"mode": mode or {}}}).json()["configuration_id"]


def test_the_picture_shows_the_configuration_on_show(tmp_path):
    with _client(tmp_path) as client:
        configuration = _project(client, mode={"preprocessor": ["ALL"], "algorithm": ["LDA", "LRN"]})
        page = client.get(f"/ui/configuration?configuration={configuration}").text
    card = page[page.index('class="card pipeline-card"'):page.index("</ul>", page.index("pipeline-legend"))]
    assert "Pipeline <span class=\"muted\">· Bred</span>" in card
    steps = re.findall(r'<li class="step (\w+)[^"]*" data-step="(\w+)">', card)
    assert [key for _, key in steps] == ORDER
    frame = card[card.index('class="pipeline-cv"'):card.index("</ol>", card.index('class="pipeline-cv"'))]
    assert re.findall(r'data-step="(\w+)"', frame) == list(CROSS_VALIDATED)
    assert "Korsvalidering · 12 kombinationer" in frame
    assert 'href="#section-scaling" data-open-section="section-scaling"' in card
    assert 'href="/ui/dataset"' in card and 'href="/ui/prediction"' in card
    assert "(Flera alternativ)" in card                                     # not by colour alone


def test_no_picture_without_a_configuration_on_show(tmp_path):
    with _client(tmp_path) as client:
        _project(client)
        client.cookies.clear()
        location = client.post("/ui/projects", data={"name": "Q", "source": "csv"},
                               follow_redirects=False).headers["location"]
        project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
        client.post(f"/ui/projects/{project}/open", follow_redirects=False)
        page = client.get("/ui/configuration").text
    assert "pipeline-card" not in page


def test_the_script_opens_linked_sections(tmp_path):
    with _client(tmp_path) as client:
        script = client.get("/ui/static/jbg.js").text
    assert "wireSectionLinks" in script and "data-open-section" in script
