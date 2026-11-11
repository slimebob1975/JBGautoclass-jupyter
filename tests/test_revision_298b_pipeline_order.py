"""298b: the Configuration tab follows the pipeline order (agreed 2026-11-11),
as the Voila GUI does. The schema's groups are in that order; field keys and
stored configurations are unchanged. Run type and algorithms are open; the
other sections are collapsed (293c's "More settings" is gone). Since 298c
every section starts closed."""
import re
from datetime import timedelta

import pytest

from JBGConfigSchema import FIELDS, GROUPS, configuration_schema

ORDER = ["run_type", "text_processing", "feature_selection", "splitting", "scaling",
         "oversampling", "undersampling", "reduction", "algorithms", "decision", "evaluation",
         "dark_numbers", "predictions", "advanced"]
SWEDISH = ["Körningstyp", "Textbehandling", "Variabelurval", "Uppdelning", "Skalning",
           "Översampling", "Undersampling", "Dimensionsreduktion", "Algoritmer", "Beslut",
           "Utvärdering", "Mörkertal", "Prediktioner", "Avancerat"]
ENGLISH = ["Run type", "Text processing", "Feature selection", "Data split", "Scaling",
           "Oversampling", "Undersampling", "Dimensionality reduction", "Algorithms", "Decision",
           "Evaluation", "Dark Numbers", "Predictions", "Advanced"]


def _schema(source="sql", language="sv"):
    from JBGConfigurationService import _server_template
    from JBGWebUI import translator
    return configuration_schema(source, translator(language), _server_template())


# --- The schema ------------------------------------------------------------------------------

def test_the_groups_follow_the_pipeline():
    assert [key for key, _ in GROUPS] == ORDER
    assert [g["key"] for g in _schema("sql")["groups"]] == ORDER
    assert [g["key"] for g in _schema("csv")["groups"]] == [k for k in ORDER if k != "predictions"]
    assert _schema()["schema_version"] == 2


@pytest.mark.parametrize("key, group", [
    ("mode.use_metas", "run_type"), ("mode.use_categorization", "text_processing"),
    ("mode.ngram_range", "text_processing"), ("mode.feature_selection_mode", "feature_selection"),
    ("mode.feature_search_preset", "feature_selection"), ("mode.test_size", "splitting"),
    ("mode.validation_group_column", "splitting"), ("mode.preprocessor", "scaling"),
    ("mode.oversamplers", "oversampling"), ("mode.undersamplers", "undersampling"),
    ("mode.feature_selection", "reduction"), ("mode.num_selected_features", "reduction"),
    ("mode.algorithm", "algorithms"), ("mode.scoring", "algorithms"),
    ("mode.decision_policy", "decision"), ("mode.calculate_feature_importance", "evaluation"),
])
def test_each_field_sits_in_its_step(key, group):
    assert FIELDS[key]["group"] == group


def test_every_group_has_fields_and_every_field_a_group():
    used = {entry["group"] for entry in FIELDS.values()}
    assert used == set(ORDER)


def test_labels_in_both_languages():
    assert [g["label"] for g in _schema(language="sv")["groups"]] == SWEDISH
    assert [g["label"] for g in _schema(language="en")["groups"]] == ENGLISH


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


def _configuration(client, source="csv"):
    location = client.post("/ui/projects", data={"name": "P", "source": source},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    created = client.post("/ui/configuration", data={"name": "C"}, follow_redirects=False)
    return re.search(r"configuration=([0-9a-f-]{36})", created.headers["location"]).group(1)


def _sections(page):
    """(key, open, label) per section, in page order."""
    return [(key, bool(opened), label) for key, opened, label in re.findall(
        r'<details class="group" id="section-(\w+)" ?(open)?>\s*<summary>([^<]+)</summary>', page)]


@pytest.mark.parametrize("source", ["csv", "sql"])
def test_the_form_shows_the_sections_in_pipeline_order(tmp_path, source):
    with _client(tmp_path) as client:
        page = client.get(f"/ui/configuration?configuration={_configuration(client, source)}").text
    sections = _sections(page)
    expected = [k for k in ORDER if source == "sql" or k != "predictions"]
    assert [key for key, _, _ in sections] == expected
    assert [key for key, opened, _ in sections if opened] == []          # 298c: all closed
    assert [label for _, _, label in sections] == [
        label for key, label in zip(ORDER, SWEDISH) if key in expected]
    assert "Fler inställningar" not in page


def test_the_form_in_english(tmp_path):
    with _client(tmp_path) as client:
        configuration = _configuration(client)
        client.post("/ui/language", data={"language": "en", "back": "/ui/projects"},
                    follow_redirects=False)
        page = client.get(f"/ui/configuration?configuration={configuration}").text
    assert [label for _, _, label in _sections(page)] == [
        label for key, label in zip(ORDER, ENGLISH) if key != "predictions"]


def test_a_refused_value_opens_its_section(tmp_path):
    from JBGConfigurationService import _server_template
    with _client(tmp_path) as client:
        configuration = _configuration(client)
        template = _server_template()
        form = {"expected_version": "1", "f:mode.train": "on", "f:mode.algorithm": "LRN",
                "f:mode.test_size": "lots",
                **{f"f:{key}": str(template[key.split(".")[0]][key.split(".")[1]])
                   for key, entry in FIELDS.items()
                   if entry["kind"] == "choice" and key.startswith("mode.")}}
        response = client.post(f"/ui/configuration/{configuration}", data=form)
    assert response.status_code == 422
    opened = [key for key, is_open, _ in _sections(response.text) if is_open]
    assert "splitting" in opened and "scaling" not in opened
