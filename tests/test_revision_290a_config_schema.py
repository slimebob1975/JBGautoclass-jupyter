"""290a: configuration schema for generated forms."""
import json
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from JBGConfigSchema import FIELDS, INLINE_OPTIONS, configuration_schema, interface_texts  # noqa: E402
from JBGConfigurationService import _server_template, user_view  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402

RUN_TIME_CHECKED = {"mode.decision_policy", "mode.holdout_split_mode"}


def _client(tmp_path):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _loads(section, name, value) -> bool:
    """Does the existing Config loader accept ``value`` for ``section.name``?"""
    from Config import Config
    payload = _server_template()
    payload[section][name] = value
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "c.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            Config.load_config_from_json(path)
            return True
        except Exception:
            return False


# --- Coverage and derivation ----------------------------------------------------------------

def test_every_user_owned_field_has_exactly_one_entry():
    keys = {f"{section}.{name}" for section, fields in user_view(_server_template()).items()
            for name in fields}
    assert keys == set(FIELDS)


def test_option_lists_are_derived_from_the_code():
    import Config
    import JBGMeta
    from JBGFeatureSearchPresets import PRESETS
    schema = configuration_schema("sql", lambda t: t, _server_template())
    options = {f["key"]: [o["value"] for o in f.get("options", [])] for f in schema["fields"]}
    assert options["mode.algorithm"] == [m.name for m in JBGMeta.Algorithm]
    assert options["mode.preprocessor"] == [m.name for m in JBGMeta.Preprocess]
    assert options["mode.scoring"] == [m.name for m in JBGMeta.ScoreMetric]
    assert options["mode.oversamplers"] == [m.name for m in JBGMeta.Oversampling]
    assert options["mode.dark_number_alpha"] == [m.name for m in Config.DarkNumberAlpha]
    assert options["mode.feature_search_preset"] == [*PRESETS, "Custom"]


@pytest.mark.parametrize("key", sorted(set(INLINE_OPTIONS) - RUN_TIME_CHECKED))
def test_declared_values_match_the_config_checks(key):
    section, name = key.split(".", 1)
    for value in INLINE_OPTIONS[key]:
        assert _loads(section, name, value), (key, value)
    assert not _loads(section, name, "not-a-valid-value"), key


def test_values_checked_at_run_time_match_that_code():
    from JBGDecisionPolicy import validate_decision_policy
    for policy in INLINE_OPTIONS["mode.decision_policy"]:
        validate_decision_policy(policy, 0.5)
    with pytest.raises(ValueError):
        validate_decision_policy("not-a-policy", 0.5)
    handler = (Path(__file__).resolve().parents[1] / "src" / "JBGHandler.py").read_text(encoding="utf-8")
    for mode in INLINE_OPTIONS["mode.holdout_split_mode"]:
        assert f'split_mode == "{mode}"' in handler, mode


def test_every_schema_text_has_a_swedish_translation():
    import JBGWebUI
    missing = sorted(t for t in interface_texts() if t not in JBGWebUI._CATALOGS["sv"])
    assert not missing, missing


# --- Endpoint --------------------------------------------------------------------------------

def test_schema_in_swedish_with_technical_names_unchanged(tmp_path):
    import JBGMeta
    with _client(tmp_path) as client:
        schema = client.get("/api/config/schema", params={"source": "csv", "language": "sv"}).json()
    assert schema["language"] == "sv" and schema["source"] == "csv"
    groups = {g["key"]: g["label"] for g in schema["groups"]}
    # 298b: pipeline-order groups (scaling was part of "Förbehandling").
    assert groups["scaling"] == "Skalning" and groups["dark_numbers"] == "Mörkertal"
    assert "predictions" not in groups                                   # SQL only
    fields = {f["key"]: f for f in schema["fields"]}
    assert "connection.class_table" not in fields
    assert fields["mode.test_size"]["label"] == "Andel testdata"
    algorithms = {o["value"]: o["label"] for o in fields["mode.algorithm"]["options"]}
    assert algorithms["LRN"] == JBGMeta.Algorithm.LRN.full_name           # never translated
    assert fields["mode.decision_policy"]["options"][1]["label"] == "Fast sannolikhetströskel"
    lrn = {p["name"]: p for p in schema["model_parameters"]["LRN"]}
    assert lrn["C"]["label"] == "Invers regularisering (C)" and lrn["C"]["minimum"] == 0.0001


def test_language_follows_the_browser_and_sources_differ(tmp_path):
    with _client(tmp_path) as client:
        english = client.get("/api/config/schema", params={"source": "sql"},
                             headers={"accept-language": "en-GB"}).json()
        assert english["language"] == "en"
        assert {g["key"]: g["label"] for g in english["groups"]}["predictions"] == "Predictions"
        assert client.get("/api/config/schema", params={"source": "file"}).status_code == 422


def test_a_form_built_from_the_schema_can_be_saved(tmp_path):
    # Exactly what a generated form would send: every field with its default.
    with _client(tmp_path) as client:
        for source in ("sql", "csv"):
            schema = client.get("/api/config/schema", params={"source": source}).json()
            settings = {}
            for field in schema["fields"]:
                section, name = field["key"].split(".", 1)
                settings.setdefault(section, {})[name] = field["default"]
            project = client.post("/api/projects", json={"name": source, "source": source}).json()
            saved = client.post(f"/api/projects/{project['project_id']}/configurations",
                                json={"name": "From schema", "settings": settings})
            assert saved.status_code == 201, saved.text
