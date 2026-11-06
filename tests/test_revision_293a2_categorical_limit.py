"""293a2: configurable categorical limit (default 30; agreed range 2-1000)."""
import json
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from Config import Config

CSV = b"id;sepal;note;species\n1;5.1;x;a\n2;7.0;y;b\n3;6.3;z;a\n4;4.9;w;b\n"


def _load(payload):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "c.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return Config.load_config_from_json(path)


# --- Config and its loader ---------------------------------------------------------------

def test_older_files_load_with_30_and_new_files_keep_their_limit():
    payload = Config()._config_to_json(username="")
    assert payload["mode"]["categorical_limit"] == 30                 # saved from now on
    older = json.loads(json.dumps(payload))
    del older["mode"]["categorical_limit"]
    assert _load(older).get_categorical_limit() == 30
    payload["mode"]["categorical_limit"] = 50
    assert _load(payload).get_categorical_limit() == 50


@pytest.mark.parametrize("value", [1, 1001, "30", 30.0, None])
def test_the_limit_must_be_a_whole_number_from_2_to_1000(value):
    payload = Config()._config_to_json(username="")
    payload["mode"]["categorical_limit"] = value
    with pytest.raises(Exception):
        _load(payload)


def test_only_listed_later_keys_may_be_missing():
    assert Config.LATER_KEYS == {"mode": {"categorical_limit": 30}}
    payload = Config()._config_to_json(username="")
    del payload["mode"]["test_size"]                                  # still strict
    with pytest.raises(Exception, match="missing keys"):
        _load(payload)
    payload = Config()._config_to_json(username="")
    payload["mode"]["unknown_setting"] = 1
    with pytest.raises(Exception, match="unknown keys"):
        _load(payload)


# --- The engine uses the configured limit ---------------------------------------------------

def _converted_category_columns(limit, distinct):
    """Run the handler's own converter set-up with ``distinct`` text values."""
    from JBGHandler import DatasetHandler
    config = Config()
    config.mode.train, config.mode.predict = True, False
    from JBGMeta import Algorithm, AlgorithmTuple
    config.mode.algorithm = AlgorithmTuple([Algorithm.LRN])
    config.mode.use_categorization = True
    config.connection.data_text_columns = ["note"]
    config.connection.data_numerical_columns = ["sepal"]
    config.mode.categorical_limit = limit
    values = [f"v{i % distinct}" for i in range(40)]
    frame = pd.DataFrame({"sepal": range(40), "note": values})
    handler = DatasetHandler.__new__(DatasetHandler)
    handler.handler = SimpleNamespace(config=config,
                                      logger=SimpleNamespace(**{name: (lambda *a, **k: None) for name in (
                                          "print_dragon", "print_key_value_pair", "print_info",
                                          "print_progress", "print_warning")}))
    handler.X, handler.X_train, handler.X_validation = frame.copy(), frame.copy(), frame.copy()
    handler.X_prediction = None
    model = SimpleNamespace(text_converter=None, algorithm=None)
    converter = handler.convert_text_and_categorical_features(model)
    return converter.limit_categorize_, list(converter.category_columns_)


def test_the_handler_passes_the_configured_limit_to_the_converter():
    assert _converted_category_columns(5, 5) == (5, ["note"])        # at the limit: categorised
    assert _converted_category_columns(5, 6) == (5, [])              # above it: free text
    assert _converted_category_columns(30, 6)[1] == ["note"]          # the old fixed 30


def test_model_compatibility_compares_with_the_configured_limit():
    from JBGModelCompatibility import _converter_setting_issues, _expected_predictor_columns
    config = Config()
    config.connection.data_numerical_columns = ["sepal"]
    config.mode.categorical_limit = 50
    columns = tuple(_expected_predictor_columns(config))
    fitted = SimpleNamespace(
        input_feature_columns_=columns,
        input_feature_type_families_={column: "numeric" for column in columns},
        configured_text_columns_=tuple(config.get_text_column_names()),
        configured_category_columns_=tuple(config.get_categorical_text_column_names()),
        limit_categorize_=50, language_="sv", stop_words_=config.use_stop_words(), df_=1.0,
        ngram_range_=tuple(config.get_ngram_range().ngram_range),
        use_encryption_=config.should_hex_encode(), use_categorization_=config.use_categorization())
    assert _converter_setting_issues(config, fitted) == []
    config.mode.categorical_limit = 30                 # a model built with 50, used with 30
    assert _converter_setting_issues(config, fitted) != []


def test_feature_search_uses_the_configured_limit():
    from JBGFeatureSearchService import text_options_from_config
    config = Config()
    config.mode.categorical_limit = 12
    assert text_options_from_config(config)["limit_categorize"] == 12


# --- The limit belongs to the dataset (decision A) ----------------------------------------------

def test_the_limit_is_stored_with_the_roles_and_used_by_runs(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("python_multipart")
    from fastapi.testclient import TestClient
    from JBGApplicationContracts import RunReference
    from JBGDatasetRoles import DatasetRoles
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    from JBGSQLServer import SQLServerSettings
    client = TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))
    roles = {"id": "id", "sepal": "numeric", "note": "text", "species": "class"}
    with client:
        project = client.post("/api/projects", json={"name": "P", "source": "csv"}).json()["project_id"]
        dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "D", "role": "training"},
                              files={"file": ("d.csv", CSV, "text/csv")}).json()["dataset_id"]
        client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
        url = f"/api/projects/{project}/datasets/{dataset}/roles"
        out_of_range = client.put(url, json={"columns": roles, "categorical_limit": 1})
        assert out_of_range.status_code == 422
        saved = client.put(url, json={"columns": roles, "categorical_limit": 75}).json()
        assert saved["categorical_limit"] == 75 and client.get(url).json()["categorical_limit"] == 75
        configuration = client.post(f"/api/projects/{project}/configurations", json={
            "name": "C", "settings": {"mode": {"categorical_limit": 999}}}).json()
        assert "categorical_limit" not in configuration["settings"]["mode"]     # server-owned
        run = client.post(f"/api/projects/{project}/runs",
                          json={"configuration_id": configuration["configuration_id"]}).json()
        spec = client.app.state.jbg["store"].get(RunReference(run["run_id"])).spec
    assert spec["configuration"]["mode"]["categorical_limit"] == 75
    # Roles stored in the 293a format (the mapping only) still read, with 30.
    legacy = DatasetRoles.from_json(json.dumps(roles))
    assert legacy.as_mapping() == roles and legacy.categorical_limit == 30
