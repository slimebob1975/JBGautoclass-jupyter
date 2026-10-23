"""Revision 176 JSON-only configuration contract."""

import json
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

from Config import Config
from JBGExceptions import ConfigException
from JBGMeta import AlgorithmTuple, PreprocessTuple, ReductionTuple


def configured_instance():
    config = Config()
    config.name = "example"
    config.connection.data_text_columns = ["text_a", "text_b"]
    config.connection.data_numerical_columns = ["value_a"]
    config.connection.class_column = "target"
    config.connection.id_column = "row_id"
    config.connection.sql_username = "runtime-user"
    config.connection.sql_password = "runtime-secret"
    config.mode.algorithm = AlgorithmTuple(["LRN", "RFCL"])
    config.mode.preprocessor = PreprocessTuple(["STA", "MAX"])
    config.mode.feature_selection = ReductionTuple(["NOR", "PCA"])
    config.mode.category_text_columns = ["text_a"]
    config.mode.feature_search_sizes = [4, 8, 16]
    config.mode.feature_search_non_linear_hidden_layers = [64, 32]
    config.mode.model_parameters = {
        "RFCL": {"n_estimators": 125, "weights": [1, 2]},
        "LRN": {"C": 0.75},
    }
    config.mail.smtp_server = "smtp.example.test"
    config.mail.notification_email = "ops@example.test"
    config.__post_init__()
    return config


def test_schema_v2_serializes_every_runtime_section_field_without_password(tmp_path):
    config = configured_instance()
    path = tmp_path / "config.json"

    config.save_to_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert set(payload) == {"schema_version", "name", "connection", "mode", "io", "debug", "mail"}
    for section_name, section_type in (
        ("connection", Config.Connection),
        ("mode", Config.Mode),
        ("io", Config.IO),
        ("debug", Config.Debug),
        ("mail", Config.Mail),
    ):
        assert set(payload[section_name]) == {item.name for item in dataclass_fields(section_type)}
    assert payload["connection"]["sql_password"] == ""
    assert "runtime-secret" not in path.read_text(encoding="utf-8")
    assert payload["mode"]["algorithm"] == ["LRN", "RFCL"]
    assert payload["mode"]["preprocessor"] == ["STA", "MAX"]
    assert payload["mode"]["feature_selection"] == ["NOR", "PCA"]
    assert payload["mode"]["feature_search_tree_estimator"] == "Extra Trees"
    assert payload["mode"]["model_parameters"]["RFCL"]["n_estimators"] == 125


def test_schema_v2_roundtrip_uses_runtime_password_only(tmp_path, monkeypatch):
    config = configured_instance()
    path = tmp_path / "config.json"
    config.save_to_file(path)
    monkeypatch.setenv(Config.SQL_PASSWORD_ENV, "from-environment")

    loaded = Config.load_config_from_json(path)

    assert loaded.connection.sql_password == "from-environment"
    assert loaded.connection.data_text_columns == config.connection.data_text_columns
    assert loaded.connection.data_numerical_columns == config.connection.data_numerical_columns
    assert loaded.mode.algorithm == config.mode.algorithm
    assert loaded.mode.preprocessor == config.mode.preprocessor
    assert loaded.mode.feature_selection == config.mode.feature_selection
    assert loaded.mode.feature_search_sizes == [4, 8, 16]
    assert loaded.mode.feature_search_non_linear_hidden_layers == [64, 32]
    assert loaded.mode.model_parameters == config.mode.model_parameters
    assert loaded.mail == config.mail


def test_old_schema_and_python_config_are_explicitly_rejected(tmp_path):
    old_json = tmp_path / "old.json"
    old_json.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ConfigException, match="schema_version"):
        Config.load_config_from_json(old_json)

    python_config = tmp_path / "old.py"
    python_config.write_text("raise RuntimeError('must never execute')\n", encoding="utf-8")
    with pytest.raises(ConfigException, match="Python configuration files are no longer supported"):
        Config.load_config_from_json(python_config)
    with pytest.raises(ConfigException, match="Python configuration files are no longer supported"):
        Config().save_to_file(python_config)


def test_schema_v2_is_strict_about_missing_and_unknown_fields(tmp_path):
    config = configured_instance()
    payload = config._config_to_json()

    missing = json.loads(json.dumps(payload))
    del missing["mode"]["decision_policy"]
    missing_path = tmp_path / "missing.json"
    missing_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ConfigException, match="missing keys"):
        Config.load_config_from_json(missing_path)

    unknown = json.loads(json.dumps(payload))
    unknown["mode"]["future_magic"] = True
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(ConfigException, match="unknown keys"):
        Config.load_config_from_json(unknown_path)


def test_cli_loader_accepts_json_only(tmp_path):
    config = configured_instance()
    path = tmp_path / "config.json"
    config.save_to_file(path)

    loaded = Config.load_config_from_arguments(["main", "--file", str(path)])
    assert loaded.name == "example"

    with pytest.raises(ConfigException, match="Unexpected command-line arguments"):
        Config.load_config_from_arguments(["main", "extra"])
    with pytest.raises(ConfigException, match="No configuration file specified"):
        Config.load_config_from_arguments(["main"])
    with pytest.raises(SystemExit) as exc:
        Config.load_config_from_arguments(["main", "--help"])
    assert exc.value.code == 0


def test_default_generated_config_filename_is_json():
    config = Config()
    assert config.filename.endswith(".json")
    assert config.get_filename("config").endswith(".json")
