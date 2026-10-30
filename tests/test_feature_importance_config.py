"""Defaults, config validation, previous-run snapshots and current JSON artifacts."""
from types import SimpleNamespace
from pathlib import Path
import json

import pytest

import Config as config_module
from Config import Config
from JBGExceptions import ConfigException
from GUIHandler import GUIHandler


def lightweight_config(mode=None):
    config = Config.__new__(Config)
    config.mode = mode or Config.Mode()
    config.data_source = Config.DataSource()
    config.connection = Config.Connection(sql_username="user", sql_password="secret", id_column="id")
    config.io = Config.IO(model_name="test")
    config.debug = Config.Debug()
    config.mail = Config.Mail()
    config.name = "test"
    config.save = False
    return config


def test_old_runtime_settings_default_off_and_five_repeats():
    config = lightweight_config()
    assert config.should_calculate_feature_importance() is False
    assert config.get_feature_importance_repeats() == 5
    config.mode = SimpleNamespace(train=True)
    assert config.should_calculate_feature_importance() is False
    assert config.get_feature_importance_repeats() == 5


def test_feature_importance_is_only_active_for_training():
    config = lightweight_config(Config.Mode(calculate_feature_importance=True))
    assert config.should_calculate_feature_importance() is True
    config.mode.train = False
    assert config.should_calculate_feature_importance() is False


@pytest.mark.parametrize("value", [True, 0, 1, 31, "5", 5.5])
def test_config_rejects_invalid_repeat_counts(value):
    with pytest.raises(ValueError, match="feature_importance_repeats"):
        Config.Mode(feature_importance_repeats=value).validate()


def test_config_rejects_nonboolean_enable_flag():
    with pytest.raises(TypeError, match="calculate_feature_importance"):
        Config.Mode(calculate_feature_importance="yes").validate()


def test_previous_run_roundtrip_and_missing_field_migration_keep_credentials_runtime_only():
    config = lightweight_config(Config.Mode(calculate_feature_importance=True, feature_importance_repeats=7))
    params = {name: getattr(config, name) for name in ('connection', 'mode', 'io', 'debug', 'mail', 'name', 'save')}
    snapshot = GUIHandler._config_params_to_last_run_snapshot(params)
    assert snapshot['mode']['calculate_feature_importance'] is True
    assert snapshot['mode']['feature_importance_repeats'] == 7
    assert 'sql_password' not in snapshot['connection']
    restored = GUIHandler._last_run_snapshot_to_config_params(snapshot, 'current-user', 'current-password')
    assert restored['mode'].calculate_feature_importance is True
    assert restored['mode'].feature_importance_repeats == 7
    assert restored['connection'].sql_password == 'current-password'
    snapshot['mode'].pop('calculate_feature_importance')
    snapshot['mode'].pop('feature_importance_repeats')
    restored = GUIHandler._last_run_snapshot_to_config_params(snapshot, 'current-user', 'current-password')
    assert restored['mode'].calculate_feature_importance is False
    assert restored['mode'].feature_importance_repeats == 5


def test_saved_model_missing_current_mode_fields_is_rejected(monkeypatch):
    saved = lightweight_config()
    # Dataclass defaults remain class attributes after del on an instance.
    # A legacy artifact must truly lack the fields, not merely their instance values.
    old_mode = vars(saved.mode).copy()
    old_mode.pop("calculate_feature_importance")
    old_mode.pop("feature_importance_repeats")
    saved.mode = SimpleNamespace(**old_mode)
    monkeypatch.setattr(config_module, 'load_model_config', lambda filename: saved)
    with pytest.raises(ConfigException, match="missing current fields"):
        Config.load_config_from_model_file('unused.sav')


def test_importance_export_paths_are_csvs(tmp_path):
    config = lightweight_config()
    for kind in ('feature_importance', 'feature_importance_details'):
        path = config.get_output_filepath(kind, pwd=tmp_path)
        assert path.parent == tmp_path / 'output/csvs'
        assert path.suffix == '.csv'


def test_generated_json_config_preserves_feature_importance_settings(tmp_path, monkeypatch):
    config = lightweight_config(Config.Mode(calculate_feature_importance=True, feature_importance_repeats=7))
    filename = tmp_path / 'generated.json'
    config.save_to_file(filepath=filename)
    payload = json.loads(filename.read_text(encoding='utf-8'))
    assert payload['schema_version'] == Config.JSON_CONFIG_SCHEMA_VERSION
    assert payload['mode']['calculate_feature_importance'] is True
    assert payload['mode']['feature_importance_repeats'] == 7
    assert payload['mode']['ngram_range'] == config.mode.ngram_range.name
    assert payload['connection']['sql_password'] == ''

    monkeypatch.setattr(Config, '__post_init__', lambda self: None)
    parsed = Config.load_config_from_json(filename)
    assert parsed.mode.calculate_feature_importance is True
    assert parsed.mode.feature_importance_repeats == 7
    assert parsed.mode.ngram_range is config.mode.ngram_range
