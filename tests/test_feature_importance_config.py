"""Defaults, config validation, previous-run snapshots and old artifact normalization."""
from types import SimpleNamespace
from pathlib import Path
import importlib.util

import pytest

import Config as config_module
from Config import Config
from GUIHandler import GUIHandler


def lightweight_config(mode=None):
    config = Config.__new__(Config)
    config.mode = mode or Config.Mode()
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


def test_old_model_mode_defaults_and_runtime_override(monkeypatch):
    saved = lightweight_config()
    # Remove instance fields to emulate a pre-106 artifact; dataclass defaults may still exist on its class.
    del saved.mode.calculate_feature_importance
    del saved.mode.feature_importance_repeats
    monkeypatch.setattr(config_module, 'load_model_config', lambda filename: saved)
    loaded = Config.load_config_from_model_file('unused.sav')
    assert loaded.mode.calculate_feature_importance is False
    assert loaded.mode.feature_importance_repeats == 5
    runtime = lightweight_config(Config.Mode(calculate_feature_importance=True, feature_importance_repeats=3))
    loaded = Config.load_config_from_model_file('unused.sav', runtime)
    assert loaded.mode.calculate_feature_importance is True
    assert loaded.mode.feature_importance_repeats == 3


def test_importance_export_paths_are_csvs(tmp_path):
    config = lightweight_config()
    for kind in ('feature_importance', 'feature_importance_details'):
        path = config.get_output_filepath(kind, pwd=tmp_path)
        assert path.parent == tmp_path / 'output/csvs'
        assert path.suffix == '.csv'


def test_generated_config_and_legacy_module_parsing(monkeypatch, tmp_path):
    config = lightweight_config(Config.Mode(calculate_feature_importance=True, feature_importance_repeats=7))
    config.config_path = Path(config_module.__file__).parent / 'config'
    filename = tmp_path / 'generated.py'
    config.save_to_file(filepath=filename)
    spec = importlib.util.spec_from_file_location('importance_config', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.mode['calculate_feature_importance'] is True
    assert module.mode['feature_importance_repeats'] == 7
    assert module.mode['ngram_range'] == config.mode.ngram_range.name
    assert module.connection['sql_password'] == ''
    # Parsing is separate from whole-configuration validation, covered by Mode tests above.
    monkeypatch.setattr(Config, '__post_init__', lambda self: None)
    parsed = Config.load_config(module)
    assert parsed.mode.calculate_feature_importance is True
    assert parsed.mode.feature_importance_repeats == 7
    assert parsed.mode.ngram_range is config.mode.ngram_range
    assert parsed.mail.notification_email == ''
    module.mail = {'smtp_server': 'configured-server', 'notification_email': 'configured@example.test'}
    module.mode['ngram_range'] = config.mode.ngram_range
    module.mode.pop('calculate_feature_importance')
    module.mode.pop('feature_importance_repeats')
    parsed = Config.load_config(module)
    assert parsed.mode.calculate_feature_importance is False
    assert parsed.mode.feature_importance_repeats == 5
    assert parsed.mail.smtp_server == 'configured-server'
    assert parsed.mail.notification_email == 'configured@example.test'
