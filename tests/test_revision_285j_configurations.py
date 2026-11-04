"""285j: owner-scoped, immutable versioned configuration metadata."""
import pytest

pytest.importorskip("sqlalchemy")

from JBGApplicationContracts import DataSourceKind
from JBGConfigurations import (ConfigurationVersionConflict,
                               DuplicateConfigurationName)
from JBGMetadataStore import (MetadataStoreSettings, open_configuration_repository,
                              open_project_repository)


def _repositories(tmp_path):
    settings = MetadataStoreSettings(sqlite_path=tmp_path / "metadata.sqlite")
    projects = open_project_repository(settings, create_tables=True)
    configs = open_configuration_repository(settings, create_tables=True)
    return projects, configs, settings


def _payload(value=1):
    return {"schema_version": 2, "settings": {"features": ["a", "b"], "runs": value}}


def test_create_reopen_and_revision_history(tmp_path):
    projects, configs, settings = _repositories(tmp_path)
    project = projects.create("alice", "Iris", DataSourceKind.SQL)
    first = configs.create("alice", project.project_id, "Baseline", _payload())
    assert first.version == 1 and first.settings == _payload()
    changed = configs.save("alice", project.project_id, first.configuration_id,
                           _payload(2), expected_version=1)
    assert changed.version == 2 and changed.settings == _payload(2)
    assert configs.get_owned("alice", project.project_id, first.configuration_id,
                             version=1).settings == _payload()
    configs.close()
    projects.close()
    reopened = open_configuration_repository(settings)
    assert reopened.get_owned("alice", project.project_id,
                              first.configuration_id).version == 2
    assert len(reopened.list_for_project("alice", project.project_id)) == 1
    reopened.close()


def test_owner_isolation_and_project_scope(tmp_path):
    projects, configs, _ = _repositories(tmp_path)
    mine = projects.create("alice", "Mine", DataSourceKind.SQL)
    other = projects.create("bob", "Other", DataSourceKind.CSV)
    cfg = configs.create("alice", mine.project_id, "A", _payload())
    with pytest.raises(KeyError):
        configs.get_owned("bob", mine.project_id, cfg.configuration_id)
    with pytest.raises(KeyError):
        configs.get_owned("alice", other.project_id, cfg.configuration_id)
    with pytest.raises(KeyError):
        configs.create("alice", other.project_id, "Steal", _payload())
    with pytest.raises(KeyError):
        configs.save("bob", mine.project_id, cfg.configuration_id,
                     _payload(2), expected_version=1)
    assert configs.get_owned("alice", mine.project_id, cfg.configuration_id).version == 1
    configs.close()
    projects.close()


def test_duplicate_names_per_project_and_optimistic_locking(tmp_path):
    projects, configs, _ = _repositories(tmp_path)
    p = projects.create("alice", "Iris", DataSourceKind.CSV)
    cfg = configs.create("alice", p.project_id, "Original", _payload())
    with pytest.raises(DuplicateConfigurationName):
        configs.create("alice", p.project_id, "original", _payload())
    configs.save("alice", p.project_id, cfg.configuration_id, _payload(2),
                 expected_version=1)
    with pytest.raises(ConfigurationVersionConflict):
        configs.save("alice", p.project_id, cfg.configuration_id, _payload(3),
                     expected_version=1)
    assert configs.get_owned("alice", p.project_id, cfg.configuration_id).version == 2
    configs.close()
    projects.close()


@pytest.mark.parametrize("settings", [
    {"settings": {}},
    {"schema_version": 1},
    {"schema_version": True},
    {"schema_version": 2, "connection": {"password": "hidden"}},
    {"schema_version": 2, "nested": [{"api_key": "hidden"}]},
    {"schema_version": 2, "value": float("nan")},
    {"schema_version": 2, "value": set()},
])
def test_invalid_or_secret_bearing_settings_are_rejected(tmp_path, settings):
    projects, configs, _ = _repositories(tmp_path)
    p = projects.create("alice", "P", DataSourceKind.SQL)
    with pytest.raises((ValueError, TypeError)):
        configs.create("alice", p.project_id, "Bad", settings)
    assert configs.list_for_project("alice", p.project_id) == ()
    configs.close()
    projects.close()


def test_saved_settings_are_detached_from_caller(tmp_path):
    projects, configs, _ = _repositories(tmp_path)
    p = projects.create("alice", "P", DataSourceKind.CSV)
    payload = _payload()
    cfg = configs.create("alice", p.project_id, "A", payload)
    payload["settings"]["features"].append("mutated")
    cfg.settings["settings"]["features"].append("mutated")
    assert configs.get_owned("alice", p.project_id, cfg.configuration_id).settings == _payload()
    configs.close()
    projects.close()
