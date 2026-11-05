"""Owner-bound configuration service for the HTTP API (revision 286c).

Agreed (BACKLOG item 2, decision 5): the server owns the SQL host, ODBC driver,
credentials, model paths, SMTP server, the source type and the data location;
a client cannot set them, and sending them changes nothing. Everything else
(algorithms, column roles, sampling, validation, Dark Numbers, ...) belongs to
the user. Every configuration is validated with the existing Config loader
before it is stored, so an invalid configuration can never be saved.

The data location is not part of a configuration: a run adds the chosen
dataset when it starts (BACKLOG item 3). CSV configurations are therefore
validated with a placeholder dataset reference that is never stored.

Clients see and send a "user view": the sections without server-owned keys.
On save, keys that are not sent keep their current values.
"""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

from JBGApplicationContracts import DataSourceKind
from JBGConfigurations import (Configuration, ConfigurationRepository,
                               ConfigurationVersionConflict, DuplicateConfigurationName)
from JBGProjectService import ProjectService

USER_SECTIONS = ("connection", "mode", "io", "debug", "mail")
SERVER_OWNED = {
    "connection": frozenset({"odbc_driver", "host", "trusted_connection", "sql_username",
                             "sql_password", "data_catalog", "data_table"}),
    "io": frozenset({"model_path", "model_name"}),
    "mail": frozenset({"smtp_server"}),
}
_PLACEHOLDER_ID = "00000000-0000-4000-8000-000000000000"


class ConfigurationUnavailable(LookupError):
    """Unknown configuration (or version), or one of another user's project."""


class ConfigurationNameConflict(ValueError):
    """The project already has a configuration with that name."""


class ConfigurationChanged(ValueError):
    """Someone saved a newer version; reload before saving."""


class InvalidConfiguration(ValueError):
    """The settings are not a valid JBG configuration (fixed message, no values)."""


def _server_template() -> dict:
    """Server defaults for a new configuration (no credentials, no data location)."""
    from Config import Config
    return Config()._config_to_json(username="")


def user_view(settings: dict) -> dict:
    """The part of a stored configuration that belongs to the user."""
    view = {}
    for section in USER_SECTIONS:
        owned = SERVER_OWNED.get(section, frozenset())
        view[section] = {key: value for key, value in settings[section].items()
                         if key not in owned}
    return view


def _merge(base: dict, changes: Any) -> dict:
    """Apply user changes to ``base``; server-owned keys are ignored silently."""
    if not isinstance(changes, dict):
        raise InvalidConfiguration("configuration settings must be a JSON object")
    merged = copy.deepcopy(base)
    for section, values in changes.items():
        if section in ("schema_version", "name", "data_source"):
            continue                                         # server-owned, ignored
        if section not in USER_SECTIONS:
            raise InvalidConfiguration("unknown configuration section")
        if not isinstance(values, dict):
            raise InvalidConfiguration("configuration sections must be JSON objects")
        owned = SERVER_OWNED.get(section, frozenset())
        for key, value in values.items():
            if key in owned:
                continue                                     # server-owned, ignored
            if key not in merged[section]:
                raise InvalidConfiguration("unknown configuration setting")
            merged[section][key] = value
    return merged


def _normalized(settings: dict, project) -> dict:
    """Apply server-owned values, validate with Config, return the stored form."""
    from Config import Config
    template = _server_template()
    candidate = copy.deepcopy(settings)
    candidate["schema_version"] = template["schema_version"]
    candidate["name"] = project.project_id               # the stable identity
    for section, keys in SERVER_OWNED.items():
        for key in keys:
            candidate[section][key] = template[section][key]
    candidate["data_source"] = dict(template["data_source"], kind=project.source.value)

    probe = copy.deepcopy(candidate)
    if project.source is DataSourceKind.CSV:
        # Validation only: a CSV configuration needs a dataset reference, which
        # a run supplies later. The placeholder is never stored.
        probe["data_source"].update(dataset_id=_PLACEHOLDER_ID, artifact_id=_PLACEHOLDER_ID)
    try:
        with tempfile.TemporaryDirectory(prefix="jbg_cfg_") as folder:
            path = Path(folder) / "configuration.json"
            path.write_text(json.dumps(probe), encoding="utf-8")
            loaded = Config.load_config_from_json(path)
    except Exception:
        # Loader messages may contain values or temporary paths: never echo them.
        raise InvalidConfiguration("configuration is invalid") from None

    stored = loaded._config_to_json(username="")        # normalized, password blank
    for section, keys in SERVER_OWNED.items():
        for key in keys:
            stored[section][key] = template[section][key]
    stored["data_source"] = candidate["data_source"]
    stored["name"] = project.project_id
    return stored


def configuration_response(configuration: Configuration) -> dict[str, Any]:
    return {"configuration_id": configuration.configuration_id,
            "project_id": configuration.project_id, "name": configuration.name,
            "version": configuration.version, "settings": user_view(configuration.settings),
            "created_at": configuration.created_at.isoformat(),
            "updated_at": configuration.updated_at.isoformat()}


class ConfigurationService:
    def __init__(self, repository: ConfigurationRepository, projects: ProjectService) -> None:
        if not isinstance(repository, ConfigurationRepository):
            raise TypeError("ConfigurationRepository required")
        if not isinstance(projects, ProjectService):
            raise TypeError("ProjectService required")
        self._repository = repository
        self._projects = projects

    def list(self, owner: str, project_id: str) -> list[dict[str, Any]]:
        self._projects.require(owner, project_id)
        return [configuration_response(c)
                for c in self._repository.list_for_project(owner, project_id)]

    def get(self, owner: str, project_id: str, configuration_id: str,
            version: int | None = None) -> dict[str, Any]:
        self._projects.require(owner, project_id)
        try:
            return configuration_response(self._repository.get_owned(
                owner, project_id, configuration_id, version=version))
        except KeyError:
            raise ConfigurationUnavailable("configuration unavailable") from None

    def stored_settings(self, owner: str, project_id: str, configuration_id: str,
                        version: int | None = None) -> dict:
        """Complete stored schema-v2 settings, for run submission (item 3)."""
        self._projects.require(owner, project_id)
        try:
            return self._repository.get_owned(owner, project_id, configuration_id,
                                              version=version).settings
        except KeyError:
            raise ConfigurationUnavailable("configuration unavailable") from None

    def stored_configuration(self, owner: str, project_id: str, configuration_id: str,
                             version: int | None = None) -> Configuration:
        """The stored record (settings and version number), for run submission."""
        self._projects.require(owner, project_id)
        try:
            return self._repository.get_owned(owner, project_id, configuration_id,
                                              version=version)
        except KeyError:
            raise ConfigurationUnavailable("configuration unavailable") from None

    def create(self, owner: str, project_id: str, *, name: str,
               settings: dict) -> dict[str, Any]:
        project = self._projects.require(owner, project_id)
        stored = _normalized(_merge(_server_template(), settings), project)
        try:
            created = self._repository.create(owner, project_id, name, stored)
        except DuplicateConfigurationName:
            raise ConfigurationNameConflict("configuration name already exists") from None
        return configuration_response(created)

    def save(self, owner: str, project_id: str, configuration_id: str, *,
             settings: dict, expected_version: int) -> dict[str, Any]:
        project = self._projects.require(owner, project_id)
        current = self.stored_settings(owner, project_id, configuration_id)
        stored = _normalized(_merge(current, settings), project)
        try:
            saved = self._repository.save(owner, project_id, configuration_id, stored,
                                          expected_version=expected_version)
        except ConfigurationVersionConflict:
            raise ConfigurationChanged("configuration changed; reload before saving") from None
        except KeyError:
            raise ConfigurationUnavailable("configuration unavailable") from None
        return configuration_response(saved)
