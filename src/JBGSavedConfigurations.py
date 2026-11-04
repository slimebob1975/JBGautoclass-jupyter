"""Interim catalog of runnable configurations, shared with the Voila GUI (285d).

Until a configuration store exists (design gap 10), the API starts runs from
what Voila already saves:

* SQL: the ``autoclassconfig_*.json`` files Voila writes to ``src/config``
  whenever it starts an SQL run. Clients refer to them by file name; names are
  only accepted if they appear in the listing (never a path).
* CSV: the saved CSV configuration records of a project (column roles plus the
  complete schema-v2 settings), as listed in Voila's CSV model dropdown.
  Clients refer to them by project name and opaque configuration ID.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent


def default_csv_storage(project: str):
    """Voila's CSV storage for ``project`` (repository root under the repo root)."""
    from JBGRunComposition import local_csv_storage
    return local_csv_storage(project, base=ROOT)


class UnknownConfiguration(LookupError):
    """The named configuration is not in the catalog."""


class SavedConfigurations:
    def __init__(self, *, config_dir: Path = SRC / "config",
                 csv_storage: Callable[[str], tuple] = default_csv_storage) -> None:
        self._config_dir = Path(config_dir)
        self._csv_storage = csv_storage

    # --- SQL: configuration files written by Voila -------------------------------------

    def _sql_files(self) -> dict[str, Path]:
        from Config import Config
        if not self._config_dir.is_dir():
            return {}
        return {path.stem: path for path in sorted(self._config_dir.glob(
            f"{Config.CONFIG_FILENAME_START}*.json")) if path.is_file()}

    def load_sql(self, name: str):
        from Config import Config
        if not isinstance(name, str):
            raise UnknownConfiguration("unknown configuration")
        path = self._sql_files().get(name)
        if path is None:
            raise UnknownConfiguration("unknown configuration")
        config = Config.load_config_from_json(path)
        if config.data_source.kind != "sql":
            raise UnknownConfiguration("not an SQL configuration")
        return config

    def sql_configurations(self, project: str | None = None) -> list[dict]:
        result = []
        for name in self._sql_files():
            try:
                config = self.load_sql(name)
            except Exception:
                continue                           # unreadable or not SQL: not offered
            if project is None or config.name == project:
                result.append({"name": name, "project": config.name})
        return result

    # --- CSV: saved configuration records of a project -------------------------------------

    def csv_configurations(self, project: str) -> list[dict]:
        from Storage.CSVDatasetCatalog import list_csv_dataset_labels
        from Storage.CSVModelConfiguration import list_csv_model_configurations
        repository, scope = self._csv_storage(project)
        result = []
        for dataset_label, artifact_id in list_csv_dataset_labels(repository, scope):
            for label, configuration_id in list_csv_model_configurations(
                    repository, scope, artifact_id):
                result.append({"configuration_id": configuration_id, "label": label,
                               "dataset": dataset_label})
        return result

    def load_csv(self, project: str, configuration_id: str):
        """Return (repository, scope, saved record) for one of the project's records."""
        from Storage.CSVModelConfiguration import load_csv_model_configuration
        if not isinstance(configuration_id, str) or not configuration_id:
            raise UnknownConfiguration("unknown configuration")
        repository, scope = self._csv_storage(project)
        known = {item["configuration_id"] for item in self.csv_configurations(project)}
        if configuration_id not in known:
            raise UnknownConfiguration("unknown configuration")
        saved = load_csv_model_configuration(repository, scope, configuration_id)
        if saved.shared_json is None:
            raise UnknownConfiguration("configuration has no saved run settings")
        return repository, scope, saved
