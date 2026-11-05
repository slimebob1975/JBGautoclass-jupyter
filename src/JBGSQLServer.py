"""The SQL Server that SQL projects use: server settings, never client input (288b).

Read from the repository's .env (the environment wins), the same keys the Voilà
GUI uses:

    DEFAULT_HOST, DEFAULT_ODBC_DRIVER        server and driver of SQL projects
    DEFAULT_DATA_CATALOG                     catalog for the sign-in probe
    DEFAULT_CLASSIFICATION_CATALOG / _TABLE  default prediction table

Agreed 2026-11-04 (decision 2): one SQL sign-in per user and server, valid for
all SQL projects on it; SQL Server itself decides which catalogs the login may
use.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from JBGMetadataStore import ROOT, _environment


@dataclass(frozen=True)
class SQLServerSettings:
    host: str = ""
    driver: str = ""
    probe_catalog: str = ""
    class_catalog: str = ""
    class_table: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.host and self.driver)

    def probe_config(self):
        """A minimal SQL Config for verifying a sign-in against this server."""
        from Config import Config
        config = Config()
        config.connection.host = self.host
        config.connection.odbc_driver = self.driver
        config.connection.data_catalog = self.probe_catalog
        config.connection.class_catalog = self.class_catalog or self.probe_catalog
        config.data_source.kind = "sql"
        return config

    def apply_to(self, config) -> None:
        """Set the server-owned connection fields of a run's Config."""
        config.connection.host = self.host
        config.connection.odbc_driver = self.driver
        if not (config.connection.class_catalog or "").strip():
            config.connection.class_catalog = self.class_catalog
        if not (config.connection.class_table or "").strip():
            config.connection.class_table = self.class_table


def sql_server_settings(root: Path = ROOT, environment: dict | None = None) -> SQLServerSettings:
    env = _environment(root) if environment is None else dict(environment)
    get = lambda key: (env.get(key) or "").strip()
    return SQLServerSettings(host=get("DEFAULT_HOST"), driver=get("DEFAULT_ODBC_DRIVER"),
                             probe_catalog=get("DEFAULT_DATA_CATALOG"),
                             class_catalog=get("DEFAULT_CLASSIFICATION_CATALOG"),
                             class_table=get("DEFAULT_CLASSIFICATION_TABLE"))
