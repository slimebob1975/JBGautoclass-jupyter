"""Where the durable run jobs live: SQL Server from .env, else a SQLite file (285d).

Settings (in the repository's .env or the environment; the environment wins):

    JBG_METADATA_DB_HOST      SQL Server host; empty selects the local SQLite file
    JBG_METADATA_DB_NAME      database (catalog)
    JBG_METADATA_DB_SCHEMA    schema for the jbg_run_ tables (optional)
    JBG_METADATA_DB_USERNAME  SQL login; empty selects Windows authentication
    JBG_METADATA_DB_PASSWORD  SQL password
    JBG_METADATA_DB_DRIVER    ODBC driver; defaults to DEFAULT_ODBC_DRIVER

The web server and every worker read the same settings themselves, so the
password never appears in a command line.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE = Path("output") / "run_jobs" / "jobs.sqlite"
_KEYS = ("HOST", "NAME", "SCHEMA", "USERNAME", "PASSWORD", "DRIVER")


@dataclass(frozen=True)
class MetadataStoreSettings:
    host: str = ""
    database: str = ""
    schema: str = ""
    username: str = ""
    password: str = field(default="", repr=False)
    driver: str = ""
    sqlite_path: Path = ROOT / DEFAULT_SQLITE

    @property
    def uses_sql_server(self) -> bool:
        return bool(self.host)

    def url(self):
        """SQLAlchemy URL; its string form hides the password."""
        from sqlalchemy.engine import URL
        if not self.uses_sql_server:
            return URL.create("sqlite", database=str(self.sqlite_path))
        if not self.database or not self.driver:
            raise ValueError("JBG_METADATA_DB_NAME and an ODBC driver are required")
        # Same transport options as JBG's own SQL connections (JBGSqlHelper).
        query = {"driver": self.driver, "Encrypt": "yes", "TrustServerCertificate": "yes"}
        if not self.username:
            query["Trusted_Connection"] = "yes"
        return URL.create("mssql+pyodbc", username=self.username or None,
                          password=self.password or None, host=self.host,
                          database=self.database, query=query)


def _environment(root: Path) -> dict:
    """The .env file's values, overridden by the real environment."""
    values = {}
    env_file = Path(root) / ".env"
    if env_file.is_file():
        from dotenv import dotenv_values
        values.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    values.update(os.environ)
    return values


def metadata_store_settings(root: Path = ROOT, environment: dict | None = None
                            ) -> MetadataStoreSettings:
    env = _environment(root) if environment is None else dict(environment)
    get = lambda key: (env.get(f"JBG_METADATA_DB_{key}") or "").strip()
    values = {key: get(key) for key in _KEYS}
    driver = values["DRIVER"] or (env.get("DEFAULT_ODBC_DRIVER") or "").strip()
    return MetadataStoreSettings(
        host=values["HOST"], database=values["NAME"], schema=values["SCHEMA"],
        username=values["USERNAME"], password=env.get("JBG_METADATA_DB_PASSWORD") or "",
        driver=driver, sqlite_path=Path(root) / DEFAULT_SQLITE)


def open_metadata_store(settings: MetadataStoreSettings | None = None, *,
                        create_tables: bool = False):
    """Open the RunJobStore described by ``settings`` (default: from .env)."""
    from JBGRunJobStore import RunJobStore
    settings = metadata_store_settings() if settings is None else settings
    if settings.uses_sql_server:
        from sqlalchemy import create_engine
        store = RunJobStore(create_engine(settings.url(), pool_pre_ping=True),
                            schema=settings.schema or None)
    else:
        settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        store = RunJobStore.sqlite(settings.sqlite_path)
    if create_tables:
        store.create_tables()
    return store


def open_project_repository(settings: MetadataStoreSettings | None = None, *,
                            create_tables: bool = False):
    """Open project metadata using the same DB settings as durable runs (285i)."""
    from JBGProjects import ProjectRepository
    from sqlalchemy import create_engine
    settings = metadata_store_settings() if settings is None else settings
    if not settings.uses_sql_server:
        settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    repository = ProjectRepository(create_engine(settings.url(), pool_pre_ping=True),
                                   schema=settings.schema or None)
    if create_tables:
        repository.create_tables()
    return repository
