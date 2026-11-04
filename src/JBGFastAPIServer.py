"""Local FastAPI server with durable runs (revision 285d).

    uvicorn --factory JBGFastAPIServer:create_server_app --host 127.0.0.1

Starts and observes runs from saved configurations, stores them in the metadata
database (SQL Server from .env, else a local SQLite file) and executes them in
worker processes. Creating tables and starting the dispatcher happen when the
server starts, not on import. Local mode only: one Windows user, loopback.

JBGFastAPILocal (282b) remains the read-only, in-memory composition.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from JBGDurableRuns import DurableRunService
from JBGFastAPI import create_app
from JBGIdentity import AuthenticationMode, IdentitySettings, LocalIdentityProvider
from JBGMetadataStore import MetadataStoreSettings, metadata_store_settings, open_metadata_store, open_project_repository, open_configuration_repository, open_dataset_repository, open_model_catalog
from JBGRunDispatcher import RunDispatcher
from JBGProjectService import ProjectService
from JBGDatasetService import DatasetService
from JBGConfigurationService import ConfigurationService
from JBGProjectStorage import DEFAULT_CSV_ROOT
from JBGRunSubmission import RunSubmission
from JBGSavedConfigurations import ROOT, SavedConfigurations
from JBGSQLSessions import SQLSessionRegistry

DEFAULT_CSV_ARTIFACTS = ROOT / "output" / "csv_artifacts" / "objects"
DEFAULT_RUN_LOGS = ROOT / "logs" / "runs"


def _no_scope(principal):
    # Reads and cancellation never need a scope; submissions bring their own.
    raise PermissionError("storage scope is resolved per submission")


def create_server_app(*, settings: MetadataStoreSettings | None = None,
                      host: str = "127.0.0.1", subject: str = "developer",
                      configurations: SavedConfigurations | None = None,
                      csv_artifacts: Path = DEFAULT_CSV_ARTIFACTS,
                      run_logs: Path = DEFAULT_RUN_LOGS,
                      csv_model_root: Path | None = None,
                      csv_root: Path = DEFAULT_CSV_ROOT,
                      dispatcher_options: dict | None = None):
    identity = LocalIdentityProvider(IdentitySettings(AuthenticationMode.LOCAL, "development",
                                                      host), subject)
    settings = metadata_store_settings() if settings is None else settings
    store = open_metadata_store(settings)          # no connection until first use
    projects = open_project_repository(settings)
    saved_configs = open_configuration_repository(settings)
    datasets = open_dataset_repository(settings)
    models = open_model_catalog(settings)
    sessions = SQLSessionRegistry()
    options = {"heartbeat_seconds": 30.0, "stale_after": timedelta(minutes=5)}
    options.update(dispatcher_options or {})
    if settings.uses_sql_server:
        options["store_from_env"] = True
        options["store_schema"] = settings.schema or None
    else:
        options["store_url"] = f"sqlite:///{settings.sqlite_path}"
    dispatcher = RunDispatcher(store, log_dir=run_logs, sessions=sessions,
                               artifacts=csv_artifacts, **options)
    submission_options = {} if csv_model_root is None else {"csv_model_root": csv_model_root}
    submission = RunSubmission(store, configurations=configurations or SavedConfigurations(),
                               sessions=sessions, on_submit=dispatcher.wake,
                               **submission_options)
    runs = DurableRunService(store, scope_for=_no_scope)
    project_service = ProjectService(projects)
    configuration_service = ConfigurationService(saved_configs, project_service)

    @asynccontextmanager
    async def lifespan(app):
        store.create_tables()
        projects.create_tables()
        saved_configs.create_tables()
        datasets.create_tables()
        models.create_tables()
        dispatcher.start()
        try:
            yield
        finally:
            dispatcher.stop()
            store.close()
            projects.close()
            saved_configs.close()
            datasets.close()
            models.close()

    app = create_app(runs, current_principal=identity.current_principal,
                     submission=submission, sessions=sessions,
                     projects=project_service,
                     datasets=DatasetService(datasets, project_service, csv_root=csv_root),
                     configurations=configuration_service,
                     lifespan=lifespan)
    app.state.jbg = {"store": store, "dispatcher": dispatcher, "sessions": sessions, "projects": projects, "configurations": saved_configs, "datasets": datasets, "models": models, "configuration_service": configuration_service}
    return app
