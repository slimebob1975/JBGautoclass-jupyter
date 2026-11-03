"""Per-run composition of the existing application services (revision 284a).

The resolvers, validators and runtimes are bound to one storage scope (and, for
CSV, one saved column-role configuration) when they are constructed. A single
graph built at server start can therefore not serve several projects. This
module builds that graph per run from trusted, server-side inputs; it adds no
ML, validation or authorization rules of its own.

SQL runs (revision 284c) use a DataLayer opened by JBGSQLSessions with the
requesting user's own credentials in the current server session; pass the same
registry's ``authorize`` as PrincipalRunAccessService(authorize_dataset=...).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from JBGApplicationContracts import DataSourceKind
from JBGDatasetApplicationService import DatasetApplicationService
from JBGDatasetResolvers import ConfiguredSQLDatasetResolver, ScopedCSVDatasetResolver
from JBGRunConfigurationValidators import (CSVRunConfigurationValidator,
                                           SQLRunConfigurationValidator)
from JBGRunExecutionService import (RegisteredCSVClassifierRuntime, RunExecutionService,
                                    SQLClassifierRuntime)
from JBGRunPreparationService import RunPreparationService
from JBGRunProgress import ProgressObservingLogger, RunProgressService
from JBGSQLSessions import SQLSessionRegistry
from Storage.ArtifactContract import StorageScope

# Same location as the Voila GUI (GUI/Widgets.py), relative to the working directory.
LOCAL_CSV_STORAGE = Path("output") / "csv_artifacts"


def local_csv_storage(project: str, *, base: str | Path | None = None):
    """Return the repository and scope the Voila GUI uses for ``project``.

    Local, single-installation storage only; this is NOT authorization. The
    project name is the project identity until a Project entity exists, so a
    renamed project resolves to a different, empty scope (as in Voila).
    """
    from Storage.FileArtifactRepository import FileArtifactRepository
    from Storage.LocalCSVScope import local_csv_scope

    root = Path(base if base is not None else Path.cwd()) / LOCAL_CSV_STORAGE
    return FileArtifactRepository(root / "objects"), local_csv_scope(root, project)


@dataclass(frozen=True)
class ComposedRun:
    """Preparation and execution services bound to one scope and configuration.

    For SQL, ``configuration`` is the run's own Config copy that carries the
    session credentials: use it as the RunConfiguration payload and never
    persist it. For CSV it is None; the caller supplies its own Config.
    """

    preparation: RunPreparationService
    execution: RunExecutionService
    configuration: object = None

    def sql_dataset_key(self) -> str:
        """Dataset key for an SQL run: the configured table of ``configuration``."""
        if self.configuration is None:
            raise ValueError("Not an SQL composition")
        return self.configuration.get_data_table(include_database=True)


def compose_csv_run(repository, scope: StorageScope, configuration_id: str, logger, *,
                    progress: RunProgressService | None = None) -> ComposedRun:
    """Wire the existing CSV resolver, validator and runtime for one run.

    ``scope`` and ``configuration_id`` must come from the trusted server side,
    never from a client request. Pass the job service's ``progress`` so that
    engine progress is recorded for the job being executed.
    """
    if not isinstance(scope, StorageScope):
        raise TypeError("trusted StorageScope required")
    if not isinstance(configuration_id, str) or not configuration_id.strip():
        raise ValueError("configuration_id required")
    logger = _observed(logger, progress)
    datasets = DatasetApplicationService(csv=ScopedCSVDatasetResolver(repository, scope))
    preparation = RunPreparationService(datasets, csv=CSVRunConfigurationValidator())
    execution = RunExecutionService(
        preparation,
        csv=RegisteredCSVClassifierRuntime(repository, scope, configuration_id, logger))
    return ComposedRun(preparation, execution)


def compose_sql_run(sessions: SQLSessionRegistry, principal, config, logger, *,
                    progress: RunProgressService | None = None) -> ComposedRun:
    """Wire the existing SQL resolver, validator and runtime for one run.

    The DataLayer is opened by ``sessions`` with ``principal``'s own credentials
    on a copy of ``config``; ``config`` itself is not modified. Raises
    PermissionError when the principal has not signed in to SQL.
    """
    if not isinstance(sessions, SQLSessionRegistry):
        raise TypeError("SQLSessionRegistry required")
    logger = _observed(logger, progress)
    layer = sessions.data_layer_for(principal, config, logger)
    datasets = DatasetApplicationService(sql=ConfiguredSQLDatasetResolver(layer, layer.config))
    preparation = RunPreparationService(datasets, sql=SQLRunConfigurationValidator())
    execution = RunExecutionService(preparation, sql=SQLClassifierRuntime(logger))
    return ComposedRun(preparation, execution, configuration=layer.config)


def compose_run(source: DataSourceKind, **arguments) -> ComposedRun:
    """Dispatch on the project's source type.

    CSV takes ``repository, scope, configuration_id, logger``; SQL takes
    ``sessions, principal, config, logger``; both accept ``progress``.
    """
    if not isinstance(source, DataSourceKind):
        raise TypeError("source must be a DataSourceKind")
    if source is DataSourceKind.SQL:
        return compose_sql_run(**arguments)
    return compose_csv_run(**arguments)


def _observed(logger, progress: RunProgressService | None):
    if progress is None:
        return logger
    if not isinstance(progress, RunProgressService):
        raise TypeError("progress must be RunProgressService")
    return ProgressObservingLogger(logger, progress)
