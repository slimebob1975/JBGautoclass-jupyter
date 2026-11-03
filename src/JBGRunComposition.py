"""Per-run composition of the existing application services (revision 284a).

The resolvers, validators and runtimes are bound to one storage scope (and, for
CSV, one saved column-role configuration) when they are constructed. A single
graph built at server start can therefore not serve several projects. This
module builds that graph per run from trusted, server-side inputs; it adds no
ML, validation or authorization rules of its own.

Only CSV is supported. SQL composition needs a DataLayer opened with the
requesting user's own credentials in the current session (BACKLOG design gap 8)
and is rejected explicitly until then.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from JBGApplicationContracts import DataSourceKind
from JBGDatasetApplicationService import DatasetApplicationService
from JBGDatasetResolvers import ScopedCSVDatasetResolver
from JBGRunConfigurationValidators import CSVRunConfigurationValidator
from JBGRunExecutionService import RegisteredCSVClassifierRuntime, RunExecutionService
from JBGRunPreparationService import RunPreparationService
from JBGRunProgress import ProgressObservingLogger, RunProgressService
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
    """Preparation and execution services bound to one scope and configuration."""

    preparation: RunPreparationService
    execution: RunExecutionService


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
    if progress is not None:
        if not isinstance(progress, RunProgressService):
            raise TypeError("progress must be RunProgressService")
        logger = ProgressObservingLogger(logger, progress)
    datasets = DatasetApplicationService(csv=ScopedCSVDatasetResolver(repository, scope))
    preparation = RunPreparationService(datasets, csv=CSVRunConfigurationValidator())
    execution = RunExecutionService(
        preparation,
        csv=RegisteredCSVClassifierRuntime(repository, scope, configuration_id, logger))
    return ComposedRun(preparation, execution)


def compose_run(source: DataSourceKind, *, repository, scope: StorageScope,
                configuration_id: str, logger,
                progress: RunProgressService | None = None) -> ComposedRun:
    """Dispatch on the project's source type; SQL is not supported yet."""
    if not isinstance(source, DataSourceKind):
        raise TypeError("source must be a DataSourceKind")
    if source is DataSourceKind.SQL:
        raise NotImplementedError(
            "SQL run composition requires a session-bound SQL data layer "
            "opened with the user's own login (design gap 8); not yet available")
    return compose_csv_run(repository, scope, configuration_id, logger, progress=progress)
