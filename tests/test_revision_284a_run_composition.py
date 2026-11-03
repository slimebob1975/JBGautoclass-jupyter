"""284a: per-run composition of the existing application services.

Unit tests use mocks only. The end-to-end class reuses the synthetic native
CSV fixture of revision 281c and is skipped without the full ML dependencies.
"""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from JBGApplicationContracts import DataSourceKind, DatasetReference, RunReference, RunStatus
from JBGDatasetApplicationService import DatasetApplicationService
from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
from JBGPrincipalRunAccess import PrincipalRunAccessService
from JBGRunAPIPresentation import RunAPIPresentationService
from JBGRunComposition import (ComposedRun, LOCAL_CSV_STORAGE, compose_csv_run,
                               compose_run, local_csv_storage)
from JBGRunExecutionService import RunExecutionService
from JBGRunJobService import RunJobService
from JBGRunPreparationService import RunConfiguration, RunPreparationService
from JBGRunProgress import RunProgressService
from Storage.ArtifactContract import StorageScope


def _scope():
    return StorageScope(str(uuid4()), str(uuid4()))


def _local_principal(subject='developer'):
    return AuthenticatedPrincipal(AuthenticationMode.LOCAL, subject)


def _csv_execution(scope, outcome):
    """A mock CSV pipeline whose resolver reports ``scope``."""
    resolver = SimpleNamespace(resolve=Mock(return_value=SimpleNamespace(scope=scope)))
    preparation = RunPreparationService(DatasetApplicationService(csv=resolver),
                                        csv=SimpleNamespace(validate=Mock()))
    runtime = SimpleNamespace(execute=Mock(return_value=outcome))
    return preparation, RunExecutionService(preparation, csv=runtime), runtime


def _prepared(preparation):
    return preparation.prepare(DatasetReference(DataSourceKind.CSV, 'dataset'),
                               RunConfiguration(DataSourceKind.CSV, object()))


# --- RunJobService: optional per-job execution -------------------------------

def test_job_uses_its_own_execution_and_default_remains():
    scope = _scope()
    default_prep, default_exec, default_runtime = _csv_execution(scope, {'run': 'default'})
    own_prep, own_exec, own_runtime = _csv_execution(scope, {'run': 'own'})
    jobs = RunJobService(default_exec)
    composed_job = jobs.create(_prepared(own_prep), execution=own_exec)
    default_job = jobs.create(_prepared(default_prep))
    assert jobs.execute(composed_job) == {'run': 'own'}
    assert jobs.execute(default_job) == {'run': 'default'}
    own_runtime.execute.assert_called_once()
    default_runtime.execute.assert_called_once()
    assert jobs.snapshot(composed_job).status is RunStatus.SUCCEEDED


def test_job_rejects_execution_of_wrong_type():
    prep, execution, runtime = _csv_execution(_scope(), {})
    jobs = RunJobService(execution)
    with pytest.raises(TypeError, match='execution must be RunExecutionService'):
        jobs.create(_prepared(prep), execution=SimpleNamespace(execute=Mock()))
    runtime.execute.assert_not_called()


def test_composed_execution_passes_through_principal_access():
    scope = _scope()
    _, default_exec, default_runtime = _csv_execution(scope, {'run': 'default'})
    own_prep, own_exec, own_runtime = _csv_execution(scope, {'run': 'own'})
    access = PrincipalRunAccessService(
        RunAPIPresentationService(RunJobService(default_exec)), scope_for=lambda p: scope)
    me = _local_principal()
    run = RunReference(access.create(me, _prepared(own_prep), execution=own_exec)['run_id'])
    assert access.execute(me, run)['status'] == 'succeeded'
    own_runtime.execute.assert_called_once()
    default_runtime.execute.assert_not_called()
    with pytest.raises(PermissionError, match='unavailable'):
        access.snapshot(_local_principal('someone-else'), run)


# --- Composition --------------------------------------------------------------

def test_sql_composition_requires_a_session_registry():
    # 284c replaced the explicit SQL rejection with session-bound composition.
    with pytest.raises(TypeError, match='SQLSessionRegistry required'):
        compose_run(DataSourceKind.SQL, sessions=object(), principal=_local_principal(),
                    config=object(), logger=Mock())


def test_compose_run_requires_a_source_kind():
    with pytest.raises(TypeError, match='DataSourceKind'):
        compose_run('csv', repository=Mock(), scope=_scope(),
                    configuration_id='cfg', logger=Mock())


@pytest.mark.parametrize('scope, configuration_id, progress, error', [
    (SimpleNamespace(tenant_id='t', project_id='p'), 'cfg', None, TypeError),
    (None, '', None, ValueError),
    (None, 'cfg', object(), TypeError),
])
def test_compose_csv_run_validates_trusted_inputs(scope, configuration_id, progress, error):
    scope = _scope() if scope is None else scope
    with pytest.raises(error):
        compose_csv_run(Mock(), scope, configuration_id, Mock(), progress=progress)


def test_compose_csv_run_returns_bound_services():
    composed = compose_run(DataSourceKind.CSV, repository=Mock(), scope=_scope(),
                           configuration_id='cfg', logger=Mock(),
                           progress=RunProgressService())
    assert isinstance(composed, ComposedRun)
    assert isinstance(composed.preparation, RunPreparationService)
    assert isinstance(composed.execution, RunExecutionService)


# --- Local storage shared with the Voila GUI -----------------------------------

def test_local_csv_storage_matches_voila_location_and_scope(tmp_path):
    from Storage.LocalCSVScope import local_csv_scope
    _, scope_a = local_csv_storage('Project A', base=tmp_path)
    _, scope_a_again = local_csv_storage('Project A', base=tmp_path)
    _, scope_b = local_csv_storage('Project B', base=tmp_path)
    root = tmp_path / 'output' / 'csv_artifacts'
    assert LOCAL_CSV_STORAGE == Path('output') / 'csv_artifacts'
    assert (root / 'objects').is_dir()
    assert (root / '.local-tenant-id').is_file()
    assert scope_a == scope_a_again == local_csv_scope(root, 'Project A')
    assert scope_b != scope_a
    assert scope_b.tenant_id == scope_a.tenant_id


def test_local_csv_storage_requires_a_project(tmp_path):
    with pytest.raises(ValueError, match='Select a project'):
        local_csv_storage('  ', base=tmp_path)


# --- End to end with the real native ML engine ----------------------------------

# Import the module, not the class, so pytest does not collect 281c's tests here.
import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402


class CSVRunComposition284a(revision_281c.CSVApplicationEndToEnd281c):
    # Inherited probes are exercised by revisions 281c/281f themselves.
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def _inputs(self):
        reference = DatasetReference(DataSourceKind.CSV, self.ref.artifact_id)
        return reference, RunConfiguration(DataSourceKind.CSV, self.config)

    def _guards(self, stack):
        stack.enter_context(patch('socket.create_connection',
                                  side_effect=AssertionError('Network forbidden')))
        import pyodbc
        stack.enter_context(patch.object(pyodbc, 'connect',
                                         side_effect=AssertionError('SQL connection forbidden')))
        from SQLDataLayer import DataLayer as SQLDataLayer
        stack.enter_context(patch.object(SQLDataLayer, '__init__',
                                         side_effect=AssertionError('SQL layer forbidden')))
        stack.enter_context(patch('JBGTaskRunner.send_email', return_value=False))

    def test_composed_csv_run_trains_through_principal_access(self):
        model_path = self._configure_training()
        progress = RunProgressService()
        composed = compose_run(DataSourceKind.CSV, repository=self.repo, scope=self.scope,
                               configuration_id=self.roles.configuration_id,
                               logger=self.logger, progress=progress)
        # The default execution is bound to a foreign scope: using it would fail.
        unused = compose_run(DataSourceKind.CSV, repository=self.repo, scope=_scope(),
                             configuration_id=self.roles.configuration_id, logger=self.logger)
        jobs = RunJobService(unused.execution, progress=progress)
        access = PrincipalRunAccessService(RunAPIPresentationService(jobs),
                                           scope_for=lambda p: self.scope)
        me = _local_principal()
        prepared = composed.preparation.prepare(*self._inputs())
        run = RunReference(access.create(me, prepared, execution=composed.execution)['run_id'])
        with ExitStack() as stack:
            self._guards(stack)
            snapshot = access.execute(me, run)
        self.assertEqual(snapshot['status'], 'succeeded')
        self.assertTrue(model_path.is_file(), 'Composed run did not save a model')
        self.assertTrue(access.result(me, run)['available'])
        events = jobs.progress_history(run)
        self.assertTrue(events, 'No engine progress recorded for the composed job')
        self.assertEqual(events[0].message, 'Starting up ...')

    def test_composed_run_cannot_reach_another_projects_dataset(self):
        self._configure_training()
        foreign = compose_run(DataSourceKind.CSV, repository=self.repo, scope=_scope(),
                              configuration_id=self.roles.configuration_id, logger=self.logger)
        with self.assertRaises((FileNotFoundError, ValueError)):
            foreign.preparation.prepare(*self._inputs())

        composed = compose_run(DataSourceKind.CSV, repository=self.repo, scope=self.scope,
                               configuration_id=self.roles.configuration_id, logger=self.logger)
        prepared = composed.preparation.prepare(*self._inputs())
        access = PrincipalRunAccessService(
            RunAPIPresentationService(RunJobService(composed.execution)),
            scope_for=lambda p: _scope())
        with self.assertRaises(PermissionError):
            access.create(_local_principal(), prepared, execution=composed.execution)
        self.assertFalse(Path(self.config.get_model_filename()).exists())
