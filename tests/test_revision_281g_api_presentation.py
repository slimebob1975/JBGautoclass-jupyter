"""281g: API-facing run observations must not leak internal ML objects."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from JBGApplicationContracts import DataSourceKind, DatasetReference, RunReference
from JBGDatasetApplicationService import DatasetApplicationService
from JBGRunAPIPresentation import RunAPIPresentationService
from JBGRunExecutionService import RunExecutionService
from JBGRunJobService import RunJobService
from JBGRunPreparationService import RunConfiguration, RunPreparationService
from JBGRunProgress import RunProgressService


def make_facade(*, result=None, error=None):
    datasets = DatasetApplicationService(sql=SimpleNamespace(resolve=Mock(return_value=object())))
    preparation = RunPreparationService(datasets, sql=SimpleNamespace(validate=Mock()))
    prepared = preparation.prepare(DatasetReference(DataSourceKind.SQL, 'trusted'),
                                   RunConfiguration(DataSourceKind.SQL, object()))
    runtime = SimpleNamespace(execute=Mock(return_value=result, side_effect=error))
    execution = RunExecutionService(preparation, sql=runtime)
    progress = RunProgressService()
    jobs = RunJobService(execution, progress=progress)
    return RunAPIPresentationService(jobs), jobs, progress, prepared, runtime


def test_success_returns_only_safe_status_and_availability():
    secret = {'model': object(), 'password': 'top-secret', 'path': 'C:/private/file'}
    api, jobs, _, prepared, runtime = make_facade(result=secret)
    created = api.create('alice', prepared)
    run = RunReference(created['run_id'])
    assert created == {'run_id': run.key, 'status': 'pending', 'has_error': False}
    assert api.execute('alice', run)['status'] == 'succeeded'
    assert api.result('alice', run) == {'run_id': run.key, 'available': True}
    assert jobs.result(run) is secret
    assert 'top-secret' not in str(api.snapshot('alice', run))
    runtime.execute.assert_called_once()


def test_failed_run_hides_engine_exception_but_preserves_internal_error():
    failure = RuntimeError('password=secret database=C:/sensitive/path')
    api, jobs, _, prepared, _ = make_facade(error=failure)
    run = RunReference(api.create('alice', prepared)['run_id'])
    assert api.execute('alice', run) == {'run_id': run.key, 'status': 'failed', 'has_error': True}
    assert api.result('alice', run)['available'] is False
    assert jobs.error(run) is failure
    assert 'secret' not in str(api.snapshot('alice', run))


def test_cross_owner_and_unknown_job_never_exposed():
    api, _, _, prepared, runtime = make_facade(result=42)
    run = RunReference(api.create('alice', prepared)['run_id'])
    for ref in (run, RunReference('made-up')):
        for action in (api.snapshot, api.execute, api.progress, api.result):
            with pytest.raises(PermissionError, match='unavailable'):
                action('bob', ref)
    runtime.execute.assert_not_called()


def test_missing_principal_is_rejected():
    api, _, _, prepared, _ = make_facade()
    with pytest.raises(PermissionError, match='authentication'):
        api.create('', prepared)
    run = RunReference(api.create('alice', prepared)['run_id'])
    with pytest.raises(PermissionError, match='authentication'):
        api.snapshot(None, run)


def test_progress_exposes_only_timestamp_and_percent_not_raw_message():
    api, jobs, progress, prepared, _ = make_facade()
    run = RunReference(api.create('alice', prepared)['run_id'])
    progress.record(run, message='C:/secret/password.txt', percent=0.3)
    events = api.progress('alice', run)
    assert len(events) == 1
    assert set(events[0]) == {'run_id', 'timestamp', 'percent'}
    assert events[0]['percent'] == 0.3
    assert 'secret' not in str(events)
    assert jobs.latest_progress(run).message == 'C:/secret/password.txt'


def test_external_job_not_registered_through_facade_is_inaccessible():
    api, jobs, _, prepared, _ = make_facade()
    internal = jobs.create(prepared)
    with pytest.raises(PermissionError, match='unavailable'):
        api.snapshot('alice', internal)
