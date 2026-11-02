"""281d: GUI-independent synchronous job state and result reporting."""
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from JBGApplicationContracts import DataSourceKind, DatasetReference, RunReference, RunStatus
from JBGDatasetApplicationService import DatasetApplicationService
from JBGRunExecutionService import RunExecutionService
from JBGRunJobService import RunJobService
from JBGRunPreparationService import RunConfiguration, RunPreparationService


def make_service(*, outcome=None, error=None):
    resolved = object()
    datasets = DatasetApplicationService(sql=SimpleNamespace(resolve=Mock(return_value=resolved)))
    preparation = RunPreparationService(datasets, sql=SimpleNamespace(validate=Mock()))
    prepared = preparation.prepare(DatasetReference(DataSourceKind.SQL, 'table'),
                                   RunConfiguration(DataSourceKind.SQL, object()))
    runtime = SimpleNamespace(execute=Mock(return_value=outcome, side_effect=error))
    execution = RunExecutionService(preparation, sql=runtime)
    return RunJobService(execution), prepared, runtime


def test_pending_running_success_and_result():
    started, release = Event(), Event()
    service, prepared, runtime = make_service(outcome={'model': 1})
    def slow(_):
        started.set()
        assert release.wait(5)
        return {'model': 1}
    runtime.execute.side_effect = slow
    ref = service.create(prepared)
    assert service.snapshot(ref).status is RunStatus.PENDING
    assert service.snapshot(ref).message is None
    assert service.error(ref) is None
    with pytest.raises(ValueError, match='not succeeded'):
        service.result(ref)
    returned = []
    worker = Thread(target=lambda: returned.append(service.execute(ref)))
    worker.start()
    try:
        assert started.wait(5)
        assert service.snapshot(ref).status is RunStatus.RUNNING
        with pytest.raises(ValueError, match='already started'):
            service.execute(ref)
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert returned == [{'model': 1}]
    assert service.snapshot(ref).status is RunStatus.SUCCEEDED
    assert service.result(ref) == {'model': 1}
    runtime.execute.assert_called_once()


def test_failed_job_preserves_original_exception_and_identity():
    failure = RuntimeError('training failed')
    service, prepared, runtime = make_service(error=failure)
    ref = service.create(prepared)
    with pytest.raises(RuntimeError, match='training failed') as raised:
        service.execute(ref)
    assert raised.value is failure
    assert service.snapshot(ref).status is RunStatus.FAILED
    assert service.snapshot(ref).message == 'training failed'
    assert service.error(ref) is failure
    with pytest.raises(ValueError, match='not succeeded'):
        service.result(ref)
    with pytest.raises(ValueError, match='already started'):
        service.execute(ref)
    runtime.execute.assert_called_once()


def test_two_jobs_have_independent_references_and_results():
    service, prepared, _ = make_service(outcome=42)
    first, second = service.create(prepared), service.create(prepared)
    assert first != second
    service.execute(first)
    assert service.snapshot(first).status is RunStatus.SUCCEEDED
    assert service.snapshot(second).status is RunStatus.PENDING
    assert service.result(first) == 42


def test_no_result_is_still_a_successful_result():
    service, prepared, _ = make_service(outcome=None)
    ref = service.create(prepared)
    assert service.execute(ref) is None
    assert service.result(ref) is None
    assert service.snapshot(ref).status is RunStatus.SUCCEEDED


def test_unknown_and_invalid_references_fail_closed():
    service, prepared, runtime = make_service(outcome=1)
    with pytest.raises(KeyError, match='unknown'):
        service.snapshot(RunReference('unknown'))
    with pytest.raises(KeyError, match='unknown'):
        service.execute(RunReference('unknown'))
    with pytest.raises(TypeError, match='RunReference'):
        service.result('unknown')
    runtime.execute.assert_not_called()


def test_invalid_prepared_run_rejected_before_job_creation():
    service, _, _ = make_service()
    with pytest.raises(TypeError, match='PreparedRun'):
        service.create(object())


def test_job_forwards_execution_validation_failure():
    service, prepared, runtime = make_service(outcome=1)
    # PreparedRun is frozen but can still be deliberately tampered with by a caller.
    object.__setattr__(prepared, 'dataset', DatasetReference(DataSourceKind.CSV, 'foreign'))
    ref = service.create(prepared)
    with pytest.raises(ValueError, match='source mismatch'):
        service.execute(ref)
    assert service.snapshot(ref).status is RunStatus.FAILED
    runtime.execute.assert_not_called()
