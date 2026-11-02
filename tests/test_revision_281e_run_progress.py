"""281e: real logger progress, bounded history and job-local isolation."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from JBGApplicationContracts import DataSourceKind, DatasetReference, RunReference, RunStatus
from JBGDatasetApplicationService import DatasetApplicationService
from JBGRunExecutionService import RunExecutionService
from JBGRunJobService import RunJobService
from JBGRunPreparationService import RunConfiguration, RunPreparationService
from JBGRunProgress import RunProgressService, ProgressObservingLogger


def make_job(runtime_callback, progress=None):
    dataset_service = DatasetApplicationService(sql=SimpleNamespace(resolve=Mock(return_value=object())))
    prep = RunPreparationService(dataset_service, sql=SimpleNamespace(validate=Mock()))
    prepared = prep.prepare(DatasetReference(DataSourceKind.SQL, "table"),
                            RunConfiguration(DataSourceKind.SQL, object()))
    runtime = SimpleNamespace(execute=Mock(side_effect=runtime_callback))
    execution = RunExecutionService(prep, sql=runtime)
    jobs = RunJobService(execution, progress=progress)
    return jobs, jobs.create(prepared)


def test_real_progress_forwarded_and_captured_without_fabrication():
    progress = RunProgressService()
    original = SimpleNamespace(print_progress=Mock(return_value="unchanged"), other=23)
    logger = ProgressObservingLogger(original, progress)
    def execute(_):
        assert logger.print_progress("Starting", 0.0) == "unchanged"
        assert logger.print_progress("Training") == "unchanged"
        assert logger.print_progress("Complete", 1.0) == "unchanged"
        return {"trained": True}
    jobs, ref = make_job(execute, progress)
    assert jobs.latest_progress(ref) is None
    assert jobs.execute(ref) == {"trained": True}
    assert jobs.snapshot(ref).status is RunStatus.SUCCEEDED
    assert [(x.message, x.percent) for x in jobs.progress_history(ref)] == [
        ("Starting", 0.0), ("Training", None), ("Complete", 1.0)]
    assert jobs.latest_progress(ref).percent == 1.0
    assert all(x.run == ref and x.timestamp.tzinfo is not None for x in jobs.progress_history(ref))
    assert original.print_progress.call_count == 3
    assert logger.other == 23
    logger.print_progress("outside", .5)
    assert len(jobs.progress_history(ref)) == 3


def test_failure_keeps_actual_progress_and_propagates_error():
    progress = RunProgressService()
    logger = ProgressObservingLogger(SimpleNamespace(print_progress=Mock()), progress)
    failure = RuntimeError("engine failed")
    def execute(_):
        logger.print_progress("Preparing", .2)
        raise failure
    jobs, ref = make_job(execute, progress)
    with pytest.raises(RuntimeError, match="engine failed") as raised:
        jobs.execute(ref)
    assert raised.value is failure
    assert jobs.snapshot(ref).status is RunStatus.FAILED
    assert jobs.error(ref) is failure
    assert [(e.message, e.percent) for e in jobs.progress_history(ref)] == [("Preparing", .2)]


def test_history_is_bounded_and_immutable_snapshot():
    progress = RunProgressService(max_events=2)
    ref = RunReference("r1")
    progress.register(ref)
    progress.record(ref, "a")
    before = progress.history(ref)
    progress.record(ref, "b", 0)
    progress.record(ref, "c", 1)
    assert [e.message for e in before] == ["a"]
    assert [e.message for e in progress.history(ref)] == ["b", "c"]
    assert progress.latest(ref).message == "c"


def test_unknown_refs_and_bad_values_fail_closed():
    progress = RunProgressService()
    r = RunReference("x")
    with pytest.raises(KeyError):
        progress.record(r, "unknown")
    with pytest.raises(KeyError):
        progress.history(r)
    progress.register(r)
    with pytest.raises(ValueError):
        progress.register(r)
    for value in (-1, 2, float("nan")):
        with pytest.raises(ValueError):
            progress.record(r, "bad", value)
    with pytest.raises(TypeError):
        progress.record(r, 4)
    assert progress.history(r) == ()
    with pytest.raises(ValueError):
        RunProgressService(max_events=0)


def test_isolated_per_job_and_no_invented_events():
    progress = RunProgressService()
    jobs, first = make_job(lambda _: 1, progress)
    assert jobs.execute(first) == 1
    assert jobs.progress_history(first) == ()
    with pytest.raises(KeyError):
        jobs.latest_progress(RunReference("absent"))
    second = RunReference("other")
    progress.register(second)
    progress.record(second, "other job", .6)
    assert jobs.progress_history(first) == ()
    assert progress.latest(second).percent == .6
