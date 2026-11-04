"""285m: safe, stable run overview for durable and HTTP consumers."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
pytest.importorskip("sqlalchemy")

from JBGApplicationContracts import DataSourceKind, RunReference, RunStatus
from JBGDurableRuns import DurableRunService
from JBGRunJobStore import RunJobStore
from JBGRunOverview import project_run_overview


def test_projection_is_strict_and_stage_is_estimated():
    event = SimpleNamespace(percent=0.92, message="sensitive SQL text")
    result = {"kind": "classification", "class_count": 3,
              "mispredicted_count": 1, "private_sql": "secret"}
    data = project_run_overview("abc", RunStatus.RUNNING, [event], result)
    assert data["stage"] == {"name": "finalizing", "precision": "estimated", "percent": .92}
    assert data["results"] == {"available": False, "summary": None}
    assert "secret" not in str(data) and "sensitive" not in str(data)
    done = project_run_overview("abc", RunStatus.SUCCEEDED, [event], result)
    assert done["stage"] == {"name": "succeeded", "precision": "exact", "percent": 1.0}
    assert done["results"]["summary"] == {"kind": "classification", "class_count": 3,
                                             "mispredicted_count": 1}
    assert "secret" not in str(done)


@pytest.mark.parametrize("status,phase", [
    (RunStatus.PENDING, "queued"), (RunStatus.FAILED, "failed"),
    (RunStatus.CANCELLED, "cancelled"), (RunStatus.INTERRUPTED, "interrupted"),
    (RunStatus.AWAITING_CREDENTIALS, "awaiting_credentials"),
])
def test_terminal_or_queued_phases(status, phase):
    projection = project_run_overview("x", status, [])
    assert projection["stage"]["name"] == phase
    assert not projection["results"]["available"]


def test_durable_overview_requires_owner_and_survives_reopen(tmp_path):
    path = tmp_path / "store.sqlite"
    store = RunJobStore.sqlite(path)
    store.create_tables()
    run = store.create("alice", DataSourceKind.SQL, {"spec_version": 1})
    store.record_progress(run, "private table", .55)
    svc = DurableRunService(store, scope_for=lambda _: None)
    with pytest.raises(PermissionError):
        svc.overview("bob", run)
    view = svc.overview("alice", run)
    assert view["stage"]["percent"] == .55
    assert "private table" not in str(view)
    store.close()
    reopened = RunJobStore.sqlite(path)
    reopened_svc = DurableRunService(reopened, scope_for=lambda _: None)
    assert reopened_svc.overview("alice", run)["stage"]["percent"] == .55
    reopened.close()


def test_unknown_result_fields_never_leak():
    projection = project_run_overview("x", RunStatus.SUCCEEDED, [],
                                      {"kind": "classification", "class_count": True,
                                       "mispredicted_count": -2, "token": "private"})
    assert projection["results"]["summary"]["class_count"] is None
    assert projection["results"]["summary"]["mispredicted_count"] is None
    assert "private" not in str(projection)
