"""The new application boundary can be imported without GUI or ML runtimes."""

from dataclasses import FrozenInstanceError, asdict
import importlib
import sys

import pytest


def test_contract_module_does_not_import_gui_or_ml_engine():
    module = importlib.import_module("JBGApplicationContracts")
    assert module is not None
    source = open(module.__file__, encoding="utf-8").read()
    assert "from GUI" not in source
    assert "import JBGHandler" not in source
    assert "import AutomaticClassifier" not in source


def test_source_values_match_existing_backend_kinds():
    from JBGApplicationContracts import DataSourceKind
    assert DataSourceKind.SQL.value == "sql"
    assert DataSourceKind.CSV.value == "csv"


def test_dataset_identity_is_source_scoped_and_immutable():
    from JBGApplicationContracts import DataSourceKind, DatasetReference
    sql = DatasetReference(DataSourceKind.SQL, "same-key")
    csv = DatasetReference(DataSourceKind.CSV, "same-key")
    assert sql != csv
    assert asdict(sql) == {"source": DataSourceKind.SQL, "key": "same-key"}
    with pytest.raises(FrozenInstanceError):
        sql.key = "other"


@pytest.mark.parametrize("key", ["", "   ", None, 123])
def test_dataset_key_must_be_nonempty_text(key):
    from JBGApplicationContracts import DataSourceKind, DatasetReference
    with pytest.raises((TypeError, ValueError)):
        DatasetReference(DataSourceKind.CSV, key)


def test_invalid_source_is_rejected_without_normalization():
    from JBGApplicationContracts import DatasetReference
    with pytest.raises(TypeError):
        DatasetReference("csv", "dataset")


def test_run_snapshot_is_transport_neutral_and_immutable():
    from JBGApplicationContracts import RunReference, RunSnapshot, RunStatus
    snapshot = RunSnapshot(RunReference("job-1"), RunStatus.PENDING)
    assert snapshot.message is None
    assert snapshot.status.value == "pending"
    with pytest.raises(FrozenInstanceError):
        snapshot.status = RunStatus.RUNNING


def test_run_status_values_are_stable():
    from JBGApplicationContracts import RunStatus
    # 285a deliberately added "interrupted" (agreed restart behaviour).
    assert {v.value for v in RunStatus} == {
        "pending", "running", "succeeded", "failed", "cancelled", "interrupted"
    }


def test_run_reference_and_snapshot_reject_invalid_values():
    from JBGApplicationContracts import RunReference, RunSnapshot, RunStatus
    with pytest.raises(ValueError):
        RunReference("  ")
    with pytest.raises(TypeError):
        RunSnapshot("job-1", RunStatus.PENDING)
    with pytest.raises(TypeError):
        RunSnapshot(RunReference("job-1"), "running")
