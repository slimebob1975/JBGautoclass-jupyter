"""285k: project-owned SQL/CSV dataset registration and active training context."""
from uuid import uuid4

import pytest
pytest.importorskip("sqlalchemy")

from JBGApplicationContracts import DataSourceKind
from JBGDatasets import (DatasetRole, DatasetValidationState,
                         DuplicateDatasetName)
from JBGMetadataStore import (MetadataStoreSettings, open_dataset_repository,
                              open_project_repository)


def _repos(tmp_path):
    settings = MetadataStoreSettings(sqlite_path=tmp_path / "metadata.sqlite")
    projects = open_project_repository(settings, create_tables=True)
    datasets = open_dataset_repository(settings, create_tables=True)
    return projects, datasets, settings


def test_sql_registration_persists_and_is_unverified(tmp_path):
    projects, datasets, settings = _repos(tmp_path)
    p = projects.create("alice", "Iris", DataSourceKind.SQL)
    ds = datasets.create("alice", p.project_id, "Training", DatasetRole.MIXED,
                         catalog="AIdata", table="jbg_ai.iris")
    assert ds.catalog == "AIdata" and ds.table == "jbg_ai.iris"
    assert ds.validation_state is DatasetValidationState.UNVERIFIED
    assert not ds.active_training
    datasets.close()
    datasets = open_dataset_repository(settings)
    assert datasets.get_owned("alice", p.project_id, ds.dataset_id) == ds
    datasets.close()
    projects.close()


def test_csv_requires_opaque_registration_and_source_is_locked(tmp_path):
    projects, datasets, _ = _repos(tmp_path)
    p = projects.create("alice", "CSV", DataSourceKind.CSV)
    ident = str(uuid4())
    ds = datasets.create("alice", p.project_id, "CSV", DatasetRole.TRAINING,
                         registration_id=ident)
    assert ds.registration_id == ident and ds.catalog is None
    with pytest.raises(ValueError):
        datasets.create("alice", p.project_id, "No path", DatasetRole.TRAINING,
                        registration_id="C:/private/data.csv")
    with pytest.raises(ValueError):
        datasets.create("alice", p.project_id, "Wrong kind", DatasetRole.TRAINING,
                        catalog="db", table="table")
    sql = projects.create("alice", "SQL", DataSourceKind.SQL)
    with pytest.raises(ValueError):
        datasets.create("alice", sql.project_id, "CSV in SQL", DatasetRole.TRAINING,
                        registration_id=ident)
    datasets.close()
    projects.close()


def test_owner_and_project_scopes(tmp_path):
    projects, datasets, _ = _repos(tmp_path)
    mine = projects.create("alice", "Mine", DataSourceKind.SQL)
    others = projects.create("bob", "Other", DataSourceKind.SQL)
    ds = datasets.create("alice", mine.project_id, "A", DatasetRole.TRAINING,
                         catalog="db", table="dbo.t")
    for owner, project in [("bob", mine.project_id), ("alice", others.project_id)]:
        with pytest.raises(KeyError):
            datasets.get_owned(owner, project, ds.dataset_id)
        with pytest.raises(KeyError):
            datasets.list_for_project(owner, project)
        with pytest.raises(KeyError):
            datasets.activate_training(owner, project, ds.dataset_id)
        with pytest.raises(KeyError):
            datasets.set_validation(owner, project, ds.dataset_id, DatasetValidationState.VALID)
    assert not datasets.get_owned("alice", mine.project_id, ds.dataset_id).active_training
    datasets.close()
    projects.close()


def test_one_active_training_dataset_and_prediction_rejected(tmp_path):
    projects, datasets, _ = _repos(tmp_path)
    p = projects.create("alice", "P", DataSourceKind.SQL)
    a = datasets.create("alice", p.project_id, "A", DatasetRole.TRAINING,
                        catalog="db", table="dbo.a")
    b = datasets.create("alice", p.project_id, "B", DatasetRole.MIXED,
                        catalog="db", table="dbo.b")
    c = datasets.create("alice", p.project_id, "C", DatasetRole.PREDICTION,
                        catalog="db", table="dbo.c")
    datasets.activate_training("alice", p.project_id, a.dataset_id)
    assert datasets.get_owned("alice", p.project_id, a.dataset_id).active_training
    with pytest.raises(ValueError):
        datasets.activate_training("alice", p.project_id, c.dataset_id)
    assert datasets.get_owned("alice", p.project_id, a.dataset_id).active_training
    datasets.activate_training("alice", p.project_id, b.dataset_id)
    assert sum(d.active_training for d in datasets.list_for_project("alice", p.project_id)) == 1
    assert not datasets.get_owned("alice", p.project_id, a.dataset_id).active_training
    datasets.close()
    projects.close()


def test_duplicate_names_validation_and_invalid_inputs(tmp_path):
    projects, datasets, _ = _repos(tmp_path)
    p = projects.create("alice", "P", DataSourceKind.SQL)
    ds = datasets.create("alice", p.project_id, "Iris", DatasetRole.MIXED,
                         catalog="db", table="dbo.iris")
    with pytest.raises(DuplicateDatasetName):
        datasets.create("alice", p.project_id, " iris ", DatasetRole.TRAINING,
                        catalog="db", table="dbo.iris2")
    with pytest.raises(TypeError):
        datasets.create("alice", p.project_id, "bad", "training",
                        catalog="db", table="dbo.iris2")
    with pytest.raises(ValueError):
        datasets.create("alice", p.project_id, "missing", DatasetRole.TRAINING,
                        catalog="N/A", table=" ")
    ds = datasets.set_validation("alice", p.project_id, ds.dataset_id,
                                 DatasetValidationState.VALID)
    assert ds.validation_state is DatasetValidationState.VALID
    with pytest.raises(TypeError):
        datasets.set_validation("alice", p.project_id, ds.dataset_id, "valid")
    assert datasets.get_owned("alice", p.project_id, ds.dataset_id).validation_state is DatasetValidationState.VALID
    datasets.close()
    projects.close()
