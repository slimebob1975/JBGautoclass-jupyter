"""285l model metadata must not change durable run storage or artifact I/O."""
from uuid import uuid4
from unittest.mock import patch

import pytest
pytest.importorskip("sqlalchemy")

from JBGApplicationContracts import DataSourceKind, RunStatus
from JBGDatasets import DatasetRole
from JBGMetadataStore import (MetadataStoreSettings, open_dataset_repository,
                              open_metadata_store, open_model_catalog,
                              open_project_repository)
from JBGModelCatalog import ModelState


def _repo(tmp_path):
    settings = MetadataStoreSettings(sqlite_path=tmp_path / "metadata.sqlite")
    jobs = open_metadata_store(settings, create_tables=True)
    projects = open_project_repository(settings, create_tables=True)
    datasets = open_dataset_repository(settings, create_tables=True)
    models = open_model_catalog(settings, create_tables=True)
    return settings, jobs, projects, datasets, models


def test_model_catalog_persists_and_links_successful_run(tmp_path):
    settings, jobs, projects, datasets, models = _repo(tmp_path)
    project = projects.create("alice", "Iris", DataSourceKind.SQL)
    dataset = datasets.create("alice", project.project_id, "train", DatasetRole.TRAINING,
                              catalog="AIdata", table="jbg_ai.iris")
    run = jobs.create("alice", DataSourceKind.SQL, {"spec_version": 1})
    assert jobs.claim(run, "worker")
    assert jobs.finish(run, "worker", RunStatus.SUCCEEDED)
    model = models.register("alice", project.project_id, "LDA", str(uuid4()),
                            dataset_id=dataset.dataset_id, run_id=run.key,
                            compatibility_key="sha256:test")
    assert model.run_id == run.key
    assert model.dataset_id == dataset.dataset_id
    assert model.state is ModelState.REGISTERED
    assert models.set_state("alice", project.project_id, model.model_id,
                            ModelState.AVAILABLE).state is ModelState.AVAILABLE
    models.close()
    models = open_model_catalog(settings)
    assert models.get_owned("alice", project.project_id, model.model_id).state is ModelState.AVAILABLE
    assert len(models.list_for_project("alice", project.project_id)) == 1
    models.close(); datasets.close(); projects.close(); jobs.close()


def test_owner_source_dataset_and_run_validation(tmp_path):
    _, jobs, projects, datasets, models = _repo(tmp_path)
    a = projects.create("alice", "A", DataSourceKind.SQL)
    b = projects.create("bob", "B", DataSourceKind.SQL)
    foreign_ds = datasets.create("bob", b.project_id, "b", DatasetRole.TRAINING,
                                 catalog="db", table="dbo.t")
    failed = jobs.create("alice", DataSourceKind.SQL, {"spec_version": 1})
    good = models.register("alice", a.project_id, "baseline", str(uuid4()))
    with pytest.raises(KeyError):
        models.get_owned("bob", a.project_id, good.model_id)
    with pytest.raises(KeyError):
        models.list_for_project("bob", a.project_id)
    with pytest.raises(KeyError):
        models.set_state("bob", a.project_id, good.model_id, ModelState.RETIRED)
    with pytest.raises(ValueError):
        models.register("alice", a.project_id, "foreign dataset", str(uuid4()),
                        dataset_id=foreign_ds.dataset_id)
    with pytest.raises(ValueError):
        models.register("alice", a.project_id, "failed run", str(uuid4()), run_id=failed.key)
    with pytest.raises(ValueError):
        models.register("alice", a.project_id, "duplicate", good.artifact_key)
    with pytest.raises(ValueError):
        models.register("alice", a.project_id, "BASELINE", str(uuid4()))
    with pytest.raises(ValueError):
        models.register("alice", a.project_id, "unsafe", "C:/private/model.pkl")
    assert jobs.get(failed).status is RunStatus.PENDING
    models.close(); datasets.close(); projects.close(); jobs.close()


def test_source_mismatch_and_metadata_only(tmp_path):
    _, jobs, projects, datasets, models = _repo(tmp_path)
    csv = projects.create("alice", "CSV", DataSourceKind.CSV)
    sql_run = jobs.create("alice", DataSourceKind.SQL, {"spec_version": 1})
    jobs.claim(sql_run, "worker")
    jobs.finish(sql_run, "worker", RunStatus.SUCCEEDED)
    with pytest.raises(ValueError):
        models.register("alice", csv.project_id, "incorrect source", str(uuid4()),
                        run_id=sql_run.key)
    model = models.register("alice", csv.project_id, "csv", str(uuid4()))
    models.set_state("alice", csv.project_id, model.model_id, ModelState.MISSING)
    assert models.get_owned("alice", csv.project_id, model.model_id).state is ModelState.MISSING
    assert jobs.get(sql_run).status is RunStatus.SUCCEEDED
    models.close(); datasets.close(); projects.close(); jobs.close()


def test_legacy_run_tables_are_untouched(tmp_path):
    from sqlalchemy import inspect
    settings, jobs, projects, datasets, models = _repo(tmp_path)
    inspector = inspect(jobs._engine)
    assert {"jbg_run_jobs", "jbg_run_events", "jbg_models"}.issubset(set(inspector.get_table_names()))
    assert "project_id" not in {column["name"] for column in inspector.get_columns("jbg_run_jobs")}
    models.close(); datasets.close(); projects.close(); jobs.close()
