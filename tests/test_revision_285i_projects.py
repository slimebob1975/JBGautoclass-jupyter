"""285i: durable project metadata, identity isolation, uniqueness and lifecycle."""
from datetime import timezone
from uuid import UUID

import pytest

pytest.importorskip("sqlalchemy")

from JBGApplicationContracts import DataSourceKind
from JBGMetadataStore import MetadataStoreSettings, open_project_repository
from JBGProjects import DuplicateProjectName


def _open(tmp_path):
    settings = MetadataStoreSettings(sqlite_path=tmp_path / "meta" / "jobs.sqlite")
    return open_project_repository(settings, create_tables=True)


def test_create_persists_and_lists_in_owner_scope_after_reopen(tmp_path):
    repo = _open(tmp_path)
    project = repo.create("alice", " Iris ", DataSourceKind.SQL, "research")
    UUID(project.project_id)
    assert project.name == "Iris" and project.description == "research"
    assert project.created_at.tzinfo == timezone.utc
    assert project.source == DataSourceKind.SQL
    assert repo.list_for_owner("bob") == ()
    with pytest.raises(KeyError):
        repo.get_owned("bob", project.project_id)
    repo.close()
    reopened = _open(tmp_path)
    assert reopened.get_owned("alice", project.project_id).name == "Iris"
    assert len(reopened.list_for_owner("alice")) == 1
    reopened.close()


def test_case_insensitive_uniqueness_per_owner_and_atomic_rename(tmp_path):
    repo = _open(tmp_path)
    a = repo.create("alice", "Iris", DataSourceKind.CSV)
    b = repo.create("alice", "Wine", DataSourceKind.SQL)
    repo.create("bob", "iris", DataSourceKind.SQL)
    with pytest.raises(DuplicateProjectName):
        repo.create("alice", " iRiS ", DataSourceKind.SQL)
    with pytest.raises(DuplicateProjectName):
        repo.update("alice", b.project_id, name="IRIS")
    assert repo.get_owned("alice", b.project_id).name == "Wine"
    newer = repo.update("alice", a.project_id, name="Iris-2", description="updated")
    assert newer.source == DataSourceKind.CSV
    assert newer.updated_at >= a.updated_at
    assert newer.description == "updated"
    repo.close()


def test_foreign_mutations_and_invalid_inputs_do_not_change_records(tmp_path):
    repo = _open(tmp_path)
    p = repo.create("alice", "Demo", DataSourceKind.SQL)
    with pytest.raises(KeyError):
        repo.update("bob", p.project_id, name="stolen")
    with pytest.raises(ValueError):
        repo.create("alice", " N/A " * 100, DataSourceKind.CSV)
    with pytest.raises(TypeError):
        repo.create("alice", "Bad", "sql")
    with pytest.raises(ValueError):
        repo.update("alice", p.project_id, name=" ")
    assert repo.get_owned("alice", p.project_id).name == "Demo"
    assert repo.list_for_owner("alice", limit=1)[0].project_id == p.project_id
    repo.close()
