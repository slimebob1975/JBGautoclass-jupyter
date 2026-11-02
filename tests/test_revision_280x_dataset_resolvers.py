"""Revision 280x: real storage contracts, no GUI or live SQL server."""
from dataclasses import dataclass
from uuid import uuid4
from unittest.mock import Mock
import pytest

from JBGApplicationContracts import DataSourceKind, DatasetReference
from JBGDatasetApplicationService import DatasetApplicationService
from JBGDatasetResolvers import ScopedCSVDatasetResolver, ConfiguredSQLDatasetResolver
from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository
from io import BytesIO


def new_id():
    return str(uuid4())


def test_registered_csv_resolves_actual_repository_reference(tmp_path):
    scope = StorageScope(new_id(), new_id())
    repo = FileArtifactRepository(tmp_path)
    ref = ArtifactReference(scope, new_id(), ArtifactKind.DATASET, "text/csv", dataset_id=new_id())
    repo.save_artifact(scope, ref, BytesIO(b"x,class\n1,yes\n"))
    service = DatasetApplicationService(csv=ScopedCSVDatasetResolver(repo, scope))
    result = service.resolve(DatasetReference(DataSourceKind.CSV, ref.artifact_id))
    assert result == ref


def test_csv_other_scope_and_unknown_id_fail_closed(tmp_path):
    scope = StorageScope(new_id(), new_id())
    other = StorageScope(new_id(), new_id())
    repo = FileArtifactRepository(tmp_path)
    ref = ArtifactReference(other, new_id(), ArtifactKind.DATASET, "text/csv", dataset_id=new_id())
    repo.save_artifact(other, ref, BytesIO(b"x\n1\n"))
    service = DatasetApplicationService(csv=ScopedCSVDatasetResolver(repo, scope))
    with pytest.raises((FileNotFoundError, ValueError)):
        service.resolve(DatasetReference(DataSourceKind.CSV, ref.artifact_id))
    with pytest.raises((FileNotFoundError, ValueError)):
        service.resolve(DatasetReference(DataSourceKind.CSV, new_id()))


def test_csv_rejects_model_artifact_and_no_dataset_id(tmp_path):
    scope = StorageScope(new_id(), new_id())
    repo = FileArtifactRepository(tmp_path)
    ref = ArtifactReference(scope, new_id(), ArtifactKind.MODEL, "application/octet-stream", dataset_id=new_id())
    repo.save_artifact(scope, ref, BytesIO(b"model"))
    service = DatasetApplicationService(csv=ScopedCSVDatasetResolver(repo, scope))
    with pytest.raises(ValueError, match="registered CSV"):
        service.resolve(DatasetReference(DataSourceKind.CSV, ref.artifact_id))


def test_csv_rejects_repository_reference_with_wrong_scope_or_id():
    scope = StorageScope(new_id(), new_id())
    other = StorageScope(new_id(), new_id())
    key = new_id()
    for ref in (ArtifactReference(other, key, ArtifactKind.DATASET, "text/csv", dataset_id=new_id()),
                ArtifactReference(scope, new_id(), ArtifactKind.DATASET, "text/csv", dataset_id=new_id())):
        repo = Mock()
        repo.get_reference.return_value = ref
        with pytest.raises(ValueError):
            ScopedCSVDatasetResolver(repo, scope).resolve(key)
        repo.get_reference.assert_called_once_with(scope, key)


def test_csv_requires_trusted_scope():
    with pytest.raises(TypeError):
        ScopedCSVDatasetResolver(Mock(), "not a scope")


def test_sql_uses_existing_layer_without_querying():
    config = Mock()
    config.get_data_table.return_value = "[db].[dbo].[iris]"
    layer = Mock()
    service = DatasetApplicationService(sql=ConfiguredSQLDatasetResolver(layer, config))
    assert service.resolve(DatasetReference(DataSourceKind.SQL, "[db].[dbo].[iris]")) is layer
    config.get_data_table.assert_called_once_with(include_database=True)
    layer.assert_not_called()


def test_sql_rejects_wrong_table_without_accessing_layer():
    config = Mock()
    config.get_data_table.return_value = "[db].[dbo].[iris]"
    layer = Mock()
    resolver = ConfiguredSQLDatasetResolver(layer, config)
    with pytest.raises(ValueError):
        resolver.resolve("[db].[dbo].[other]")
    layer.assert_not_called()


def test_no_source_fallback_between_sql_and_csv(tmp_path):
    scope = StorageScope(new_id(), new_id())
    repo = FileArtifactRepository(tmp_path)
    config = Mock()
    config.get_data_table.return_value = "[db].[dbo].[iris]"
    layer = Mock()
    service = DatasetApplicationService(sql=ConfiguredSQLDatasetResolver(layer, config),
                                        csv=ScopedCSVDatasetResolver(repo, scope))
    with pytest.raises((FileNotFoundError, ValueError)):
        service.resolve(DatasetReference(DataSourceKind.CSV, "[db].[dbo].[iris]"))
    layer.assert_not_called()
