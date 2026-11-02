"""280z: source-specific validators use existing backend configuration rules."""
from unittest.mock import Mock
from uuid import uuid4

import pytest

from Config import Config
from JBGApplicationContracts import DataSourceKind, DatasetReference
from JBGDatasetApplicationService import DatasetApplicationService
from JBGRunPreparationService import RunConfiguration, RunPreparationService
from JBGRunConfigurationValidators import (CSVRunConfigurationValidator,
                                            SQLRunConfigurationValidator)
from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from SQLDataLayer import DataLayer


def _csv_config():
    config = Config()
    scope = StorageScope(str(uuid4()), str(uuid4()))
    dataset_id, artifact_id = str(uuid4()), str(uuid4())
    ref = ArtifactReference(scope, artifact_id, ArtifactKind.DATASET, 'text/csv', dataset_id=dataset_id)
    config.data_source = Config.DataSource(kind='csv', dataset_id=dataset_id, artifact_id=artifact_id)
    config.connection = Config.Connection(odbc_driver='', id_column='id', class_column='class',
                                           data_numerical_columns=['feature'])
    config.mail = Config.Mail(notification_email='')
    return config, ref


def test_csv_correct_scoped_dataset_and_roles():
    config, ref = _csv_config()
    CSVRunConfigurationValidator().validate(config, ref)


def test_csv_wrong_artifact_and_dataset_fail_closed():
    config, ref = _csv_config()
    config.data_source.artifact_id = str(uuid4())
    with pytest.raises(ValueError, match='does not match'):
        CSVRunConfigurationValidator().validate(config, ref)
    config.data_source.artifact_id = ref.artifact_id
    config.data_source.dataset_id = str(uuid4())
    with pytest.raises(ValueError, match='does not match'):
        CSVRunConfigurationValidator().validate(config, ref)


def test_csv_sql_identity_and_invalid_roles_rejected():
    config, ref = _csv_config()
    config.connection.host = 'sql.example'
    with pytest.raises(ValueError, match='SQL or email'):
        CSVRunConfigurationValidator().validate(config, ref)
    config.connection.host = ''
    config.connection.data_numerical_columns = ['id']
    with pytest.raises(ValueError, match='column roles'):
        CSVRunConfigurationValidator().validate(config, ref)


def test_csv_rejects_other_artifact_kind():
    config, ref = _csv_config()
    other = ArtifactReference(ref.scope, ref.artifact_id, ArtifactKind.MANIFEST,
                              'application/json', dataset_id=ref.dataset_id)
    with pytest.raises(ValueError, match='dataset artifact'):
        CSVRunConfigurationValidator().validate(config, other)


def test_sql_requires_real_bound_layer_without_connecting():
    config = Config()
    config.data_source = Config.DataSource(kind='sql')
    config.connection.data_catalog = 'database'
    config.connection.data_table = 'records'
    layer = object.__new__(DataLayer)
    layer.config = config
    SQLRunConfigurationValidator().validate(config, layer)
    with pytest.raises(ValueError, match='bound'):
        SQLRunConfigurationValidator().validate(config, object())
    other = Config()
    layer.config = other
    with pytest.raises(ValueError, match='bound'):
        SQLRunConfigurationValidator().validate(config, layer)


def test_preparation_uses_only_its_source_validator():
    config, ref = _csv_config()
    csv_resolver = Mock(resolve=Mock(return_value=ref))
    sql_resolver = Mock(resolve=Mock(side_effect=AssertionError('SQL must not be touched')))
    service = RunPreparationService(DatasetApplicationService(csv=csv_resolver, sql=sql_resolver),
                                    csv=CSVRunConfigurationValidator(), sql=SQLRunConfigurationValidator())
    prepared = service.prepare(DatasetReference(DataSourceKind.CSV, ref.artifact_id),
                               RunConfiguration(DataSourceKind.CSV, config))
    assert prepared.resolved_dataset is ref
    sql_resolver.resolve.assert_not_called()


def test_existing_mode_validation_is_reused():
    config, ref = _csv_config()
    config.mode.train = False
    config.mode.predict = False
    with pytest.raises((TypeError, ValueError)):
        CSVRunConfigurationValidator().validate(config, ref)
