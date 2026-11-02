"""Revision 280y: non-executing run preparation and source isolation."""
from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest

from JBGApplicationContracts import DataSourceKind, DatasetReference
from JBGDatasetApplicationService import DatasetApplicationService
from JBGRunPreparationService import (PreparedRun, RunConfiguration,
                                      RunPreparationService)


class Resolver:
    def __init__(self, result):
        self.resolve = Mock(return_value=result)


class Validator:
    def __init__(self):
        self.validate = Mock()


def setup():
    sql_dataset, csv_dataset = object(), object()
    sql_resolver, csv_resolver = Resolver(sql_dataset), Resolver(csv_dataset)
    sql_validator, csv_validator = Validator(), Validator()
    service = RunPreparationService(
        DatasetApplicationService(sql=sql_resolver, csv=csv_resolver),
        sql=sql_validator, csv=csv_validator)
    return service, sql_resolver, csv_resolver, sql_validator, csv_validator, sql_dataset, csv_dataset


@pytest.mark.parametrize("kind", [DataSourceKind.SQL, DataSourceKind.CSV])
def test_prepared_run_preserves_objects_and_selects_matching_validator(kind):
    service, sql_r, csv_r, sql_v, csv_v, sql_d, csv_d = setup()
    config = object()
    ref = DatasetReference(kind, "opaque")
    result = service.prepare(ref, RunConfiguration(kind, config))
    assert isinstance(result, PreparedRun)
    assert result.dataset is ref
    assert result.configuration.payload is config
    assert result.resolved_dataset is (sql_d if kind == DataSourceKind.SQL else csv_d)
    (sql_v if kind == DataSourceKind.SQL else csv_v).validate.assert_called_once_with(config, result.resolved_dataset)
    (csv_v if kind == DataSourceKind.SQL else sql_v).validate.assert_not_called()
    (csv_r if kind == DataSourceKind.SQL else sql_r).resolve.assert_not_called()


def test_cross_source_configuration_rejected_before_any_resolver():
    service, sql_r, csv_r, *_ = setup()
    with pytest.raises(ValueError, match="sources differ"):
        service.prepare(DatasetReference(DataSourceKind.CSV, "key"), RunConfiguration(DataSourceKind.SQL, object()))
    sql_r.resolve.assert_not_called()
    csv_r.resolve.assert_not_called()


def test_missing_validator_does_not_resolve_dataset():
    resolver = Resolver(object())
    service = RunPreparationService(DatasetApplicationService(csv=resolver))
    with pytest.raises(ValueError, match="not configured"):
        service.prepare(DatasetReference(DataSourceKind.CSV, "id"), RunConfiguration(DataSourceKind.CSV, object()))
    resolver.resolve.assert_not_called()


def test_invalid_inputs_fail_closed():
    service, sql_r, csv_r, *_ = setup()
    ref = DatasetReference(DataSourceKind.CSV, "id")
    for bad in (None, object(), "csv"):
        with pytest.raises(TypeError):
            service.prepare(ref, bad)
    with pytest.raises(TypeError):
        RunConfiguration("csv", object())
    with pytest.raises(ValueError):
        RunConfiguration(DataSourceKind.CSV, None)
    sql_r.resolve.assert_not_called()
    csv_r.resolve.assert_not_called()


def test_validator_failure_propagates_without_other_source_retry():
    service, sql_r, csv_r, sql_v, csv_v, *_ = setup()
    csv_v.validate.side_effect = PermissionError("scope denied")
    with pytest.raises(PermissionError, match="scope denied"):
        service.prepare(DatasetReference(DataSourceKind.CSV, "id"), RunConfiguration(DataSourceKind.CSV, object()))
    sql_r.resolve.assert_not_called()
    sql_v.validate.assert_not_called()


def test_resolver_failure_prevents_validation():
    service, sql_r, csv_r, sql_v, csv_v, *_ = setup()
    csv_r.resolve.side_effect = LookupError("missing artifact")
    with pytest.raises(LookupError, match="missing artifact"):
        service.prepare(DatasetReference(DataSourceKind.CSV, "id"), RunConfiguration(DataSourceKind.CSV, object()))
    sql_v.validate.assert_not_called()
    csv_v.validate.assert_not_called()


def test_prepared_run_is_frozen_but_payload_identity_retained():
    service, *_ = setup()
    prepared = service.prepare(DatasetReference(DataSourceKind.SQL, "id"), RunConfiguration(DataSourceKind.SQL, object()))
    with pytest.raises(FrozenInstanceError):
        prepared.resolved_dataset = None


def test_no_gui_training_or_file_io_imports():
    import JBGRunPreparationService as module
    text = open(module.__file__, encoding="utf8").read()
    for forbidden in ("import GUI", "from GUI", "import AutomaticClassifier", "import JBGHandler", "open("):
        assert forbidden not in text
