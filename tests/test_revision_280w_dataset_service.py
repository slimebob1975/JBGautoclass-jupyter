"""Application dataset dispatch guards, independent of GUI and database."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from JBGApplicationContracts import DataSourceKind, DatasetReference
from JBGDatasetApplicationService import DatasetApplicationService, DatasetResolver


class Adapter:
    def __init__(self, result):
        self.resolve = Mock(return_value=result)


def test_sql_reference_dispatches_only_to_sql_adapter():
    sql, csv = Adapter(object()), Adapter(object())
    service = DatasetApplicationService(sql=sql, csv=csv)
    ref = DatasetReference(DataSourceKind.SQL, "same-key")
    assert service.resolve(ref) is sql.resolve.return_value
    sql.resolve.assert_called_once_with("same-key")
    csv.resolve.assert_not_called()


def test_csv_reference_dispatches_only_to_csv_adapter():
    sql, csv = Adapter(object()), Adapter(object())
    service = DatasetApplicationService(sql=sql, csv=csv)
    assert service.resolve(DatasetReference(DataSourceKind.CSV, "same-key")) is csv.resolve.return_value
    csv.resolve.assert_called_once_with("same-key")
    sql.resolve.assert_not_called()


@pytest.mark.parametrize("kind", [DataSourceKind.SQL, DataSourceKind.CSV])
def test_unconfigured_source_is_rejected_without_fallback(kind):
    other = Adapter(object())
    service = (DatasetApplicationService(csv=other) if kind == DataSourceKind.SQL
               else DatasetApplicationService(sql=other))
    with pytest.raises(ValueError, match="not configured"):
        service.resolve(DatasetReference(kind, "key"))
    other.resolve.assert_not_called()


def test_rejected_reference_never_reaches_adapter():
    adapter = Adapter(object())
    service = DatasetApplicationService(csv=adapter)
    for ref in ("csv", SimpleNamespace(source="csv", key="key"), None):
        with pytest.raises(TypeError):
            service.resolve(ref)
    adapter.resolve.assert_not_called()


def test_adapter_failures_propagate_without_cross_source_retry():
    sql, csv = Adapter(None), Adapter(object())
    sql.resolve.side_effect = PermissionError("access denied")
    service = DatasetApplicationService(sql=sql, csv=csv)
    with pytest.raises(PermissionError, match="access denied"):
        service.resolve(DatasetReference(DataSourceKind.SQL, "key"))
    csv.resolve.assert_not_called()


def test_adapter_is_mandatory_protocol_and_not_data_reader():
    with pytest.raises(TypeError, match="resolve"):
        DatasetApplicationService(csv=object())
    assert isinstance(Adapter(None), DatasetResolver)


def test_keys_remain_opaque_and_unchanged():
    csv = Adapter(object())
    service = DatasetApplicationService(csv=csv)
    service.resolve(DatasetReference(DataSourceKind.CSV, "registration:artifact/opaque"))
    csv.resolve.assert_called_once_with("registration:artifact/opaque")


def test_module_has_no_gui_or_ml_engine_imports():
    import JBGDatasetApplicationService as module
    content = open(module.__file__, encoding="utf-8").read()
    assert "from GUI" not in content
    assert "import JBGHandler" not in content
    assert "import AutomaticClassifier" not in content
