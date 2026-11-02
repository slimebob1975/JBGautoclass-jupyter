"""Revision 280f: backend dependency contracts extracted without breaking imports."""
import inspect
from typing import Protocol

import JBGHandler
from JBGHandlerContracts import Logger, DataLayer, Config


def test_legacy_exports_have_exact_same_identity():
    assert JBGHandler.Logger is Logger
    assert JBGHandler.DataLayer is DataLayer
    assert JBGHandler.Config is Config


def test_all_contracts_remain_protocols():
    assert all(issubclass(cls, Protocol) for cls in (Logger, DataLayer, Config))
    assert all(getattr(cls, '_is_protocol', False) for cls in (Logger, DataLayer, Config))


def test_logger_contract_keeps_progress_and_diagnostic_methods():
    for name in ('print_info', 'print_progress', 'update_progress',
                 'investigate_dataset', 'print_prediction_report',
                 'print_privacy_safe', 'start_inline_progress'):
        assert callable(getattr(Logger, name))


def test_data_layer_contract_keeps_io_signatures():
    for name in ('get_dataset', 'save_prediction_data', 'get_data_query', 'get_table_columns'):
        assert callable(getattr(DataLayer, name))
    assert list(inspect.signature(DataLayer.get_dataset).parameters) == ['self', 'num_rows']


def test_config_contract_keeps_training_and_prediction_controls():
    for name in ('should_train', 'should_predict', 'get_callable_algorithms',
                 'get_callable_reductions', 'get_class_column_name', 'get_id_column_name'):
        assert callable(getattr(Config, name))


def test_handler_uses_extracted_contracts_in_annotations():
    annotations = inspect.get_annotations(JBGHandler.JBGHandler, eval_str=False)
    assert annotations['datalayer'] == 'DataLayer'
    assert annotations['config'] == 'Config'
    assert annotations['logger'] == 'Logger'
