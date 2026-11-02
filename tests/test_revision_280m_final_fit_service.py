"""Revision 280m: behavior contracts for selected-model final fit."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from JBGFinalFitService import fit_selected_model
from JBGExceptions import ModelException


def make_context(params):
    log = SimpleNamespace(print_progress=Mock(), print_info=Mock(), print_code=Mock())
    model = SimpleNamespace(algorithm=SimpleNamespace(search_params=SimpleNamespace(parameters=params)),
                            pipeline=Mock())
    model.get_name = Mock(return_value='PRE-LRN')
    context = SimpleNamespace(handler=SimpleNamespace(logger=log),
                              train_picked_model=Mock(return_value='ordinary-pipeline'),
                              model_parameter_grid_search=Mock(return_value=('optimized-pipeline', {'score': 1})),
                              _export_final_search_results=Mock())
    dh = SimpleNamespace(X_train=object(), Y_train=object())
    return context, model, dh, log


def test_ordinary_fit_skips_search_and_keeps_model_identity():
    context, model, dh, log = make_context({})
    assert fit_selected_model(context, model, dh, 2, 'out.csv') is model
    assert model.pipeline == 'ordinary-pipeline'
    context.train_picked_model.assert_called_once()
    context.model_parameter_grid_search.assert_not_called()


def test_successful_grid_search_exports_result():
    context, model, dh, log = make_context({'max_depth': [2]})
    # A string would not have get_params; replace returned grid pipeline with a mock.
    optimized = Mock()
    context.model_parameter_grid_search.return_value = optimized, {'score': 1}
    assert fit_selected_model(context, model, dh, 3, 'out.csv') is model
    assert model.pipeline is optimized
    assert context.model_parameter_grid_search.call_args.args[1] == {'LRN__max_depth': [2]}
    context._export_final_search_results.assert_called_once_with({'score': 1}, 'out.csv')
    context.train_picked_model.assert_not_called()


def test_grid_search_model_exception_uses_ordinary_fit_fallback():
    context, model, dh, log = make_context({'max_depth': [2]})
    context.model_parameter_grid_search.side_effect = ModelException('search failed')
    assert fit_selected_model(context, model, dh, 2, 'out.csv') is model
    assert model.pipeline == 'ordinary-pipeline'
    context.train_picked_model.assert_called_once()
    context._export_final_search_results.assert_not_called()


def test_non_model_exception_is_not_swallowed():
    context, model, dh, log = make_context({'max_depth': [2]})
    context.model_parameter_grid_search.side_effect = RuntimeError('unexpected')
    with pytest.raises(RuntimeError, match='unexpected'):
        fit_selected_model(context, model, dh, 2, 'out.csv')
    context.train_picked_model.assert_not_called()
