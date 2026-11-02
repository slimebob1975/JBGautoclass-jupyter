"""280s: candidate-execution dependency is a structural, non-runtime port."""
import ast
import inspect
from pathlib import Path
from typing import get_type_hints

from JBGHandler import ModelHandler
from JBGSpotCheckContracts import SpotCheckExecutionOwner
from JBGSpotCheckExecution import evaluate_spot_check_candidate


def test_execution_owner_annotation_is_explicit_port():
    assert get_type_hints(evaluate_spot_check_candidate)['owner'] is SpotCheckExecutionOwner


def test_model_handler_has_every_required_port_method():
    methods = [name for name, value in vars(SpotCheckExecutionOwner).items()
               if callable(value) and not name.startswith('__')]
    assert set(methods) == {
        'get_preflight_skip_reason', '_build_spot_check_result',
        '_get_rfe_search_round_limit', 'create_pipeline_and_cv',
        'get_components_from_pipeline', '_summarize_cv_results',
        'calculate_current_features', '_consider_spot_check_candidate',
        '_benchmark_candidate_150',
    }
    assert all(callable(getattr(ModelHandler, name, None)) for name in methods)


def test_execution_module_does_not_import_model_handler():
    module = ast.parse(inspect.getsource(evaluate_spot_check_candidate))
    assert isinstance(module, ast.Module)
    source_path = Path(inspect.getfile(evaluate_spot_check_candidate))
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert 'JBGHandler' not in imports


def test_port_is_not_runtime_checkable_or_wrapper():
    assert getattr(SpotCheckExecutionOwner, '_is_protocol', False)
    assert getattr(SpotCheckExecutionOwner, '_is_runtime_protocol', False) is False
