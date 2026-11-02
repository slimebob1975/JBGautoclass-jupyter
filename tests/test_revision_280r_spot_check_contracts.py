"""280r: spot-check contracts describe existing runtime values without wrapping them."""
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import Mock

import numpy as np

from JBGSpotCheckContracts import (
    SpotCheckBestState, SpotCheckExecutionResult, SpotCheckResultRow,
    SpotCheckSelectionResult,
)
from JBGSpotCheckExecution import evaluate_spot_check_candidate
from JBGSpotCheckSelection import consider_spot_check_candidate
from JBGHandler import ModelHandler, _SpotCheckState
from JBGMeta import Reduction


def test_result_contracts_preserve_builtin_list_and_tuple_shapes():
    row: SpotCheckResultRow = ['NOS', 'NOR', 'LRN', 4, 0.75, 0.02, 0.1, '']
    result: SpotCheckExecutionResult = ([row], True)
    selection: SpotCheckSelectionResult = ('', True)
    assert type(row) is list
    assert type(result) is tuple
    assert type(selection) is tuple
    assert result[0][0] is row


def test_existing_best_state_satisfies_static_protocol_without_new_wrapper():
    state = _SpotCheckState(best_num_components=4, best_rfe_feature_selection=4)
    for attribute in SpotCheckBestState.__annotations__:
        assert hasattr(state, attribute)
    assert type(state) is _SpotCheckState


def test_public_and_extracted_return_annotations_agree():
    assert get_type_hints(evaluate_spot_check_candidate)['return'] == SpotCheckExecutionResult
    assert get_type_hints(consider_spot_check_candidate)['return'] == SpotCheckSelectionResult
    assert get_type_hints(ModelHandler._evaluate_spot_check_candidate)['return'] == SpotCheckExecutionResult
    assert get_type_hints(ModelHandler._consider_spot_check_candidate)['return'] == SpotCheckSelectionResult
    assert get_type_hints(ModelHandler._build_spot_check_result)['return'] == SpotCheckResultRow


def test_selection_still_mutates_original_state_and_returns_plain_tuple():
    state = _SpotCheckState(best_num_components=4, best_rfe_feature_selection=4)
    pipeline = object()
    def update(**values):
        state.trained_pipeline = values['pipeline']
        state.best_cv_score = values['cv_score']
    result = consider_spot_check_candidate(
        state=state, pipeline=pipeline, preprocessor=object(), reduction=object(),
        algorithm=object(), cv_score=.8, cv_stdev=.1, num_features=4,
        num_components=4, failure='', is_best_run_yet=lambda *args: True,
        on_best_candidate=update,
    )
    assert type(result) is tuple
    assert result == ('', True)
    assert state.trained_pipeline is pipeline
    assert state.best_cv_score == .8


def test_skipped_execution_returns_legacy_result_row_and_plain_tuple():
    owner = SimpleNamespace()
    owner.get_preflight_skip_reason = Mock(return_value='unsupported')
    owner._build_spot_check_result = Mock(return_value=['skipped-row'])
    dh = SimpleNamespace(
        X_train=np.zeros((6, 4)),
        Y_train=np.array([0, 1, 0, 1, 0, 1])
    )
    part = SimpleNamespace(name='ITEM')
    result = evaluate_spot_check_candidate(
        owner, dh, part, None, Reduction.PCA, None, part, None,
        None, None, None, object(),
    )
    assert type(result) is tuple
    assert result == ([['skipped-row']], False)
    owner._build_spot_check_result.assert_called_once()
