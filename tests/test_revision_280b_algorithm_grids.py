"""Regression checks for the 280b extraction (no GUI dependencies)."""

import pickle

from JBGAlgorithmGrids import _pytorch_grid_parameters as extracted_grid
from JBGMeta import (
    Algorithm, AlgorithmGridSearchParams,
    _pytorch_grid_parameters as legacy_grid,
)


def test_grid_builder_legacy_import_is_same_callable():
    assert legacy_grid is extracted_grid


def test_pytorch_variant_grid_contracts_unchanged():
    cases = {
        "TORA": ("relu", "adam"),
        "TORS": ("relu", "sgd"),
        "TOTA": ("tanh", "adam"),
        "TOTS": ("tanh", "sgd"),
        "TOSA": ("sigmoid", "adam"),
        "TOSS": ("sigmoid", "sgd"),
    }
    for name, (activation, optimizer) in cases.items():
        grid = AlgorithmGridSearchParams[name].parameters
        assert grid["activation"] == (activation,)
        assert grid["optimizer"] == (optimizer,)
        assert grid["max_epochs"] == (50,)
        assert grid["dropout_prob"] == (0.1,)
        assert grid["hidden_layer_size"] == (48, 100)
        assert grid["train_split"] == (True,)
        assert grid["learning_rate"] == (
            (0.0003, 0.001, 0.003) if optimizer == "adam"
            else ((0.02, 0.05, 0.1) if name == "TOSS" else (0.01, 0.02, 0.05))
        )
        assert grid["num_hidden_layers"] == ((0,) if name == "TOSS" else (2,))
        assert Algorithm[name].search_params is AlgorithmGridSearchParams[name]


def test_enum_pickle_compatibility_module_paths():
    assert Algorithm.__module__ == "JBGMeta"
    assert AlgorithmGridSearchParams.__module__ == "JBGMeta"
    for enum_member in (Algorithm.TORA, AlgorithmGridSearchParams.TORA):
        assert pickle.loads(pickle.dumps(enum_member)) is enum_member
