"""Real Torch/skorch checks for variant identity, isolated fits and persistence."""
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import dill
from joblib import Parallel, delayed, parallel_backend
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim

from JBGMeta import Algorithm, AlgorithmGridSearchParams
from JBGModelPersistence import load_model_artifact, save_model_artifact
from JBGNeuralNetworks import _NeuralNetwork3PL
from JBGTransformers import NNClassifier3PL


VARIANTS = [
    ("TORA", "relu", "adam"), ("TORS", "relu", "sgd"),
    ("TOTA", "tanh", "adam"), ("TOTS", "tanh", "sgd"),
    ("TOSA", "sigmoid", "adam"), ("TOSS", "sigmoid", "sgd"),
]


@pytest.fixture(autouse=True)
def isolated_cpu_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(NNClassifier3PL, "OUTPUT_DIR", str(tmp_path))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


def _data(features=5, classes=2):
    return make_classification(
        n_samples=90, n_features=features, n_informative=features,
        n_redundant=0, n_classes=classes, n_clusters_per_class=1, random_state=42,
    )


@pytest.mark.parametrize("name, activation, optimizer", VARIANTS)
def test_named_variants_use_the_requested_optimizer_and_bounded_identity_grid(
    name, activation, optimizer
):
    algorithm = Algorithm[name]
    model = algorithm.call_algorithm(max_iterations=20000, size=5)
    grid = list(ParameterGrid(algorithm.search_params.parameters))
    assert len(grid) == 6
    assert {cell["activation"] for cell in grid} == {activation}
    assert {cell["optimizer"] for cell in grid} == {optimizer}
    assert {cell["max_epochs"] for cell in grid} == {50}
    assert model.get_params() == clone(model).get_params()
    assert model.learning_rate == (0.1 if name == "TOSS" else (0.001 if optimizer == "adam" else 0.02))
    assert model.num_hidden_layers == (0 if name == "TOSS" else 2)
    # The spot-check starts from a candidate in its own final search profile.
    for parameter, choices in algorithm.search_params.parameters.items():
        assert model.get_params()[parameter] in choices
    X, y = _data()
    model.set_params(max_epochs=2)
    assert model.fit(pd.DataFrame(X), np.array([f"class-{label}" for label in y])) is model
    assert is_classifier(model)
    assert isinstance(model.net.optimizer_, optim.Adam if optimizer == "adam" else optim.SGD)
    assert model.net.optimizer_.param_groups[0]["lr"] == model.learning_rate
    assert isinstance(model.net.criterion_, nn.NLLLoss)
    assert model.net.batch_size == 128
    assert model.net.module_.layers[0].in_features == X.shape[1]
    assert len(model.net.module_.layers) == (2 if name == "TOSS" else 4)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(y), 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(probabilities, model.predict_proba(X))
    np.testing.assert_array_equal(model.predict(X), model.classes_[probabilities.argmax(axis=1)])
    assert not Path(model.checkpoint_fit_dir_).exists()


def test_dropout_is_stochastic_only_in_training_mode():
    module = _NeuralNetwork3PL(5, 3, 2, 48, nn.ReLU, optim.Adam, 0.5)
    # The historical architecture has one input-to-hidden stage plus two more.
    assert len(module.layers) == 4
    X = torch.ones((20, 5))
    module.train()
    assert not torch.equal(module(X), module(X))
    module.eval()
    assert torch.equal(module(X), module(X))


@pytest.mark.parametrize("train_split, monitor", [(True, "valid_loss_best"), (False, "train_loss_best")])
def test_checkpoint_monitor_and_cleanup_on_success_and_refit(train_split, monitor):
    X, y = _data()
    model = NNClassifier3PL(max_epochs=2, verbose=False, train_split=train_split)
    model.fit(X, y)
    assert model.net.callbacks[0].monitor == monitor
    first = model.checkpoint_fit_dir_
    assert not Path(first).exists()
    assert "_checkpoint_fit_dir" not in vars(model)
    model.fit(X[:, :3], y)
    assert model.checkpoint_fit_dir_ != first
    assert model.n_features_in_ == 3
    assert not Path(model.checkpoint_fit_dir_).exists()


def test_failed_fit_cleans_only_its_private_directory(monkeypatch):
    X, y = _data()
    model = NNClassifier3PL(verbose=False)
    root = model.history_file_dir
    root.mkdir(parents=True)
    unrelated = root / "params.pt"
    unrelated.write_text("older file must survive")
    directories = []

    def fail_setup(X, y):
        directories.append(model.history_file_dir)
        (model.history_file_dir / "history.json").write_text("partial")
        raise RuntimeError("fit failed")

    monkeypatch.setattr(model, "_setup_net", fail_setup)
    with pytest.raises(RuntimeError, match="fit failed"):
        model.fit(X, y)
    assert len(directories) == 1 and not directories[0].exists()
    assert unrelated.read_text() == "older file must survive"
    assert "_checkpoint_fit_dir" not in vars(model)
    with pytest.raises(NotFittedError):
        model.predict(X)


def _fit_worker(features, classes, checkpoint_root):
    torch.set_num_threads(1)
    X, y = _data(features, classes)
    model = NNClassifier3PL(max_epochs=3, hidden_layer_size=16, verbose=False)
    model.OUTPUT_DIR = str(checkpoint_root)
    model.fit(X, y)
    return model.checkpoint_fit_dir_, model.predict_proba(X).shape, model.net.module_.layers[-1].out_features


def test_parallel_heterogeneous_fits_use_different_checkpoint_directories(tmp_path):
    specifications = [(5, 2), (8, 3), (6, 2), (9, 4)]
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=2)(delayed(_fit_worker)(features, classes, tmp_path)
                                    for features, classes in specifications)
    assert len({result[0] for result in results}) == len(results)
    for (directory, shape, output_classes), (_, classes) in zip(results, specifications):
        assert shape == (90, classes)
        assert output_classes == classes
        assert not Path(directory).exists()


def test_parallel_cv_and_grid_search_produce_usable_classifiers():
    X, y = _data()
    pipeline = Pipeline([("scale", StandardScaler()), ("torch", NNClassifier3PL(
        max_epochs=2, hidden_layer_size=16, verbose=False, train_split=False))])
    with parallel_backend("loky", inner_max_num_threads=1):
        result = cross_validate(pipeline, X, y, cv=StratifiedKFold(3),
                                scoring="roc_auc", n_jobs=2, return_estimator=True, error_score="raise")
        search = GridSearchCV(pipeline, {"torch__optimizer": ["adam", "sgd"]},
                              cv=2, scoring="accuracy", n_jobs=2, error_score="raise").fit(X, y)
    assert np.isfinite(result["test_score"]).all()
    assert len({est.named_steps["torch"].checkpoint_fit_dir_ for est in result["estimator"]}) == 3
    assert search.predict_proba(X).shape == (90, 2)
    assert all(not Path(est.named_steps["torch"].checkpoint_fit_dir_).exists()
               for est in result["estimator"])


@pytest.mark.parametrize("serializer", [pickle, dill])
def test_fitted_classifier_roundtrip_uses_no_checkpoint_files(serializer):
    X, y = _data()
    model = NNClassifier3PL(max_epochs=2, verbose=False).fit(X, y)
    restored = serializer.loads(serializer.dumps(model))
    assert not Path(model.checkpoint_fit_dir_).exists()
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))
    # Older artifacts have num_features but no n_features_in_.
    del restored.n_features_in_
    np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))


def test_legacy_plain_net_state_is_still_readable():
    X, y = _data()
    model = NNClassifier3PL(max_epochs=2, verbose=False).fit(X, y)
    historical_state = vars(model).copy()
    historical_state.pop("n_features_in_")
    historical_state.pop("checkpoint_fit_dir_")
    restored = object.__new__(NNClassifier3PL)
    restored.__setstate__(historical_state)
    np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))


def test_prediction_contract_and_framework_fallback_policy():
    from JBGDarkNumberExecution import resolve_fallback_workers

    X, y = _data()
    model = NNClassifier3PL(max_epochs=2, verbose=False)
    with pytest.raises(NotFittedError):
        model.predict_proba(X)
    model.fit(X, y)
    with pytest.raises(ValueError, match="Expected 5 input features, got 4"):
        model.predict(X[:, :4])
    assert resolve_fallback_workers(model, -1, 20)[0] == 1


@pytest.mark.parametrize("algorithm", [Algorithm.TORA, Algorithm.TOSS])
def test_model_artifact_reload_in_fresh_process_after_checkpoint_cleanup(tmp_path, algorithm):
    X, y = _data()
    model = algorithm.call_algorithm(max_iterations=20000, size=X.shape[1])
    model.set_params(max_epochs=2)
    pipeline = Pipeline([("torch", model)]).fit(X, y)
    artifact = tmp_path / "model.sav"
    save_model_artifact(artifact, {}, None, ("NOG", "NUG", "NOS", "NOR", algorithm),
                        pipeline, None, X.shape[1])
    assert load_model_artifact(artifact)[2][-1] is algorithm
    data = tmp_path / "X.npy"
    np.save(data, X)
    code = """import json, sys, numpy as np
from JBGModelPersistence import load_model_artifact
model = load_model_artifact(sys.argv[1])[3]
print(json.dumps(model.predict_proba(np.load(sys.argv[2])).tolist()))
"""
    process = subprocess.run([sys.executable, "-c", code, str(artifact), str(data)],
                             env=os.environ.copy(), text=True, capture_output=True, check=True)
    np.testing.assert_array_equal(np.asarray(json.loads(process.stdout)), pipeline.predict_proba(X))


def test_historical_enum_values_remain_loadable_by_value():
    historical_grid = {"parameters": {
        "activation": ("relu", "tanh", "sigmoid"), "optimizer": ("adam", "sgd"),
        "learning_rate": [0.01, 0.05, 0.1], "max_epochs": [10, 30, 50],
        "dropout_prob": [0.1, 0.3, 0.5], "num_hidden_layers": [2, 3],
        "hidden_layer_size": [16, 48, 100], "train_split": [True, False],
    }}
    assert AlgorithmGridSearchParams(historical_grid) is AlgorithmGridSearchParams.PYNN
    # Algorithm's stored dict still points to the original grid, while its
    # property supplies the new search policy. Old saved enum values resolve.
    for name, _, _ in VARIANTS:
        algorithm = Algorithm[name]
        assert algorithm.value["search_params"] is AlgorithmGridSearchParams.PYNN
        assert Algorithm(dict(algorithm.value)) is algorithm
        assert pickle.loads(pickle.dumps(algorithm)) is algorithm
        assert algorithm.search_params is AlgorithmGridSearchParams[name]


@pytest.mark.parametrize("serializer", [pickle, dill])
def test_revision_116_sigmoid_sgd_grid_value_remains_loadable(serializer):
    historical_value = {"parameters": {
        "activation": ("sigmoid",), "optimizer": ("sgd",),
        "learning_rate": (0.01, 0.02, 0.05), "max_epochs": (50,),
        "dropout_prob": (0.1,), "num_hidden_layers": (2,),
        "hidden_layer_size": (48, 100), "train_split": (True,),
    }}
    grid = AlgorithmGridSearchParams(historical_value)
    assert grid is AlgorithmGridSearchParams.TOSS
    assert grid.value == historical_value
    assert serializer.loads(serializer.dumps(grid)) is grid
    assert grid.parameters["num_hidden_layers"] == (0,)
    assert grid.parameters["learning_rate"] == (0.02, 0.05, 0.1)
    # A caller cannot modify the saved enum value through its current profile.
    current = grid.parameters
    current["num_hidden_layers"] = (99,)
    assert grid.value == historical_value
    assert grid.parameters["num_hidden_layers"] == (0,)
