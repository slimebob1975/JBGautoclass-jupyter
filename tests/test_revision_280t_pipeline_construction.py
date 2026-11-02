"""280t: preserve pipeline construction with an explicit runtime context."""
from types import SimpleNamespace

import pytest
from sklearn.linear_model import LogisticRegression

from JBGHandler import ModelHandler
from JBGMeta import Algorithm, Preprocess, Reduction, Oversampling, Undersampling
from JBGExceptions import PipelineException
from JBGPipelineConstruction import build_model_pipeline


def _args():
    return (Reduction.NOR, None, Algorithm.LRN, LogisticRegression(max_iter=120),
            Preprocess.STA, Preprocess.STA.call_preprocess(),
            Oversampling.NOG, Undersampling.NUG, 4)


def _handler(mode=None):
    if mode is None:
        mode = SimpleNamespace(decision_policy='default', probability_threshold=0.5)
    return SimpleNamespace(handler=SimpleNamespace(config=SimpleNamespace(mode=mode)))


def test_legacy_method_delegates_to_same_pipeline_steps():
    actual = ModelHandler.get_pipeline(_handler(), *_args())
    direct = build_model_pipeline(*_args())
    assert [name for name, _ in actual.steps] == [name for name, _ in direct.steps]
    assert [type(step) for _, step in actual.steps] == [type(step) for _, step in direct.steps]
    assert actual.steps[-1][1].max_iter == 120


def test_estimator_is_cloned_and_does_not_mutate_input():
    args = list(_args())
    original = args[3]
    constructed = build_model_pipeline(*args)
    assert constructed.steps[-1][1] is not original
    assert original.random_state is None
    assert constructed.steps[-1][1].random_state == 42


def test_decision_policy_validation_happens_before_pipeline_creation():
    mode = SimpleNamespace(decision_policy='fixed', probability_threshold=0.8)
    actual = ModelHandler.get_pipeline(_handler(mode), *_args())
    assert actual.decision_policy == 'fixed'
    assert actual.probability_threshold == 0.8




def test_native_catboost_requires_initialized_schema():
    args = list(_args())

    # Native CatBoost does not permit external preprocessing,
    # reduction or sampling.
    args[2] = Algorithm.CBST
    args[4] = Preprocess.NOS
    args[5] = None

    with pytest.raises(
        PipelineException,
        match="Native CatBoost schema was not initialized"
    ):
        build_model_pipeline(*args)



def test_pipeline_module_has_no_handler_import():
    import ast
    import inspect
    from pathlib import Path
    tree = ast.parse(Path(inspect.getfile(build_model_pipeline)).read_text(encoding='utf-8'))
    assert 'JBGHandler' not in [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
