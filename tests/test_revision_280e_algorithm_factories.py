"""Revision 280e: estimator factory delegation with stable enum identity."""
import pickle

from JBGAlgorithmFactories import AlgorithmEstimatorFactories
from JBGMeta import Algorithm, AlgorithmGridSearchParams
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def test_enum_identity_and_pickle_roundtrip():
    assert Algorithm.__module__ == "JBGMeta"
    assert pickle.loads(pickle.dumps(Algorithm.LRN)) is Algorithm.LRN
    assert Algorithm.LRN.value["search_params"] is AlgorithmGridSearchParams.LRN


def test_all_factory_methods_remain_resolvable():
    assert issubclass(Algorithm, AlgorithmEstimatorFactories)
    for item in Algorithm:
        if item.has_function():
            assert callable(getattr(item, "do_" + item.name))


def test_factory_delegation_preserves_standard_estimator_construction():
    assert isinstance(Algorithm.DUMY.call_algorithm(max_iterations=40, size=8), DummyClassifier)
    model = Algorithm.LRN.call_algorithm(max_iterations=83, size=8)
    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 83


def test_qda_factory_still_uses_existing_grid():
    estimator = Algorithm.QDA.call_algorithm(max_iterations=80, size=8)
    assert isinstance(estimator, QuadraticDiscriminantAnalysis)
    assert estimator.reg_param == min(AlgorithmGridSearchParams.QDA.parameters["reg_param"])


def test_svc_size_fallback_still_works():
    assert Algorithm.SVC.call_algorithm(max_iterations=80, size=10001).__class__.__name__ == "LinearSVC"
    assert Algorithm.SVC.call_algorithm(max_iterations=80, size=10).__class__.__name__ == "SVC"
