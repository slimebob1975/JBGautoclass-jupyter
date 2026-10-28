import pickle

import numpy as np
import pytest
from scipy import sparse

from JBGMeta import (Algorithm, Oversampling, Preprocess, Reduction, ScoreMetric)
from Types import Detecter
from typing import runtime_checkable

class TestDetector:
    """Detecter is a structural protocol, not a callable enum registry."""

    def test_detector_contract_accepts_detect_method(self):
        class ConcreteDetector:
            def detect(self, X, y):
                return [False] * len(X)

        detector_protocol = runtime_checkable(Detecter)
        instance = ConcreteDetector()
        assert isinstance(instance, detector_protocol)
        assert instance.detect([1, 2], [0, 1]) == [False, False]

    def test_detector_contract_rejects_missing_detect(self):
        class NoDetector:
            pass

        detector_protocol = runtime_checkable(Detecter)
        assert not isinstance(NoDetector(), detector_protocol)

class TestOversampling:
    @pytest.mark.parametrize(
        "oversampler",
        [
            Oversampling.SME,
            Oversampling.SNC,
            Oversampling.ADA,
            Oversampling.BRD,
            Oversampling.KMS,
            Oversampling.SVM,
        ],
    )
    def test_interpolating_oversamplers_require_float_input(self, oversampler):
        assert oversampler.requires_float_input() is True

    @pytest.mark.parametrize("oversampler", [Oversampling.NOG, Oversampling.RND, Oversampling.SNN])
    def test_non_interpolating_oversamplers_do_not_require_float_input(self, oversampler):
        assert oversampler.requires_float_input() is False

    def test_noop_oversampler_is_standard_pickle_compatible(self):
        transformer = Oversampling.NOG.get_callable_oversampler()
        restored = pickle.loads(pickle.dumps(transformer))

        values = np.array([[1.0, 2.0]])
        assert np.array_equal(restored.transform(values), values)


class TestAlgorithm:
    """ Tests the Enum Algorithm functions """
    def test_compound_name(self):
        assert Algorithm.LDA.get_compound_name(Preprocess.STA) == "LDA-STA"

    @pytest.mark.parametrize("algorithm", [Algorithm.MNB, Algorithm.BNB, Algorithm.CNB])
    def test_naive_bayes_estimators_without_feature_importances_are_not_rfe_compatible(
        self, algorithm
    ):
        assert algorithm.rfe_compatible is False

    def test_list_callable_algorithms(self):
        """ Class Method that gets all callable algorithms and their function """
        algorithms = Algorithm.list_callable_algorithms(size=5, max_iterations=10)
        # 53 callable algorithms
        assert algorithms
        assert len(algorithms) <= len(Algorithm)

        # It's a list of tuples
        assert all(isinstance(x,tuple) for x in algorithms)

        # Each tuple have two elements
        assert all(len(x) == 2 for x in algorithms)

        # The first element in each tuple is an Algorithm enum
        assert all(isinstance(x[0], Algorithm) for x in algorithms)

        # The second element in each tuple must have the "fit" function
        assert all(hasattr(x[1], "fit") and callable(getattr(x[1], "fit")) for x in algorithms)

    def test_get_sorted_list(self):
        """ This function gives a list of tuples: (value, name) """
        sorted_list_default = Algorithm.get_sorted_list(default_terms_first=False)
        sorted_list_dumy_first = Algorithm.get_sorted_list()

        assert sorted_list_default
        assert len(sorted_list_default) >= len(Algorithm)
        assert len(sorted_list_dumy_first) == len(sorted_list_default)

        # They are a list of tuples
        assert all(isinstance(x,tuple) for x in sorted_list_default)
        assert all(isinstance(x,tuple) for x in sorted_list_dumy_first)

        # Each tuple have two elements
        assert all(len(x) == 2 for x in sorted_list_default)
        assert all(len(x) == 2 for x in sorted_list_dumy_first)

        # Each tuple have two strings as elements
        assert all(isinstance(x[0], str) for x in sorted_list_default)
        assert all(isinstance(x[1], str) for x in sorted_list_default)

        assert all(isinstance(x[0], str) for x in sorted_list_dumy_first)
        assert all(isinstance(x[1], str) for x in sorted_list_dumy_first)

        # When sorted, ADC is first
        assert sorted_list_default[0] == ("Ada Boost Classifier", "ABC")
        
        # When sorted, DUMY is first
        assert sorted_list_dumy_first[0] == ("Dummy Classifier", "DUMY")

class TestPreprocess:
    """ Tests the Enum Preprocess functions """

    def test_list_callable_preprocessors(self):
        """ Tests the list of callable Preprocessors """

        preprocessors = Preprocess.list_callable_preprocessors()

        # Includes NOS
        assert len(preprocessors) == 6

        # It's a list of tuples
        assert all(isinstance(x,tuple) for x in preprocessors)

        # Each tuple have two elements
        assert all(len(x) == 2 for x in preprocessors)

        # The first element in each tuple is an Preprocess enum
        assert all(isinstance(x[0], Preprocess) for x in preprocessors)

        # This uses the sublist of not-None and only 1 element is None
        callables = [x[1] for x in preprocessors if x[1] is not None]
        assert all(hasattr(x, "fit") and callable(getattr(x, "fit")) for x in callables)
        assert len(callables) == 6

    def test_get_sorted_list(self):
        """ This function gives a list of tuples: (value, name) """
        sorted_list_default = Preprocess.get_sorted_list(default_terms_first=False)
        sorted_list_nos_first = Preprocess.get_sorted_list()
        
        
        assert len(sorted_list_default) == 6
        assert len(sorted_list_nos_first) == 6

        # They are a list of tuples
        assert all(isinstance(x,tuple) for x in sorted_list_default)
        assert all(isinstance(x,tuple) for x in sorted_list_nos_first)

        # Each tuple have two elements
        assert all(len(x) == 2 for x in sorted_list_default)
        assert all(len(x) == 2 for x in sorted_list_nos_first)

        # Each tuple have two strings as elements
        assert all(isinstance(x[0], str) for x in sorted_list_default)
        assert all(isinstance(x[1], str) for x in sorted_list_default)

        assert all(isinstance(x[0], str) for x in sorted_list_nos_first)
        assert all(isinstance(x[1], str) for x in sorted_list_nos_first)

        # When sorted, BIN is first
        assert sorted_list_default[0] == ("Binarizer", "BIN")
        
        # When sorted, NOS is first
        assert sorted_list_nos_first[0] == ("No Scaling", "NOS")

class TestReduction:
    """ Tests the Enum Reduction functions """
    def test_default_pca_variance_target_supports_sparse_input(self):
        rng = np.random.default_rng(1)
        X = sparse.csr_matrix(rng.normal(size=(40, 8)))

        pca = Reduction.PCA.get_PCA(
            num_samples=X.shape[0],
            num_features=X.shape[1],
            num_selected_features=None,
        )
        transformed = pca.fit_transform(X)

        assert pca.n_components == 0.95
        assert pca.svd_solver == "covariance_eigh"
        assert transformed.shape[0] == X.shape[0]
        assert 1 <= pca.n_components_ <= X.shape[1]

    def test_tsvd_components_do_not_exceed_available_features(self):
        rng = np.random.default_rng(1)
        X = sparse.csr_matrix(rng.normal(size=(128, 93)))

        tsvd = Reduction.TSVD.get_TSVD(
            num_samples=X.shape[0],
            num_features=X.shape[1],
            num_selected_features=None,
        )
        transformed = tsvd.fit_transform(X)

        assert tsvd.n_components == X.shape[1] == 93
        assert transformed.shape[0] == X.shape[0]

    def test_tsvd_explicit_components_are_capped_to_available_features(self):
        tsvd = Reduction.TSVD.get_TSVD(
            num_samples=128,
            num_features=93,
            num_selected_features=120,
        )

        assert tsvd.n_components == 93

    def test_fastica_components_do_not_exceed_effective_input_dimension(self):
        fica = Reduction.FICA.get_FICA(
            num_samples=128,
            num_features=13,
            num_selected_features=None,
        )

        assert fica.n_components == 13
        assert fica.max_iter == 1000
        assert fica.random_state == 1

    def test_fastica_explicit_components_are_capped_to_samples_and_features(self):
        fica = Reduction.FICA.get_FICA(
            num_samples=40,
            num_features=93,
            num_selected_features=120,
        )

        assert fica.n_components == 40

    def test_get_sorted_list(self):
        """ This function gives a list of tuples: (value, name) """
        sorted_list_default = Reduction.get_sorted_list(default_terms_first=False)
        sorted_list_nor_first = Reduction.get_sorted_list()

        assert len(sorted_list_default) == 10
        assert len(sorted_list_nor_first) == 10

        # They are a list of tuples
        assert all(isinstance(x,tuple) for x in sorted_list_default)
        assert all(isinstance(x,tuple) for x in sorted_list_nor_first)

        # Each tuple have two elements
        assert all(len(x) == 2 for x in sorted_list_default)
        assert all(len(x) == 2 for x in sorted_list_nor_first)

        # Each tuple have two strings as elements
        assert all(isinstance(x[0], str) for x in sorted_list_default)
        assert all(isinstance(x[1], str) for x in sorted_list_default)

        assert all(isinstance(x[0], str) for x in sorted_list_nor_first)
        assert all(isinstance(x[1], str) for x in sorted_list_nor_first)

        # When sorted, FICA is first
        assert sorted_list_default[0] == ("Fast Indep. Component Analysis", "FICA")
        
        # When sorted, NONE is first
        assert sorted_list_nor_first[0] == ("No Reduction", "NOR")

class TestScoreMetric:
    """ Tests the Enum ScoreMetric functions """
    def test_get_sorted_list(self):
        """ This function gives a list of tuples: (value, name) """
        # For ScoreMetric there is no difference between these two, as there is no NON or ALL
        sorted_list_default = ScoreMetric.get_sorted_list(default_terms_first=False)
        sorted_list_all_first = ScoreMetric.get_sorted_list()

        assert sorted_list_default == sorted_list_all_first
        assert len(sorted_list_default) == 17
        
        # They are a list of tuples
        assert all(isinstance(x,tuple) for x in sorted_list_default)
        
        # Each tuple have two elements
        assert all(len(x) == 2 for x in sorted_list_default)
        
        # Each tuple have two strings as elements
        assert all(isinstance(x[0], str) for x in sorted_list_default)
        assert all(isinstance(x[1], str) for x in sorted_list_default)

        # When sorted, accuracy is first
        assert sorted_list_default[0] == ("Accuracy", "accuracy")
    @pytest.mark.parametrize("metric,label,legacy_label", [
        ("f1_micro", "F1 Micro", "Balanced F1 Micro"),
        ("f1_macro", "F1 Macro", "Balanced F1 Macro"),
        ("f1_weighted", "F1 Weighted", "Balanced F1 Weighted"),
    ])
    def test_f1_display_labels_preserve_legacy_enum_values_and_scorers(self, metric, label, legacy_label):
        from sklearn.metrics import f1_score

        member = ScoreMetric[metric]
        legacy_value = {"full_name": legacy_label, "callable": f1_score,
                        "kwargs": {"average": metric.removeprefix("f1_"), "pos_label": None}}
        assert member.full_name == member.get_full_name() == label
        assert member.value == legacy_value
        assert ScoreMetric(legacy_value) is member
        assert pickle.loads(pickle.dumps(member)) is member
        assert (label, metric) in ScoreMetric.get_sorted_list()
        assert member.callable is f1_score
        assert ScoreMetric.defaultScoreMetric() is ScoreMetric.mcc

    def test_legacy_value_based_pickle_loads_as_current_f1_enum(self):
        from sklearn.metrics import f1_score

        class LegacyValuePickle:
            def __reduce__(self):
                return (ScoreMetric, ({"full_name": "Balanced F1 Micro", "callable": f1_score,
                                       "kwargs": {"average": "micro", "pos_label": None}},))

        assert pickle.loads(pickle.dumps(LegacyValuePickle())) is ScoreMetric.f1_micro

    def test_micro_f1_scorer_retains_accuracy_semantics_on_imbalanced_labels(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import accuracy_score

        X = np.zeros((100, 1))
        y = np.array(["0"] * 99 + ["1"])
        model = DummyClassifier(strategy="most_frequent").fit(X, y)
        assert ScoreMetric.f1_micro.get_parametrized_scorer()(model, X, y) == accuracy_score(y, model.predict(X))
