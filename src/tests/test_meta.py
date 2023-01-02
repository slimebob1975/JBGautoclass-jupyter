import pytest

from JBGMeta import (Algorithm, AlgorithmTuple, Detector, Preprocess,
                    PreprocessTuple, Reduction, ReductionTuple, ScoreMetric)

class TestDetector:
    """ Tests the Enum Detector functions """

    def test_list_callable_detectors(self):
        """ Class Method that gets all callable detectors and their function """
        detectors = Detector.list_callable_detectors()
        # 32 callable detectors
        assert len(detectors) == 8

        # It's a list of tuples
        assert all(isinstance(x,tuple) for x in detectors)

        # Each tuple have two elements
        assert all(len(x) == 2 for x in detectors)

        # The first element in each tuple is an Detector enum
        assert all(isinstance(x[0], Detector) for x in detectors)

        # The second element in each tuple must have the "detect" function
         # 1e: This uses the sublist of not-None and only 1 element is None
        callables = [x[1] for x in detectors if x[1] is not None]
        assert all(hasattr(x, "detect") and callable(getattr(x, "detect")) for x in callables)
        assert len(callables) == 7
    
    def test_get_sorted_list(self):
        """ This function gives a list of tuples: (value, name) """
        sorted_list_default = Detector.get_sorted_list(default_terms_first=False)
        sorted_list_all_first = Detector.get_sorted_list()

        assert len(sorted_list_default) == 9
        assert len(sorted_list_all_first) == 9

        # They are a list of tuples
        assert all(isinstance(x,tuple) for x in sorted_list_default)
        assert all(isinstance(x,tuple) for x in sorted_list_all_first)

        # Each tuple have two elements
        assert all(len(x) == 2 for x in sorted_list_default)
        assert all(len(x) == 2 for x in sorted_list_all_first)

        # Each tuple have two strings as elements
        assert all(isinstance(x[0], str) for x in sorted_list_default)
        assert all(isinstance(x[1], str) for x in sorted_list_default)

        assert all(isinstance(x[0], str) for x in sorted_list_all_first)
        assert all(isinstance(x[1], str) for x in sorted_list_all_first)

        # When sorted, ALL is first, as none of the others begin with A
        assert sorted_list_default[0] == ("All", "ALL")
        
        # When sorted, ALL is first
        assert sorted_list_all_first[0] == ("All", "ALL")

class TestAlgorithm:
    """ Tests the Enum Algorithm functions """
    def test_compound_name(self):
        assert Algorithm.LDA.get_compound_name(Preprocess.STA) == "LDA-STA"

    def test_list_callable_algorithms(self):
        """ Class Method that gets all callable algorithms and their function """
        algorithms = Algorithm.list_callable_algorithms(size=5, max_iterations=10)
        # 53 callable algorithms
        assert len(algorithms) == 57

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

        assert len(sorted_list_default) == 58
        assert len(sorted_list_dumy_first) == 58

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
        assert len(sorted_list_default) == 16
        
        # They are a list of tuples
        assert all(isinstance(x,tuple) for x in sorted_list_default)
        
        # Each tuple have two elements
        assert all(len(x) == 2 for x in sorted_list_default)
        
        # Each tuple have two strings as elements
        assert all(isinstance(x[0], str) for x in sorted_list_default)
        assert all(isinstance(x[1], str) for x in sorted_list_default)

        # When sorted, accuracy is first
        assert sorted_list_default[0] == ("Accuracy", "accuracy")