import os
from typing import Callable
from path import Path
import pytest
from Config import (Algorithm, Config, Detector, Preprocess, Reduction, Scoretype,
                    get_model_name)
from IAFExceptions import ConfigException
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

@pytest.fixture
def valid_iris_config() -> Config:
    config = Config(
        Config.Connection(
            odbc_driver="Mock Server",
            host="tcp:database.iaf.mock",
            trusted_connection=True,
            class_catalog="DatabaseOne",
            class_table="ResultTable",
            class_table_script="createtable.sql.txt",
            class_username="some_fake_name",
            class_password="",
            data_catalog="DatabaseTwo",
            data_table="InputTable",
            class_column="class",
            data_text_columns=[],
            data_numerical_columns=["sepal-length","sepal-width","petal-length", "petal-width"],
            id_column="id",
            data_username="some_fake_name",
            data_password=""
        ),
        Config.Mode(
            train=False,
            predict=True,
            mispredicted=False,
            use_stop_words=False,
            specific_stop_words_threshold=1.0,
            hex_encode=True,
            use_categorization=True,
            category_text_columns=[],
            test_size=0.2,
            smote=False,
            undersample=False,
            algorithm=Algorithm.LDA,
            preprocessor=Preprocess.STA,
            feature_selection=Reduction.PCA,
            num_selected_features=None,
            scoring=Scoretype.accuracy,
            max_iterations=20000
        ),
        Config.IO(
            verbose=True,
            model_path="./model",
            model_name="test"
        ),
        Config.Debug(
            on=True,
            num_rows=150
        ),
        name="test",
        filename="autoclassconfig_test_some_fake_name.py"
    )

    return config

@pytest.fixture
def bare_iris_config() -> Config:
    config = Config(
        Config.Connection(
            odbc_driver="Mock Server",
            host="tcp:database.iaf.mock",
            trusted_connection=True,
            class_catalog="DatabaseOne",
            class_table="ResultTable",
            class_table_script="createtable.sql.txt",
            class_username="some_fake_name",
            class_password="",
            data_catalog="DatabaseTwo",
            data_table="InputTable",
            class_column="class",
            data_text_columns=[],
            data_numerical_columns=["sepal-length","sepal-width","petal-length","petal-width"],
            id_column="id",
            data_username="some_fake_name",
            data_password=""
        ),
        Config.Mode(
            train=False,
            predict=True,
            mispredicted=False,
            use_stop_words=False,
            specific_stop_words_threshold=1.0,
            hex_encode=True,
            use_categorization=True,
            category_text_columns=[],
            test_size=0.2,
            smote=False,
            undersample=False,
            algorithm=Algorithm.LDA,
            preprocessor=Preprocess.STA,
            feature_selection=Reduction.PCA,
            num_selected_features=None,
            scoring=Scoretype.accuracy,
            max_iterations=20000
        ),
        Config.IO(
            verbose=True,
            model_path="./model",
            model_name="test"
        ),
        Config.Debug(
            on=True,
            num_rows=150
        ),
        name="iris",
        filename="autoclassconfig_iris_.py"
    )

    config.connection.data_catalog = ""
    config.connection.data_table = ""
    config.mode.train = None
    config.mode.predict = None
    config.mode.mispredicted = None
    config.io.model_name = ""
    config.debug.num_rows = 0
    

    return config

@pytest.fixture
def saved_with_valid_iris_config() -> Config:
    config = Config(
        Config.Connection(
            odbc_driver="Mock Server",
            host="tcp:database.iaf.mock",
            trusted_connection=True,
            class_catalog="DatabaseOne",
            class_table="ResultTable",
            class_table_script="createtable.sql.txt",
            class_username="some_fake_name",
            class_password="",
            data_catalog="DatabaseTwo",
            data_table="InputTable",
            class_column="class",
            data_text_columns=[],
            data_numerical_columns=["sepal-length","sepal-width","petal-length","petal-width"],
            id_column="id",
            data_username="some_fake_name",
            data_password=""
        ),
        Config.Mode(
            train=False,
            predict=True,
            mispredicted=False,
            use_stop_words=False,
            specific_stop_words_threshold=1.0,
            hex_encode=True,
            use_categorization=True,
            category_text_columns=[],
            test_size=0.2,
            smote=False,
            undersample=False,
            algorithm=Algorithm.LDA,
            preprocessor=Preprocess.STA,
            feature_selection=Reduction.PCA,
            num_selected_features=None,
            scoring=Scoretype.accuracy,
            max_iterations=20000
        ),
        Config.IO(
            verbose=True,
            model_path="./model",
            model_name="test"
        ),
        Config.Debug(
            on=True,
            num_rows=150
        ),
        name="iris",
        filename="autoclassconfig_iris_.py"
    )

    return config

def get_fixture_path() -> Path:
    pwd = os.path.dirname(os.path.realpath(__file__))
    return Path(pwd) / "fixtures"

def get_fixture_content_as_string(filename: str) -> str:
    """ Returns the content of one of the fixtures """
    with open(get_fixture_path() / filename) as f:
            content = f.read()

    return content

class TestConfig:
    def test_defaults(self):
        config = Config()
        
        assert config.name == 'iris'

    def test_column_names(self, valid_iris_config):
        """ This will test known values with various functions that return column names"""

        # As this config has no categorical or text columns, these are the relevant lists
        data_numerical_columns = ["sepal-length","sepal-width","petal-length","petal-width"]
        
        # This is a set, since the order doesn't matter, only each item
        column_names = set(["id", "class", "sepal-length","sepal-width","petal-length","petal-width"])
        
        assert valid_iris_config.get_text_column_names() == []

        assert valid_iris_config.get_numerical_column_names() == data_numerical_columns

        # data_text, data_numerical
        assert valid_iris_config.get_data_column_names() == data_numerical_columns

        # ID, class, data_text, data_numerical
        difference = set(valid_iris_config.get_column_names()) ^ column_names
        assert not difference

        # Changing the config to check that we do not get empty strings in get_column_names()
        valid_iris_config.connection.id_column = ""
        assert "" not in valid_iris_config.get_column_names()
        
        assert valid_iris_config.get_categorical_text_column_names() == []

    def test_calculated_booleans(self, valid_iris_config):
        """ This tests that calculated booleans are calculated correctly """
        # Expected values
        is_text_data = False
        is_numerical_data = True
        force_categorization = False
        use_feature_selection = True
        is_categorical = False # This is always False for this dataset, no matter the column name

        assert valid_iris_config.is_text_data() == is_text_data
        assert valid_iris_config.is_numerical_data() == is_numerical_data
        assert valid_iris_config.force_categorization() == force_categorization
        assert valid_iris_config.use_feature_selection() == use_feature_selection
        assert valid_iris_config.is_categorical(column_name="irrelevant") == is_categorical

        # Feature selection is Reduction.PCA, so let's test two cases for feature_selection_in
        # 1: True
        assert valid_iris_config.feature_selection_in([Reduction.PCA])
        # 2. False
        assert not valid_iris_config.feature_selection_in([Reduction.NON])

        # Checks if the key exists in the specific set of columns
        # 1. Valid numerical
        assert valid_iris_config.column_is_numeric("sepal-width")

        # 2. Invalid numerical
        assert not valid_iris_config.column_is_numeric("none-such-exists")

        # 3. There are no text columns
        assert not valid_iris_config.column_is_text("none-such-exists") 

        # Check for RFE-usage
        # 1. No, this is Reduction.PCA
        assert not valid_iris_config.use_RFE()

        # 2. Change the feature, and now it'll be true
        valid_iris_config.mode.feature_selection = Reduction.RFE
        assert valid_iris_config

        
        

    def test_smote_and_undersampler(self, valid_iris_config):
        # Using the default both smote and undersampler are False and should return None
        assert valid_iris_config.get_smote() is None
        assert valid_iris_config.get_undersampler() is None
        assert not valid_iris_config.use_imb_pipeline(), "Neither smote nor undersampler means no imb_pipeline"
        
        # Update the config to confirm that it's the right type of class
        valid_iris_config.mode.smote = True
        
        assert valid_iris_config.use_imb_pipeline(), "Smote or undersampler means imb_pipeline"
        valid_iris_config.mode.undersample = True

        assert isinstance(valid_iris_config.get_smote(), SMOTE)
        assert isinstance(valid_iris_config.get_undersampler(), RandomUnderSampler)
        assert valid_iris_config.use_imb_pipeline(), "Smote and undersampler means imb_pipeline"
        
    def test_none_or_positive_int(self, valid_iris_config):
        """ Some values return either an int or None """
        assert valid_iris_config.get_num_selected_features() == 0 # None returns 0
        assert valid_iris_config.get_max_iterations() == 20000
        assert valid_iris_config.get_max_limit() == 150

        # What happens if we give an invalid type?
        attribute = "foo"
        with pytest.raises(ConfigException):
            valid_iris_config.get_none_or_positive_value(attribute)

        attribute = "foo.fum"
        with pytest.raises(ConfigException):
            valid_iris_config.get_none_or_positive_value(attribute)
            
    
    def test_updating_attributes(self):
        default_config = Config()

        # Test set_num_selected_features
        default_config.set_num_selected_features(37)
        assert default_config.get_num_selected_features() == 37

        # Test update_attribute(self, attribute: Union[str, dict], new_value)
        # 1: with valid string attribute
        # a: x.y
        default_config.update_attribute("mode.test_size", 0.7)
        assert default_config.mode.test_size == 0.7
        # b: y
        default_config.update_attribute("name", "test_name")
        assert default_config.name == "test_name"
        # 2: with invalid string attribute
        # a: non-existent
        with pytest.raises(ConfigException):
            default_config.update_attribute("does_not_exist", "test_name")
        # b: non-existent longer
        with pytest.raises(ConfigException):
            default_config.update_attribute("does_not.exist", "test_name")
        # c: non-existent longer
        with pytest.raises(ConfigException):
            default_config.update_attribute("mode.does_not_exist", "test_name")
        # d: too many periods
        with pytest.raises(ConfigException):
            default_config.update_attribute("does.not.exist", "test_name")
        # 3: with valid dict attribute
        default_config.update_attribute({"type":"mode", "name":"test_size"}, 0.5)
        assert default_config.mode.test_size == 0.5
        # 4: with invalid dict attribute format
        with pytest.raises(ConfigException):
            default_config.update_attribute({"name":"test_size"}, "test_name")
        # 5: with invalid dict attribute
        with pytest.raises(ConfigException):
            default_config.update_attribute({"type": "does_not_exist", "name":"test_size"}, "test_name")
        
        # updates = {"algorithm": algorithm, "preprocessor" : pprocessor, "num_selected_features": best_feature_selection}
        #self.handler.config.update_attributes(type="mode", updates=updates)
        # Test update_values
        # 1. Type = None (updates has keys on form "x.y")
        default_config.update_attributes({"name": "new_name", "connection.class_column": "foo"})
        assert default_config.name == "new_name"
        assert default_config.connection.class_column == "foo"

        # 2. type is not none
        default_config.update_attributes({"id_column": "id_column", "class_column": "class_column"}, "connection")
        assert default_config.connection.id_column == "id_column"
        assert default_config.connection.class_column == "class_column"

    def test_get_attribute(self, valid_iris_config):
        # 1: Get name
        assert valid_iris_config.get_attribute("name") == "test"
        # 2. Get connection.id_column
        assert valid_iris_config.get_attribute("connection.id_column") == "id"
        # 3: Exception: x.y.z
        with pytest.raises(ConfigException):
            valid_iris_config.get_attribute("x.y.z")
        # 4: Except does_not_exist
        # a: does_not_exist
        with pytest.raises(ConfigException):
            valid_iris_config.get_attribute("does_not_exist")
        # b: does_not.exist
        with pytest.raises(ConfigException):
            valid_iris_config.get_attribute("does_not.exist")
        # c: mode.does_not_exist
        with pytest.raises(ConfigException):
            valid_iris_config.get_attribute("mode.does_not_exist")

    def test_get_calculated_values(self, valid_iris_config):
        """ Testing simple calculated values """
        expected = "DatabaseOne.ResultTable"
        assert valid_iris_config.get_class_table() == expected

        assert valid_iris_config.get_stop_words_threshold_percentage() == 100

        assert valid_iris_config.get_test_size_percentage() == 20
   
    

    def test_get_named_attributes(self, valid_iris_config):
        """ 
        Several of the functions are just wrappers around getting a single, named attribute.
        This tests them, just to make sure nothing weird is happening
        """
        assert valid_iris_config.get_feature_selection() == Reduction.PCA
        
        assert valid_iris_config.get_test_size() == 0.2

        assert valid_iris_config.is_verbose()

        assert valid_iris_config.get_id_column_name() == "id"

        assert not valid_iris_config.should_train()

        assert valid_iris_config.should_predict()
        
        assert not valid_iris_config.should_display_mispredicted()
        
        assert not valid_iris_config.use_stop_words()

        assert valid_iris_config.get_stop_words_threshold() == 1.0

        assert valid_iris_config.should_hex_encode()

        assert valid_iris_config.use_categorization()

        assert valid_iris_config.get_algorithm() == Algorithm.LDA

        assert valid_iris_config.get_preprocessor() == Preprocess.STA

        # TODO: This is not going to work everywhere, due to the checking of OS
        # For now, comment it out
        # assert valid_iris_config.get_data_username() == "some_fake_name"

        assert valid_iris_config.get_data_catalog() == "DatabaseTwo"

        assert valid_iris_config.get_data_table() == "InputTable"

    def test_get_model_filename(self, valid_iris_config):
        """ Tests the model filename functionality with injected dependency"""
        expected = Path("fake_path\\model\\test.sav")
        real = valid_iris_config.get_model_filename(pwd=Path("fake_path"))
        
        assert real == expected

    def test_clean_config(self, valid_iris_config, bare_iris_config):
        """ gets a stripped down version for saving with the .sav file """
        cleaned_config = valid_iris_config.get_clean_config()

        assert cleaned_config == bare_iris_config

    def test_saving_config(self, tmp_path, valid_iris_config):
        """ Tests saving the config to a temporary directory """
        d = tmp_path / "config"
        d.mkdir()
        p = d / "config.py"
        
        valid_iris_config.save_to_file(p, "some_fake_name")
        assert p.read_text() == get_fixture_content_as_string("test-iris-saved.py")

    def test_load_config_from_model_file(self, valid_iris_config, bare_iris_config, saved_with_valid_iris_config):
        """ Loads config from a .sav file """

        # 1. Exception if there's something wrong, ex missing file
        with pytest.raises(ConfigException):
            Config.load_config_from_model_file("does_not_exist")

        filename = get_fixture_path() / "config-save.sav"
        # 2. Without a config
        new_config = Config.load_config_from_model_file(filename)
        assert new_config == bare_iris_config

        # 3. With a config
        new_config = Config.load_config_from_model_file(filename, valid_iris_config)
        assert new_config == saved_with_valid_iris_config

    def test_load_config_from_module(self, valid_iris_config):
        """ While it uses the load_config_from_module, it mainly checks load_config_2 """
        argv = [
            "test_config.py",
            "-f",
            ".\\fixtures\\test-iris-saved.py"
        ]

        loaded_config = Config.load_config_from_module(argv)

        assert loaded_config == valid_iris_config

    def test_scoring_mechanism(self):
        """ This has two results: string or Callable """

        config = Config() # By default as Scorertype.accuracy
        assert config.get_scoring_mechanism() == "accuracy"

        # Will return a Callable
        config.mode.scoring = Scoretype.mcc
        assert isinstance(config.get_scoring_mechanism(), Callable)

    def test_update_configuration(self, valid_iris_config):
        new_debug = Config.Debug(
            on=True,
            num_rows=125
        )
        
        default_debug = valid_iris_config.debug
        
        assert new_debug != default_debug, "Just making sure these are different"
        valid_iris_config.update_configuration({
            "debug": new_debug
        })

        assert valid_iris_config.debug == new_debug

    def test_model_name(self):
        assert get_model_name(Algorithm.LDA, Preprocess.STA) == "LDA-STA"

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
        sorted_list_default = Detector.get_sorted_list(none_all_first=False)
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
        sorted_list_default = Algorithm.get_sorted_list(none_all_first=False)
        sorted_list_all_first = Algorithm.get_sorted_list()

        assert len(sorted_list_default) == 59
        assert len(sorted_list_all_first) == 59

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

        # When sorted, ADC is first
        assert sorted_list_default[0] == ("Ada Boost Classifier", "ABC")
        
        # When sorted, ALL is first
        assert sorted_list_all_first[0] == ("All", "ALL")

class TestPreprocess:
    """ Tests the Enum Preprocess functions """

    def test_list_callable_preprocessors(self):
        """ There's two cases here, whether it's text_data or not """

        # 1. No textdata
        preprocessors = Preprocess.list_callable_preprocessors(is_text_data=False)

        # 1a: 5 elements (not BIN, but includes NON)
        assert len(preprocessors) == 5

        # 1b: It's a list of tuples
        assert all(isinstance(x,tuple) for x in preprocessors)

        # 1c: Each tuple have two elements
        assert all(len(x) == 2 for x in preprocessors)

        # 1d: The first element in each tuple is an Preprocess enum
        assert all(isinstance(x[0], Preprocess) for x in preprocessors)

        # 1e: This uses the sublist of not-None and only 1 element is None
        callables = [x[1] for x in preprocessors if x[1] is not None]
        assert all(hasattr(x, "fit") and callable(getattr(x, "fit")) for x in callables)
        assert len(callables) == 4

        # 2. Is textdata
        preprocessors = Preprocess.list_callable_preprocessors(is_text_data=True)

        # 2a: 5 elements (includes BIN and NON)
        assert len(preprocessors) == 6

        # 2b: It's a list of tuples
        assert all(isinstance(x,tuple) for x in preprocessors)

        # 2c: Each tuple have two elements
        assert all(len(x) == 2 for x in preprocessors)

        # 2d: The first element in each tuple is an Preprocess enum
        assert all(isinstance(x[0], Preprocess) for x in preprocessors)

        # 2e: This uses the sublist of not-None and only 1 element is None
        callables = [x[1] for x in preprocessors if x[1] is not None]
        assert all(hasattr(x, "fit") and callable(getattr(x, "fit")) for x in callables)
        assert len(callables) == 5

    def test_get_sorted_list(self):
        """ This function gives a list of tuples: (value, name) """
        sorted_list_default = Preprocess.get_sorted_list(none_all_first=False)
        sorted_list_all_first = Preprocess.get_sorted_list()

        assert len(sorted_list_default) == 7
        assert len(sorted_list_all_first) == 7

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

        # When sorted, ALL is first, BIN
        assert sorted_list_default[0] == ("All", "ALL")
        assert sorted_list_default[1] == ("Binarizer", "BIN")
        
        # When sorted, ALL is first, NONE is second
        assert sorted_list_all_first[0] == ("All", "ALL")
        assert sorted_list_all_first[1] == ("None", "NON")

class TestReduction:
    """ Tests the Enum Reduction functions """
    def test_get_sorted_list(self):
        """ This function gives a list of tuples: (value, name) """
        sorted_list_default = Reduction.get_sorted_list(none_all_first=False)
        sorted_list_all_first = Reduction.get_sorted_list()

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

        # When sorted, FICA is first
        assert sorted_list_default[0] == ("Fast Indep. Component Analysis", "FICA")
        
        # When sorted, NONE is first
        assert sorted_list_all_first[0] == ("None", "NON")

class TestScoretype:
    """ Tests the Enum Scoretype functions """
    def test_get_sorted_list(self):
        """ This function gives a list of tuples: (value, name) """
        # For Scoretype there is no difference between these two, as there is no NON or ALL
        sorted_list_default = Scoretype.get_sorted_list(none_all_first=False)
        sorted_list_all_first = Scoretype.get_sorted_list()

        assert sorted_list_default == sorted_list_all_first
        assert len(sorted_list_default) == 11
        
        # They are a list of tuples
        assert all(isinstance(x,tuple) for x in sorted_list_default)
        
        # Each tuple have two elements
        assert all(len(x) == 2 for x in sorted_list_default)
        
        # Each tuple have two strings as elements
        assert all(isinstance(x[0], str) for x in sorted_list_default)
        assert all(isinstance(x[1], str) for x in sorted_list_default)

        # When sorted, accuracy is first
        assert sorted_list_default[0] == ("Accuracy", "accuracy")
