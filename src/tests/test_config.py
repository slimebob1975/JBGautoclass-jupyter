import os
from typing import Callable

import pytest
from JBGMeta import (Algorithm, AlgorithmTuple, Preprocess,
                    PreprocessTuple, Reduction, ReductionTuple, ScoreMetric)
from Config import Config
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from JBGExceptions import ConfigException
from path import Path


@pytest.fixture
def valid_iris_config() -> Config:
    config = Config(
        Config.Connection(
            odbc_driver="Mock Server",
            host="tcp:database.jbg.mock",
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
            use_metas= False,
            use_stop_words=False,
            specific_stop_words_threshold=1.0,
            hex_encode=True,
            use_categorization=True,
            category_text_columns=[],
            test_size=0.2,
            smote=False,
            undersample=False,
            algorithm=AlgorithmTuple([Algorithm.LDA]),
            preprocessor=PreprocessTuple([Preprocess.NOS]),
            feature_selection=ReductionTuple([Reduction.NOR]),
            num_selected_features=None,
            scoring=ScoreMetric.accuracy,
            max_iterations=20000
        ),
        Config.IO(
            verbose=True,
            model_path="./model",
            model_name="test"
        ),
        Config.Debug(
            on=True,
            data_limit=150
        ),
        name="test"
    )

    return config

@pytest.fixture
def bare_iris_config() -> Config:
    """ This is the bare_iris_config from test_config, so if that one has changed, this one should too """
    config = Config(
        Config.Connection(
            odbc_driver="Mock Server",
            host="tcp:database.jbg.mock",
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
            use_metas=False,
            use_stop_words=False,
            specific_stop_words_threshold=1.0,
            hex_encode=True,
            use_categorization=True,
            category_text_columns=[],
            test_size=0.2,
            smote=False,
            undersample=False,
            algorithm=AlgorithmTuple([Algorithm.LDA]),
            preprocessor=PreprocessTuple([Preprocess.NOS]),
            feature_selection=ReductionTuple([Reduction.NOR]),
            num_selected_features=None,
            scoring=ScoreMetric.accuracy,
            max_iterations=20000
        ),
        Config.IO(
            verbose=True,
            model_path="./model",
            model_name="test"
        ),
        Config.Debug(
            on=True,
            data_limit=150
        ),
        name="iris"
    )

    config.connection.data_catalog = ""
    config.connection.data_table = ""
    config.mode.train = None
    config.mode.predict = None
    config.mode.mispredicted = None
    config.io.model_name = ""
    config.debug.data_limit = 0
    

    return config

@pytest.fixture
def saved_with_valid_iris_config() -> Config:
    config = Config(
        Config.Connection(
            odbc_driver="Mock Server",
            host="tcp:database.jbg.mock",
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
            use_metas= False,
            use_stop_words=False,
            specific_stop_words_threshold=1.0,
            hex_encode=True,
            use_categorization=True,
            category_text_columns=[],
            test_size=0.2,
            smote=False,
            undersample=False,
            algorithm=AlgorithmTuple([Algorithm.LDA]),
            preprocessor=PreprocessTuple([Preprocess.NOS]),
            feature_selection=ReductionTuple([Reduction.NOR]),
            num_selected_features=None,
            scoring=ScoreMetric.accuracy,
            max_iterations=20000
        ),
        Config.IO(
            verbose=True,
            model_path="./model",
            model_name="test"
        ),
        Config.Debug(
            on=True,
            data_limit=150
        ),
        name="iris"
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
        is_numerical_data = True
        force_categorization = False
        is_categorical = False # This is always False for this dataset, no matter the column name

        assert valid_iris_config.is_numerical_data() == is_numerical_data
        assert valid_iris_config.force_categorization() == force_categorization
        assert valid_iris_config.is_categorical(column_name="irrelevant") == is_categorical

        # Checks if the key exists in the specific set of columns
        # 1. Valid numerical
        assert valid_iris_config.column_is_numeric("sepal-width")

        # 2. Invalid numerical
        assert not valid_iris_config.column_is_numeric("none-such-exists")

        # 3. There are no text columns
        assert not valid_iris_config.column_is_text("none-such-exists") 


        
        

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
        expected = "[DatabaseOne].[ResultTable]"
        assert valid_iris_config.get_class_table() == expected

        assert valid_iris_config.get_stop_words_threshold_percentage() == 100

        assert valid_iris_config.get_test_size_percentage() == 20
   
    

    def test_get_named_attributes(self, valid_iris_config):
        """ 
        Several of the functions are just wrappers around getting a single, named attribute.
        This tests them, just to make sure nothing weird is happening
        """
        
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

        # TODO: This is not going to work everywhere, due to the checking of OS
        # For now, comment it out
        # assert valid_iris_config.get_data_username() == "some_fake_name"

        assert valid_iris_config.get_data_catalog() == "[DatabaseTwo]"

        assert valid_iris_config.get_data_table() == "[InputTable]"

    def test_get_model_filename(self, valid_iris_config):
        """ Tests the model filename functionality with injected dependency"""
        expected = Path("fake_path\\model\\test.sav")
        real = valid_iris_config.get_model_filename(pwd=Path("fake_path"))
        
        assert real == expected

    def test_clean_config(self, valid_iris_config, bare_iris_config):
        """ gets a stripped down version for saving with the .sav file """
        cleaned_config = valid_iris_config.get_clean_config()
        #TODO: Tuples need to be able to be compared so that if they contain the 
        # same Algorithms/etc (even if different orders), they are considered equal
        assert cleaned_config == bare_iris_config

    def test_saving_config(self, tmp_path, valid_iris_config):
        """ Tests saving the config to a temporary directory """
        d = tmp_path / "config"
        d.mkdir()
        p = d / "config.py"
        
        valid_iris_config.save_to_file(p, "some_fake_name")
        # TODO: Yeah, no. Probably the same issue as above + needing to get the new file
        assert p.read_text() == get_fixture_content_as_string("test-iris-saved.py")

    def test_load_config_from_model_file(self, valid_iris_config, bare_iris_config, saved_with_valid_iris_config):
        """ Loads config from a .sav file """

        # 1. Exception if there's something wrong, ex missing file
        with pytest.raises(ConfigException):
            Config.load_config_from_model_file("does_not_exist")

        filename = get_fixture_path() / "config-save.sav"
        # 2. Without a config
        new_config = Config.load_config_from_model_file(filename)
        
        # TODO: see above
        assert new_config == bare_iris_config

        # 3. With a config
        new_config = Config.load_config_from_model_file(filename, valid_iris_config)
        # TODO: see above
        assert new_config == saved_with_valid_iris_config

    def test_load_config_from_module(self, valid_iris_config):
        """ While it uses the load_config_from_module, it mainly checks load_config_2 """
        argv = [
            "test_config.py",
            "-f",
            ".\\fixtures\\test-iris-saved.py"
        ]

        loaded_config = Config.load_config_from_module(argv)
        # TODO: see above
        assert loaded_config == valid_iris_config

    def test_scoring_mechanism(self):
        """ This returns a Callable """

        config = Config() # By default as Scorertype.accuracy
        assert isinstance(config.get_scoring_mechanism(), Callable)

    def test_update_configuration(self, valid_iris_config):
        new_debug = Config.Debug(
            on=True,
            data_limit=125
        )
        
        default_debug = valid_iris_config.debug
        
        assert new_debug != default_debug, "Just making sure these are different"
        valid_iris_config.update_configuration({
            "debug": new_debug
        })

        assert valid_iris_config.debug == new_debug



