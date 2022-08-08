import os
from typing import Callable
from path import Path
import pytest
from Config import (Algorithm, Config, Preprocess, Reduction, Scoretype,
                    get_model_name, positive_int_or_none, set_none_or_int)
from IAFExceptions import ConfigException
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def get_valid_iris_config(filename: str = None) -> Config:
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
            data_text_columns="",
            data_numerical_columns="sepal-length,sepal-width,petal-length,petal-width",
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
            category_text_columns="",
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
        filename=filename
    )

    return config

def get_bare_iris_config() -> Config:
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
            data_text_columns="",
            data_numerical_columns="sepal-length,sepal-width,petal-length,petal-width",
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
            category_text_columns="",
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

def get_saved_with_valid_iris_config() -> Config:
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
            data_text_columns="",
            data_numerical_columns="sepal-length,sepal-width,petal-length,petal-width",
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
            category_text_columns="",
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

    def test_column_names(self):
        """ This will test known values with various functions that return column names"""
        valid_iris_config = get_valid_iris_config()

        # removes single empty value
        column_names = "a,,b"
        cleaned = valid_iris_config.clean_column_names_list(column_names)
        assert cleaned == ["a", "b"]
        
        # Removes multiple empty values
        column_names = "a,,b,"
        cleaned = valid_iris_config.clean_column_names_list(column_names)
        assert cleaned == ["a", "b"]

        # should return an empty list
        column_names = ""
        assert valid_iris_config.clean_column_names_list(column_names) == []

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

    def test_calculated_booleans(self):
        """ This tests that calculated booleans are calculated correctly """
        valid_iris_config = get_valid_iris_config()
        
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
        assert valid_iris_config.feature_selection_in([Reduction.PCA]) is True
        # 2. False
        assert valid_iris_config.feature_selection_in([Reduction.NON]) is False
        

    def test_smote_and_undersampler(self):
        valid_iris_config = get_valid_iris_config()

        # Using the default both smote and undersampler are False and should return None
        assert valid_iris_config.get_smote() is None
        assert valid_iris_config.get_undersampler() is None

        # Update the config to confirm that it's the right type of class
        valid_iris_config.mode.smote = True
        valid_iris_config.mode.undersample = True

        assert isinstance(valid_iris_config.get_smote(), SMOTE)
        assert isinstance(valid_iris_config.get_undersampler(), RandomUnderSampler)
        
    def test_none_or_positive_int(self):
        """ Some values return either an int or None """
        valid_iris_config = get_valid_iris_config()

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

    def test_get_attribute(self):
        # Using this to know exactly what values to expect
        valid_iris_config = get_valid_iris_config()

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

    def test_get_calculated_values(self):
        """ Testing simple calculated values """
        # Using this to know exactly what values to expect
        valid_iris_config = get_valid_iris_config()
        
        expected = "DatabaseOne.ResultTable"
        assert valid_iris_config.get_class_table() == expected

        assert valid_iris_config.get_stop_words_threshold_percentage() == 100

        assert valid_iris_config.get_test_size_percentage() == 20        
    

    def test_get_named_attributes(self):
        """ 
        Several of the functions are just wrappers around getting a single, named attribute.
        This tests them, just to make sure nothing weird is happening
        """
        valid_iris_config = get_valid_iris_config()

        assert valid_iris_config.get_feature_selection() == Reduction.PCA
        
        assert valid_iris_config.get_test_size() == 0.2

        assert valid_iris_config.is_verbose() is True

        assert valid_iris_config.get_id_column_name() == "id"

        assert valid_iris_config.should_train() is False

        assert valid_iris_config.should_predict() is True
        
        assert valid_iris_config.should_display_mispredicted() is False
        
        assert valid_iris_config.use_stop_words() is False

        assert valid_iris_config.get_stop_words_threshold() == 1.0

        assert valid_iris_config.should_hex_encode() is True

        assert valid_iris_config.use_categorization() is True

        assert valid_iris_config.get_algorithm() == Algorithm.LDA

        assert valid_iris_config.get_preprocessor() == Preprocess.STA

        # TODO: This is not going to work everywhere, due to the checking of OS
        # For now, comment it out
        # assert valid_iris_config.get_data_username() == "some_fake_name"

        assert valid_iris_config.get_data_catalog() == "DatabaseTwo"

        assert valid_iris_config.get_data_table() == "InputTable"

    def test_get_model_filename(self):
        """ Tests the model filename functionality with injected dependency"""
        # Using this to know exactly what values to expect
        valid_iris_config = get_valid_iris_config()
        
        expected = "fake_path\\model\\test.sav"
        real = valid_iris_config.get_model_filename(pwd="fake_path")
        assert str(real) == expected

    def test_clean_config(self):
        """ gets a stripped down version for saving with the .sav file """
        # Using this to know exactly what values to expect
        valid_iris_config = get_valid_iris_config()

        expected_config = get_bare_iris_config()
        
        cleaned_config = valid_iris_config.get_clean_config()

        assert cleaned_config == expected_config

    def test_saving_config(self, tmp_path):
        """ Tests saving the config to a temporary directory """
        valid_iris_config = get_valid_iris_config()

        #with open(get_fixture_path() / "test-iris-saved.py") as f:
        #    content = f.read()

        d = tmp_path / "config"
        d.mkdir()
        p = d / "config.py"
        
        valid_iris_config.save_to_file(p, "some_fake_name")
        assert p.read_text() == get_fixture_content_as_string("test-iris-saved.py")

    def test_load_config_from_model_file(self):
        """ Loads config from a .sav file """

        # 1. Exception if there's something wrong, ex missing file
        with pytest.raises(ConfigException):
            Config.load_config_from_model_file("does_not_exist")

        filename = get_fixture_path() / "config-save.sav"
        # 2. Without a config
        new_config = Config.load_config_from_model_file(filename)
        assert new_config == get_bare_iris_config()

        # 3. With a config
        valid_iris_config = get_valid_iris_config()
        new_config = Config.load_config_from_model_file(filename, valid_iris_config)
        assert new_config == get_saved_with_valid_iris_config()

    def test_load_config_from_module(self):
        """ While it uses the load_config_from_module, it mainly checks load_config_2 """
        argv = [
            "test_config.py",
            "-f",
            ".\\fixtures\\test-iris-saved.py"
        ]

        loaded_config = Config.load_config_from_module(argv)

        assert loaded_config == get_valid_iris_config()

    def test_scoring_mechanism(self):
        """ This has two results: string or Callable """

        config = Config() # By default as Scorertype.accuracy
        assert config.get_scoring_mechanism() == "accuracy"

        # Will return a Callable
        config.mode.scoring = Scoretype.mcc
        assert isinstance(config.get_scoring_mechanism(), Callable)

    def test_positive_int_or_none(self):
        """ Help function to test input params"""
        # 1. None
        assert positive_int_or_none(None) is True
        # 2. 3
        assert positive_int_or_none(3) is True
        # 3. -2
        assert positive_int_or_none(-2) is False
        # 4. 3.5
        assert positive_int_or_none(3.5) is False
    
    def test_set_none_or_int(self):
        """ Help function for loading input params from file """
        # 1: "None"
        assert set_none_or_int("None") is None
        # 2. "5"
        value = set_none_or_int("5")
        assert value == 5 and isinstance(value, int)
        # 3. 4
        assert set_none_or_int(4) == 4
        # 4. -3
        assert set_none_or_int(-3) is None
        # 5. -3.5
        assert set_none_or_int(-3.5) is None
    
    def test_model_name(self):
        assert get_model_name(Algorithm.LDA, Preprocess.STA) == "LDA-STA"

