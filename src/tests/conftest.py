import os
import pytest
from Config import Config
from JBGMeta import AlgorithmTuple, Algorithm, PreprocessTuple, Preprocess, ReductionTuple, Reduction, ScoreMetric
import SQLDataLayer
import JBGHandler
from typing import Callable, Union
from path import Path
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from JBGHandler import JBGHandler, DatasetHandler, Model, ModelHandler, PredictionsHandler

def get_fixture_path() -> Path:
    pwd = os.path.dirname(os.path.realpath(__file__))
    return Path(pwd) / "fixtures"

def get_fixture_content_as_string(filename: str) -> str:
    """ Returns the content of one of the fixtures """
    with open(get_fixture_path() / filename) as f:
            content = f.read()

    return content

class MockLogger():
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""

    def print_info(self, *args) -> None:
        """printing info"""

    def print_progress(self, message: str = None, percent: float = None) -> None:
        """Printing progress"""

    def print_components(self, component, components, exception=None) -> None:
        """ Printing Reduction components"""

    def print_formatted_info(self, message: str) -> None:
        """ Printing info with """

    def investigate_dataset(self, dataset, show_class_distribution: bool = True, show_statistics: bool = True) -> bool:
        """ Print information about dataset """

    def print_warning(self, *args) -> None:
        """ print warning """

    def print_table_row(self, items: list[str], divisor: str = None) -> None:
        """ Prints row of items, with optional divisor"""

    def abort_cleanly(self, message: str) -> None:
        """ Exits the process """

    def print_percentage_checked(self, text: str, old_percent, percent_checked) -> None:
        """ Prints using the normal print() """

    def print_percentage(self, text: str, old_percent: float, percent_checked: float) -> None:
        """ Prints using the normal print() """

    def print_dragon(self, exception: Exception) -> None:
        """ Type of Unhandled Exceptions, to handle them for the future """

    def print_linebreak(self) -> None:
        """ Important after using \r for updates """
        

class MockConfig():
    # Methods to hide implementation of Config
    def set_num_selected_features(self, num_features: int) -> None:
        """ Updates the config with the number """

    def get_model_filename(self, pwd: str = None) -> str:
        """ Set the filename based on prediction or training """

    def is_text_data(self) -> bool:
        """True or False"""
    
    def is_numerical_data(self) -> bool:
        """True or False"""

    def force_categorization(self) -> bool:
        """True or False"""

    def column_is_numeric(self, column: str) -> bool:
        """ Checks if the column is numerical """
        
    def column_is_text(self, column: str) -> bool:
        """ Checks if the column is text based """

    def use_feature_selection(self) -> bool:
        """True or False"""

    def feature_selection_in(self, selection: list[Reduction]) -> bool:
        """ Checks if the selection is one of the given Reductions"""

    def get_feature_selection(self) -> Reduction:
        """ Returns the chosen feature selection """

    def get_num_selected_features(self) -> int:
        """ Gets the number of selected features--0 if None"""

    def get_test_size(self) -> float:
        """ Gets the test_size """

    def get_max_limit(self) -> int:
        """ Get the max limit. Name might change depending on GUI names"""

    def get_max_iterations(self) -> int:
        """ Get max iterations """

    def is_verbose(self) -> bool:
        """ Returns what the io.verbose is set to"""

    def get_column_names(self) -> list[str]:
        """ Gets the column names based on connection columns """
        return ["name", "age", "test_class", "test_id"]

    def get_text_column_names(self) -> list[str]:
        """ Gets the specified text columns"""
        return ["name"]

    def get_numerical_column_names(self) -> list[str]:
        """ Gets the specified numerical columns"""
        return ["age"]

    def get_class_column_name(self) -> str:
        """ Gets the name of the column"""

        return "test_class"

    def get_id_column_name(self) -> str:
        """ Gets the name of the ID column """

        return "test_id"

    def get_data_column_names(self) -> list[str]:
        """ Gets data columns, so not Class or ID """
        return self.get_text_column_names() + self.get_numerical_column_names()

    def is_categorical(self, column_name: str) -> bool:
        """ Returns if a specific column is categorical """
        return column_name == "is_categorical" # To be able to mock the response

    def should_train(self) -> bool:
        """ Returns if this is a training config """
        return True

    def should_predict(self) -> bool:
        """ Returns if this is a prediction config """

    def should_display_mispredicted(self) -> bool:
        """ Returns if this is a misprediction config """

    def use_stop_words(self) -> bool:
        """ Returns whether stop words should be used """

    def get_stop_words_threshold(self) -> float:
        """ Returns the threshold fo the stop words """

    def should_hex_encode(self) -> bool:
        """ Returns whether dataset should be hex encoded """

    def use_categorization(self) -> bool:
        """ Returns if categorization should be used """
        return True

    def get_smote(self) -> Union[SMOTE, None]:
        """ Gets the SMOTE for the model, or None if it shouldn't be """
        
    def get_undersampler(self) -> Union[RandomUnderSampler, None]:
        """ Gets the UnderSampler, or None if there should be none"""

    def update_attributes(self, updates: dict,  type: str = None) -> None:
        """ Updates several values inside the config """

    def get_scoring_mechanism(self) -> Union[str, Callable]:
        """ While the actual function is in the mechanism, this allows us to hide where Scoring is """

    def get_algorithm(self) -> Algorithm:
        """ Get algorithm from Config"""


    def get_preprocessor(self) -> Preprocess:
        """ get preprocessor from Config """

class MockDataLayer():
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""

    def get_dataset(self, num_rows: int = None):
        """ Get the dataset and query """

        data = [
            ["Karan",23, "odd", 1],
            ["Rohit",22, "even", 2],
            ["Sahil",21, "odd", 3],
            ["Aryan",24, "even", 4]
        ]

        return data



    def save_data(self, results: list, class_rate_type: str, model: Model)-> int:
        """ Saves classification for X_unknown in classification database """

        return 1

    def get_sql_command_for_recently_classified_data(self, num_rows: int) -> str:
        """ What name says """

        return "SQL"


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

@pytest.fixture
def default_sqldatalayer(valid_iris_config) -> SQLDataLayer.DataLayer:
    return SQLDataLayer.DataLayer(config=valid_iris_config, logger=MockLogger(), validate=False)


@pytest.fixture
def default_handler() -> JBGHandler:
    handler = JBGHandler(datalayer=MockDataLayer(), config=MockConfig(), logger=MockLogger(), progression={"progress": 0.0})

    return handler

@pytest.fixture
def default_dataset_handler(default_handler) -> DatasetHandler:
    return DatasetHandler(handler=default_handler)

@pytest.fixture
def filled_dataset_handler(default_handler) -> DatasetHandler:
    dh = DatasetHandler(handler=default_handler)
    dh.read_in_data()

    return dh

@pytest.fixture
def default_model_handler(default_handler) -> ModelHandler:
    return ModelHandler(handler=default_handler)


@pytest.fixture
def default_predictions_handler(default_handler) -> PredictionsHandler:
    return PredictionsHandler(handler=default_handler)