from typing import Callable, Protocol, Union
from numpy import ndarray
import pandas
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from Config import Algorithm, Preprocess, Reduction
from IAFExceptions import HandlerException

from IAFHandler import IAFHandler, DatasetHandler, Model, ModelHandler, PredictionsHandler
# One class per class in the module

class Logger():
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""
    def print_info(self, *args) -> None:
        """printing info"""

    def print_progress(self, message: str = None, percent: float = None) -> None:
        """Printing progress"""

    def print_components(self, component, components, exception = None) -> None:
        """ Printing Reduction components"""

    def print_formatted_info(self, message: str) -> None:
        """ Printing info with """

    def investigate_dataset(self, dataset: pandas.DataFrame, show_class_distribution: bool = True, show_statistics: bool = True) -> bool:
        """ Print information about dataset """
    
    def print_warning(self, *args) -> None:
        """ print warning """

    def print_table_row(self, items: list[str], divisor: str = None) -> None:
        """ Prints row of items, with optional divisor"""

    def abort_cleanly(self, message: str) -> None:
        """ Exits the process """

class Config():
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

    def get_text_column_names(self) -> list[str]:
        """ Gets the specified text columns"""

    def get_numerical_column_names(self) -> list[str]:
        """ Gets the specified numerical columns"""

    def get_class_column_name(self) -> str:
        """ Gets the name of the column"""

    def get_id_column_name(self) -> str:
        """ Gets the name of the ID column """

    def is_categorical(self, column_name) -> bool:
        """ Returns if a specific column is categorical """

    def should_train(self) -> bool:
        """ Returns if this is a training config """

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

class DataLayer():
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""

    def get_dataset(self, num_rows: int, train: bool, predict: bool):
        """ Get the dataset, query and number of rows"""

    def save_data(self, results: list, class_rate_type: str, model: Model, config: Config)-> int:
        """ Saves classification for X_unknown in classification database """

        return 1

    def get_sql_command_for_recently_classified_data(self, num_rows: int) -> str:
        """ What name says """

        return "SQL"

@pytest.fixture
def default_handler() -> IAFHandler:
    handler = IAFHandler(datalayer=DataLayer(), config=Config(), logger=Logger(), progression={"progress": 0.0})

    return handler

class TestHandler():
    """ The main class """

    def add_handlers(self, handler: IAFHandler) -> None:
        handler.add_handler("dataset")
        handler.add_handler("predictions")
        handler.add_handler("model")

    def test_add_handler(self, default_handler):
        """ This has two cases: either it returns a valid <type>Handler or throws an exception"""
    
        with pytest.raises(HandlerException):
            default_handler.add_handler("does_not_exist")

        assert isinstance(default_handler.add_handler("dataset"), DatasetHandler)

    def test_get_handler(self, default_handler):
        """ Two cases: returns a valid <type>Handler or throws an exception """

        with pytest.raises(HandlerException):
            default_handler.get_handler("does_not_exist")

        default_handler.add_handler("dataset")
        
        assert isinstance(default_handler.get_handler("dataset"), DatasetHandler)

    def test_save_classification_data(self, default_handler):
        """ Tests that the classification data gets saved properly"""

        # 1. One (or more) of the handlers are not set, HandlerException
        with pytest.raises(HandlerException):
            default_handler.save_classification_data()

        # 2. Upredicted keys are not set in datasethandler
        self.add_handlers(default_handler)
        with pytest.raises(HandlerException):
            default_handler.save_classification_data()

        # 3. Should hopefully work
        dh = default_handler.get_handler("Dataset")
        keys = pandas.Series(dtype="float64") # This is to silence a warning about deprecation--the actual type is irrelevant
        dh.set_unpredicted_keys(keys)
        default_handler.save_classification_data()

    def test_update_progress(self, default_handler):
        """ Two cases, percentage or percentage + message """
        assert default_handler.update_progress(10) == 10.0

        assert default_handler.update_progress(10, "a message is printed") == 20.0


