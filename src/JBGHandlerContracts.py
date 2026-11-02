"""Backend-facing structural contracts used by the JBGHandler orchestration layer.

Keep the legacy exports in JBGHandler for existing consumers.  These Protocols
are interface descriptions only, not concrete implementations.
"""
from __future__ import annotations

from typing import Callable, Protocol, Union

class Logger(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""
    def print_info(self, *args, print_always: bool = False) -> None:
        """printing info"""

    def print_prediction_report(self,
        accuracy_score: float,
        confusion_matrix: np.ndarray,
        class_labels: list,
        classification_matrix: dict,
        sample_rates: tuple[str, str] = None) -> None:
        """ Printing out info about the prediction"""

    def print_progress(self, message: str = None, percent: float = None) -> None:
        """Printing progress"""

    def print_formatted_info(self, message: str) -> None:
        """ Printing info with """

    def investigate_dataset(self, dataset: pd.DataFrame, class_name: str, show_class_distribution: bool = True, show_statistics: bool = True) -> bool:
        """ Print information about dataset """

    def print_warning(self, *args) -> None:
        """ print warning """

    def print_test_performance(self, listOfResults: list, cross_validation_filepath: str) -> None:
        """
            Prints out the matrix of model test performance, given a list of results
            Both to screen and CSV
        """

    def print_result_line(self, result: list, ending: str = '\r') -> None:
        """ Prints information about a specific result line """

    def clear_last_printed_result_line(self):
        """ Clears a line that's printed using \r """

    def abort_cleanly(self, message: str) -> None:
        """ Exits the process """

    def print_dragon(self, exception: Exception) -> None:
        """ Type of Unhandled Exceptions, to handle them for the future """

    def update_progress(self, percent: float, message: str = None) -> float:
        """ Tracks the progress through the run """

    def print_key_value_pair(self, key: str, value, print_always: bool = False) -> None:
        """ Prints out '<key>: <value>'"""

    def print_code(self, key: str, code: str) -> None:
        """ Prints out a text with a (in output) code-tagged end """

    def print_query(self, type: str, query: str) -> None:
        """Show an SQL query with persistent-log privacy handling."""

    def print_privacy_safe(self, full_message: str, safe_message: str, level: str = "INFO",
                           html_function=None, print_always: bool = False):
        """Show full operator detail while persisting a privacy-safe summary."""

    def update_inline_progress(self, key: str, current_count: int, terminal_text: str) -> None:
        """ Updates progress bars within the script"""

    def start_inline_progress(self, key: str, description: str, final_count: int, tooltip: str) -> None:
        """ This will overwrite any prior bars with the same key """

    def reset_inline_progress(self, key: str) -> None:
        """ If the progress bar needs to be set back to 0"""

    def end_inline_progress(self, key: str, set_100: bool = True) -> None:
        """ Ensures that any loose ends are tied up after the progress is done """

class DataLayer(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""

    def get_dataset(self, num_rows: int = None) -> list:
        """ Gets the needed data from the database """

    def save_prediction_data(self, results: list, class_rate_type: RateType, model_name: str, class_labels, create_tables: bool = True, test_run: int = 0) -> dict:
        """ Saves prediction data, with the side-effect of creating the tables if necessary """

    def get_data_query(self, num_rows: int) -> str:
        """ Prepares the query for getting data from the database """

    def get_table_columns(self, database: str, table: str) -> dict:
        """Return source-column SQL datatypes for saved-model compatibility checks."""

class Config(Protocol):
    # Methods to hide implementation of Config
    def get_categorical_text_column_names(self) -> list[str]:
        """ Gets the specified categorical text columns"""

    def set_num_selected_features(self, num_features: int) -> None:
        """ Updates the config with the number """

    def column_is_text(self, column: str) -> bool:
        """ Checks if the column is text based """

    def use_feature_selection(self) -> bool:
        """True or False"""

    def feature_selection_in(self, selection: list[Reduction]) -> bool:
        """ Checks if the selection is one of the given Reductions"""

    def get_num_selected_features(self) -> int:
        """ Gets the number of selected features--0 if None"""

    def get_test_size(self) -> float:
        """ Gets the test_size """

    def get_data_limit(self) -> int:
        """ Get the data limit"""

    def get_max_iterations(self) -> int:
        """ Get max iterations """

    def get_column_names(self) -> list[str]:
        """ Gets the column names based on connection columns """

    def get_text_column_names(self) -> list[str]:
        """ Gets the specified text columns"""

    def get_class_column_name(self) -> str:
        """ Gets the name of the column"""

    def get_id_column_name(self) -> str:
        """ Gets the name of the ID column """

    def is_categorical(self, column_name) -> bool:
        """ Returns if a specific column is categorical """

    def use_imb_pipeline(self) -> bool:
        """ Returns true if oversampler or undersampler is used """

    def should_train(self) -> bool:
        """ Returns if this is a training config """

    def should_predict(self) -> bool:
        """ Returns if this is a prediction config """

    def should_display_mispredicted(self) -> bool:
        """ Returns if this is a misprediction config """

    def use_stop_words(self) -> bool:
        """ Returns whether stop words should be used """

    def get_ngram_range(self) -> NgramRange:
        """ Returns the threshold fo the stop words """

    def should_hex_encode(self) -> bool:
        """ Returns whether dataset should be hex encoded """

    def use_categorization(self) -> bool:
        """ Returns if categorization should be used """

    def get_callable_oversamplers(self) -> Union[Oversampling, None]:
        """ Gets the oversampler for the model, or None if it shouldn't be """

    def use_oversampling(self) -> bool:
        """ Simple check if it's used or note """

    def set_oversampler(self, oversampler: Oversampling) -> None:
        """ Sets oversampler """

    def get_undersampler(self) -> Union[Undersampling, None]:
        """ Gets the UnderSampler, or None if there should be none"""

    def set_undersampler(self, undersampler: Undersampling) -> None:
        """ Sets undersampler """

    def update_attributes(self, updates: dict,  type: str = None) -> None:
        """ Updates several values inside the config """

    def get_scoring_mechanism(self) -> Union[str, Callable]:
        """ While the actual function is in the mechanism, this allows us to hide where Scoring is """

    def get_clean_config(self):
        """ Extracts the config information to save with a model """

    def get_callable_reductions(self, num_samples: int, num_features: int, num_selected_features: int = None) -> list:
        """ Returns callable reductions from mode.feature_selection """


    def get_callable_algorithms(self, size: int, max_iterations: int) -> list:
        """ Returns callable algorithms from mode.algorithm """


    def get_callable_preprocessors(self) -> list:
        """ Returns callable preprocessors from mode.algorithm """
