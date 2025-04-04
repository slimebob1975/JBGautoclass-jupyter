from __future__ import annotations
from pathlib import Path

import dill
import time
import psutil
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from math import ceil
from typing import Callable, Protocol, Union, Any

import langdetect
import numpy as np
import pandas as pd
from lexicalrichness import LexicalRichness
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, make_scorer, f1_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split, GridSearchCV)
from tensorflow import keras
from imblearn.pipeline import Pipeline
from stop_words import get_stop_words

from Config import Config
from JBGMeta import (Algorithm, Library, Preprocess, Reduction, RateType, Estimator, Transform,
                     Oversampling, Undersampling, NgramRange)
from JBGExceptions import (DatasetException, MissingScorerException, ModelException, HandlerException, ModelInitializationException, 
    UnstableModelException, PipelineException)
from JBGTransformers import MLPKerasClassifier, TextDataToNumbersConverter
from JBGDarkNumbers import compute_dark_numbers
import Helpers

GIVE_EXCEPTION_TRACEBACK = False

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

        
@dataclass
class JBGHandler:
    datalayer: DataLayer
    config: Config
    logger: Logger
    
    handlers: dict = field(default_factory=dict, init=False)
    
    # This is defined this way so that VS code can find the different methods
    def add_handler(self, name: str) -> Union[DatasetHandler, ModelHandler, PredictionsHandler]:
        """ Returns the newly created handler"""
        name = name.lower()  # Just in case one wants to capitalize it or such
        create = f"create_{name}_handler"

        if hasattr(self, create) and callable(func := getattr(self, create)):
            self.handlers[name] = func()
            return self.handlers[name]
        else:
            raise HandlerException(f"Handler of type {name} not implemented")
            

    def create_dataset_handler(self) -> DatasetHandler:
        return DatasetHandler(handler=self)

    def create_model_handler(self) -> ModelHandler:
        return ModelHandler(handler=self)

    def create_predictions_handler(self) -> PredictionsHandler:
        return PredictionsHandler(handler=self)
    

    def get_handler(self, name: str) -> Union[DatasetHandler, ModelHandler, PredictionsHandler]:
        name = name.lower()
        if name in self.handlers:
            return self.handlers[name]

        raise HandlerException(f"{name.capitalize()}Handler does not exist")

    @property
    def read_data_query(self) -> str:
        """ Property to get the query for reading data """
        
        return self.datalayer.get_data_query(self.config.get_data_limit())

    
    def get_dataset(self):
        """ By putting this here, Datalayer does not need to be supplied to Dataset Handler"""
        data =  self.datalayer.get_dataset()
        
        return data
   
    def save_predictions(self) -> dict:
        """ Save new predictions for X_unknown in prediction tables """
        try:
            dh = self.get_handler("dataset")
            ph = self.get_handler("predictions")
            mh = self.get_handler("model")
        except HandlerException as e:
            raise e

        return ph.save_predictions(dh, mh)
    
    def update_progress(self, percent: float, message: str = None) -> float:
        """ Wrapper for the logger update_progress """
        self.logger.update_progress(percent, message)

@dataclass
class DatasetHandler:
    STANDARD_LANG = "sv"
    LIMIT_IS_CATEGORICAL = 30
    
    handler: JBGHandler
    dataset: pd.DataFrame = field(init=False)
    classes: list[str] = field(init=False)
    keys: pd.Series = field(init=False)
    unpredicted_keys: pd.Series = field(init=False)
    X_original: pd.DataFrame = field(init=False)
    Y_original: pd.DataFrame = field(init=False)
    X: pd.DataFrame = field(init=False)
    Y: pd.Series = field(init=False)
    X_train: pd.DataFrame = field(init=False)
    X_validation: pd.DataFrame = field(init=False)
    Y_train: pd.DataFrame = field(init=False)
    Y_validation: pd.DataFrame = field(init=False)
    Y_prediction: pd.DataFrame = field(init=False)
    X_prediction: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        """ Empty for now """
    
    # Sets the unpredicted keys
    def set_unpredicted_keys(self, keys: pd.Series) -> None:
        self.unpredicted_keys = keys

    def load_data(self, data: list):
        """" Validates and processes the dataset loaded into the handler """
        # Set the column names of the data array
        column_names = self.handler.config.get_column_names()
        class_column = self.handler.config.get_class_column_name()
        id_column = self.handler.config.get_id_column_name()
        
        # TODO: validate_dataset should probably do a report of potentional issues, or lead into the function that does
        # Warning, not error
        dataset = self.validate_dataset(data, column_names, class_column)
        self.handler.logger.print_key_value_pair("Dataset memory usage", Helpers.bytes_to_suffix(dataset.memory_usage(deep=True).sum()))
        
        dataset = self.shuffle_dataset(dataset)
        
        self.dataset, self.keys = self.split_keys_from_dataset(dataset, id_column)
        
        # Extract unique class labels from dataset
        self.classes = Helpers.clean_list(self.dataset[class_column].tolist())

        # Investigare dataset
        if self.handler.config.should_train():
            self.handler.logger.investigate_dataset(self.dataset, class_column) # Returns True if the investigation/printing was not suppressed
        
    def validate_dataset(self, data: list, column_names: list, class_column: str) -> pd.DataFrame:
        dataset = pd.DataFrame(data, columns = column_names)
        
        # Make sure the class column is a categorical variable by setting it as string
        try:
            dataset.astype({class_column: 'str'}, copy=False)
        except Exception as e:
            self.handler.logger.print_dragon(exception=e)
            raise DatasetException(f"Could not convert class column {class_column} to string variable: {e}")
                    
        # Make an extensive search through the data for any inconsistencies (like NaNs and NoneType). 
        # Also convert datetime numerical variables to ordinals, i.e., convert them to the number of days 
        # or similar from a certain starting point, if any are left after the conversion above.
        number_data_columns = len(dataset.columns) - 1
        column_number = 0
        progress_key = "validate_dataset"
        self.handler.logger.start_inline_progress(progress_key, "Validation progress", number_data_columns, "Percent data checked of fetched")
        try:
            for key, column in dataset.items():
                if key != class_column:
                            
                    # TODO: If possible, parallelize this call over available CPU:s
                    dataset[key] = self.validate_column(key, column)
                    
                    column_number += 1
                    self.handler.logger.update_inline_progress(progress_key, column_number, "Data checked of fetched")
                    
        except Exception as ex:
            self.handler.logger.print_dragon(exception=ex)
            print(ex)
            raise DatasetException(f"Something went wrong in inconsistency check at column {key}: {ex}")
        
        self.handler.logger.end_inline_progress(progress_key)
        return dataset

    def validate_column(self, key: str, column: pd.Series) -> pd.Series:
        
        column_is_text = self.handler.config.column_is_text(key)
        return column.apply(self.sanitize_value, convert_dtype = True, args = (column_is_text,))
        
    def shuffle_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """ Impossible to test, due to random being random """
        # Shuffle the upper part of the data, such that all already classified material are
        # put in random order keeping the unclassified material in the bottom
        # Notice: perhaps this operation is now obsolete, since we now use a NEWID() in the 
        # data query above
        # Addendum: När man får in data så vill man att de olika klasserna i träningsdata 
        # ska vara jämt utspridda över hela träningsdata, inte först 50 iris-setosa och sen 
        # 50 iris-virginica, etc. Utan man vill ha lite slumpmässig utspridning. 
        # Kommentera bort rnd-mojen och kolla med att köra dataset.head(20) eller något sånt, 
        # om du tycker att class-variablen är hyfsat jämt utspridd för iris före och efter bortkommenteringen, ta bort rnd
        self.handler.logger.print_formatted_info("Shuffling data")
        num_un_pred = self.get_num_unpredicted_rows(dataset) 
        num_lines = len(dataset.index)
        dataset['rnd'] = np.concatenate([np.random.rand(num_lines - num_un_pred), [num_lines]*num_un_pred])
        dataset.sort_values(by = 'rnd', inplace = True )
        dataset.drop(['rnd'], axis = 1, inplace = True )

        return dataset
    
    def split_keys_from_dataset(self, dataset: pd.DataFrame, id_column: str) -> tuple[pd.DataFrame, pd.Series]:
        # Use the unique id column from the data as the index column and take a copy, 
        # since it will not be used in the classification but only to reconnect each 
        # data row with classification results later on
        try:
            keys = dataset[id_column].copy(deep = True).apply(Helpers.get_rid_of_decimals)
        except Exception as e:
            self.handler.logger.print_dragon(exception=e)
            raise DatasetException(f"Could not convert to integer: {e}")
            
        
        try:
            dataset.set_index(keys.astype('int64'), drop=False, append=False, inplace=True, verify_integrity=False)
        except Exception as e:
            self.handler.logger.print_dragon(exception=e)
            raise DatasetException(f"Could not set index for dataset: {e}")
        
        dataset = dataset.drop([id_column], axis = 1)
        
        return dataset, keys

    def sanitize_value(self, value, column_is_text: bool) -> Union[str, int, float]:
        """ Massages a value into a proper value """
        
        # Set NoneType objects as zero or empty strings
        if value is None:
            return "" if column_is_text else 0

        if column_is_text:
            # TODO: I'm unsure on how this should be done. What values breaks this so badly?
            # Helpers.is_str() should maybe check the type of the value?
            # Set text values that cannot be casted as strings to empty strings
            if type(value) != str and not Helpers.is_str(value):
                return ""

            # Remove line breaks and superfluous blank spaces from text strings
            return " ".join(value.split())

        # Only numerical values left now
        # Convert datetime values to ordinals
        if datetime_value := Helpers.get_datetime(value):
            return datetime.toordinal(datetime_value)
        

        # Set remaining numerical values that cannot be casted as integer or floating point numbers to zero, i.e., do not
        # take them into account
        if not (Helpers.is_int(value) or Helpers.is_float(value)):
            return 0


        return float(value)
    
    # Text handling a more nice way
    def convert_text_and_categorical_features(self, model: Model):
        
        # Get the old converter even if None
        ttnc = model.text_converter
        
        # The model has no text converter means we need to train a new one
        if self.handler.config.should_train():
            try:
                ttnc = TextDataToNumbersConverter(
                    text_columns=self.handler.config.get_text_column_names(), 
                    category_columns=self.handler.config.get_categorical_text_column_names(), \
                    limit_categorize=TextDataToNumbersConverter.LIMIT_IS_CATEGORICAL, \
                    language=TextDataToNumbersConverter.STANDARD_LANGUAGE, \
                    stop_words=self.handler.config.use_stop_words(), \
                    df=1.0, \
                    ngram_range=(self.handler.config.get_ngram_range()).ngram_range, \
                    use_encryption=self.handler.config.should_hex_encode() \
                )
            
            except Exception as ex:
                self.handler.logger.print_dragon(ex)
                raise Exception(f"Could not initiate text converter object: {str(ex)}")# from ex 
            
            # Fit converter to training data only, which is important for not getting to optimistic results
            ttnc.fit_transform(self.X_train)
            
            # Apply converter to the whole of X (for future references) and the parts X_train and X_validation
            self.X = ttnc.transform(self.X)
            self.X_train = ttnc.transform(self.X_train)
            self.X_validation = ttnc.transform(self.X_validation)

            self.handler.logger.print_key_value_pair("Number of features after text and category conversion", self.X.shape[1])
            
        # If we should predict, apply converter even to X_prediction part of data
        if self.handler.config.should_predict():
            if self.X_prediction is None:
                self.handler.logger.print_warning("X_prediction is not set and cannot be text-converted")
            else:
                self.X_prediction = ttnc.transform(self.X_prediction)
            

        return ttnc
    
    
    def create_X(self, frames: list[pd.DataFrame], index: pd.Int64Index = None) -> pd.DataFrame:
        """ Creates a dataframe from text and numerical data """
        if index is None:
            index = self.dataset.index
        
        X = pd.DataFrame()
        for df in frames:
            X = self.concat_with_index(X, df, index)
        
        return X

    def concat_with_index(self, X: pd.DataFrame, concat: pd.DataFrame, index: pd.Int64Index) -> pd.DataFrame:
        """ The try/except ensures that the dataframe added has the right number of rows """
        try:
            concat.set_index(index, drop=False, append=False, inplace=True, verify_integrity=False)
        except ValueError:
            return X
        
        return pd.concat([concat, X], axis = 1)

    # Separate data in X and Y and in parts where classification is known and not known.
    # X and Y are further split in training and validation part in split_dataset
    def separate_dataset(self) -> None:
        
        num_un_pred = self.get_num_unpredicted_rows()
        self.unpredicted_keys = self.keys[-num_un_pred:]
        class_column = self.handler.config.get_class_column_name()
        
        if num_un_pred > 0:
            self.Y = self.dataset.head(-num_un_pred)[class_column]
            self.X = self.dataset.head(-num_un_pred).drop([class_column], axis = 1, inplace=False)
            self.Y_prediction = self.dataset.tail(num_un_pred)[class_column]
            self.X_prediction = self.dataset.tail(num_un_pred).drop([class_column], axis = 1, inplace=False)
        else:
            self.Y = self.dataset[class_column]
            self.X = self.dataset.drop([class_column], axis = 1, inplace=False)
            self.Y_prediction = None
            self.X_prediction = None
        
        self.handler.logger.print_key_value_pair("Data rows with known class label", self.X.shape[0])
        self.handler.logger.print_key_value_pair("Data rows with unknown class label", num_un_pred)
        # Original training+validation data are stored for later reference
        self.X_original = self.X.copy(deep=True)
        self.Y_original = self.Y.copy(deep=True)
        
                
    # Use the bag of words technique to convert text corpus into numbers
    def word_in_a_bag_conversion(self, dataset: pd.DataFrame, model: Model ) -> tuple:

        # Start working with datavalues in array
        X = dataset.values
        
        count_vectorizer = self.get_count_vectorizer_from_dataset(model.count_vectorizer, X)
        
        # Mask all material by encryption (optional)
        # När vi skapar våra count_vectorizers så använder vi kryptering på X eller inte beroende på användarens val. 
        # När vi använder våra count_vectorizers på X måste vi vara konsekventa. Är X inte krypterat då också, så kommer 
        # count_vectorizers att vara tränat på krypterat material, men hittar ingenting i X som den känner igen.
        if (self.handler.config.should_hex_encode()):
            X = Helpers.do_hex_base64_encode_on_data(X)
        # Do the word in a bag now
        X = count_vectorizer.transform(X)
        
        tfid_transformer = self.get_tfid_transformer_from_dataset(model.tfid_transformer, X)
        
        # Generate the sequences
        X = (tfid_transformer.transform(X)).toarray()
        
        return pd.DataFrame(X), count_vectorizer, tfid_transformer

    def get_tfid_transformer_from_dataset(self, tfid_transformer: TfidfTransformer, dataset: np.ndarray) -> TfidfTransformer:
        """ Generate frequencies, assuming tfid_transformer is not yet set """
        if tfid_transformer is not None:
            return tfid_transformer

        # Also generate frequencies instead of occurences to normalize the information.
        tfid_transformer = TfidfTransformer(use_idf = False)
        tfid_transformer.fit(dataset)

        return tfid_transformer

    def  get_count_vectorizer_from_dataset(self, count_vectorizer: CountVectorizer, dataset: np.ndarray) -> CountVectorizer:
        """ Transform things, if there is no count_vectorizer """
        if count_vectorizer is not None:
            return count_vectorizer

        # Find actual languange if stop words are used
        my_language = None
        if self.handler.config.use_stop_words():
            try:
                my_language = langdetect.detect(' '.join(dataset))
            except Exception as e:
                my_language = self.STANDARD_LANG
                self.handler.logger.print_warning(f"Language could not be detected automatically: {e}. Fallback option, use: {my_language}.")
            else:
                self.handler.logger.print_key_value_pair("Detected language is", my_language)

        # Calculate the lexical richness
        try:
            lex = LexicalRichness(' '.join(dataset)) 
            self.handler.logger.print_key_value_pair("#Words, #Terms and TTR for original text is", "{0}, {1}, {2:5.2f} %".format(lex.words,lex.terms,100*float(lex.ttr)))
        except Exception as e:
            self.handler.logger.print_warning(f"Could not calculate lexical richness: {e}")

        # Mask all material by encryption (optional)
        if (self.handler.config.should_hex_encode()):
            dataset = Helpers.do_hex_base64_encode_on_data(dataset)

        # Text must be turned into numerical feature vectors ("bag-of-words"-technique).
        # If selected, remove stop words
        my_stop_words = None
        if self.handler.config.use_stop_words():

            # Get the languange specific stop words and encrypt them if necessary
            my_stop_words = get_stop_words(my_language)
            self.handler.logger.print_key_value_pair("Using standard stop words", my_stop_words)
            if (self.handler.config.should_hex_encode()):
                for word in my_stop_words:
                    word = Helpers.cipher_encode_string(str(word))

            # Collect text specific stop words (already encrypted if encryption is on)
            text_specific_stop_words = []
            threshold = 1.0
            if threshold < 1.0:
                try:
                    stop_vectorizer = CountVectorizer(min_df = threshold)
                    stop_vectorizer.fit_transform(dataset)
                    text_specific_stop_words = stop_vectorizer.get_feature_names()
                    self.handler.logger.print_key_value_pair("Using specific stop words", text_specific_stop_words)
                except ValueError as e:
                    self.handler.logger.print_warning(f"Specified stop words threshold at {threshold} generated no stop words.")
                
                my_stop_words = sorted(set(my_stop_words + text_specific_stop_words))
                self.handler.logger.print_key_value_pair("Total list of stop words", my_stop_words)

        # Use the stop words and count the words in the matrix        
        count_vectorizer = CountVectorizer(stop_words = my_stop_words)
        count_vectorizer.fit(dataset)

        return count_vectorizer
        
    # Feature selection (reduction) function.
    # NB: This used to take number_of_components, from mode.num_selected_features
    def perform_feature_selection(self, model: Model) -> None:
        
        # Early return if we shouldn't use feature election, or the selection is RFE (done elsewhere)
        if self.handler.config.feature_selection_in([Reduction.NOR, Reduction.RFE]):
            return

        # Use the saved transform associated with trained model
        if model.transform is not None:
            self.X = model.transform.transform(self.X)

        # In the case of missing training, we issue a warning
        else:
            self.handler.logger.print_warning(f"You are trying to apply a non-existent {model.transform.full_name} transformation")
        
        return

    def compute_feature_selection(self, reduction: Reduction):
        
        # Early return if we shouldn't use feature election, or the selection is RFE (done elsewhere)
        if reduction in [Reduction.NOR, Reduction.RFE]:
            return None
 
        self.X, feature_selection_transform = reduction.call_transformation_theory(logger=self.handler.logger, X=self.X, num_selected_features=None) 
        
        
        return feature_selection_transform
        
    # Split dataset into training and validation parts
    def split_dataset_for_training_and_validation(self) -> bool:
        
        self.handler.logger.print_progress(message="Split dataset for machine learning")
        
        # Quick return if no training
        if not self.handler.config.should_train():
            return False
        else:
        
            # Split data validation dataset from the upper part
            self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split( 
                self.X, self.Y, test_size = self.handler.config.get_test_size(), shuffle = True, random_state = 42, 
                stratify = self.Y)

            return True
        
    # Find out if a DataFrame column contains categorical data or not
    def is_categorical_data(self, column: pd.Series) -> bool:
        if not (self.handler.config.should_train() and self.handler.config.use_categorization()):
            return False
        
        return column.value_counts().count() <= self.LIMIT_IS_CATEGORICAL or self.handler.config.is_categorical(column.name)
        
    # Calculate the number of unclassified rows in data matrix
    def get_num_unpredicted_rows(self, dataset: pd.DataFrame = None) -> int:
        if dataset is None:
            dataset = self.dataset
        num = 0
        for item in dataset[self.handler.config.get_class_column_name()]:
            if item == None or not str(item).strip(): # Corresponds to empty string and SQL NULL
                num += 1
        return num

@dataclass
class Model:

    text_converter: TextDataToNumbersConverter = field(default=None)
    oversampler: Oversampling = field(default=None)
    undersampler: Undersampling = field(default=None)
    preprocess: Preprocess = field(default=None)
    reduction: Reduction = field(default=None)
    algorithm: Algorithm = field(default=None)
    pipeline: Pipeline = field(default=None)
    n_features_out: int = field(default=None)

    def update_fields(self, fields: list[str], update_function: Callable) -> bool:
        """ Updates fields, getting the values from a Callable (ex dh.convert_textdata_to_numbers) """
        values = update_function(self)

        try:
            for i, field in enumerate(fields):
                self.update_field(field, values[i])
        except IndexError as e:
            print(f"Update function returns too few values. {len(fields)} expected, recieved {len(values)}")


    def update_field(self, field: str, value) -> None:
        if hasattr(self, field):
            setattr(self, field, value)

    def get_name(self) -> str:
        if self.pipeline is None:
            return "Empty model"

        name = ""
        for step in self.pipeline.steps:
            name = name  + step[0] + '-'
        name = name[:-1]

        return name

    def get_num_selected_features(self, X) -> int:
        if self.n_features_out is None:
            return X.shape[1]

        return self.n_features_out

    @property
    def class_labels(self) -> list:
        return self.pipeline.classes_
        

@dataclass
class ModelHandler:
    handler: JBGHandler
    model: Model = field(init=False)
    use_feature_selection: bool = field(init=False)
    text_data: bool = field(init=False)
    SPOT_CHECK_REPETITIONS: int = 5
    STANDARD_K_FOLDS: int = 10
    STANDARD_NUM_SAMPLES_PER_FOLD_FOR_OVERSAMPLER: int = 2
    
    def __post_init__(self) -> None:
        """ Empty for now """
    
    def get_class_labels(self, Y: pd.Series) -> list:
        """ Wrapper function go get class labels from the Model, with backup """
        try:
            class_labels = self.model.class_labels
        except AttributeError as e:
            self.handler.logger.print_key_value_pair("No classes_ attribute in Model.Pipeline, using original classes as fallback", e)
            class_labels = [y for y in set(Y) if y is not None]

        return class_labels

    def load_model(self, filename: str = None) -> None:
        """ Called explicitly in run() """
        if filename is None:
            self.model = self.load_empty_model()
        else:
            self.model = self.load_model_from_file(filename)

    # Load ml model
    def load_model_from_file(self, filename: Path, dh: DatasetHandler = None) -> Model:

        # Load model from file, but handle Keras models differently since their algorithm was
        # replaced by a path to model training information
        try:
            config, text_converter, (oversampler, undersampler, preprocess, reduction, algorithm), pipeline, \
                keras_name, n_features = dill.load(open(filename, 'rb'))
                        
            # Handle Keras models differently
            if (keras_name is not None):
                
                if dh is None:
                    raise ModelInitializationException("Could not initialize Keras model: no training or prediction data available!")
                keras_model = keras.models.load_model(str(filename) + "." + keras_name)
                
                try:
                    # TODO: change explicit MLPKerasClassifier constructor to dynamic KERAS model constructor
                    # Note: we need to apply the preceeding transforms to data before initializing the Keras model
                    try:
                        keras_step = MLPKerasClassifier(keras_model).initialize(pipeline.transform(dh.X_train), dh.Y_train)
                    except Exception as e:
                        keras_step = MLPKerasClassifier(keras_model).initialize(pipeline.transform(dh.X_prediction), dh.Y_prediction)
                except Exception as e:
                    raise ModelInitializationException(str(e))
    
                # Load the rest of steps of Pipeline and add keras_model
                pipeline = Pipeline(steps=pipeline.steps + [(keras_name, keras_step)])
            
            the_model = Model(
                text_converter=text_converter,
                oversampler=oversampler,
                undersampler=undersampler,
                preprocess=preprocess,
                reduction=reduction,
                algorithm=algorithm,
                pipeline=pipeline,
                n_features_out=n_features
            )
            
            if self.handler.config.should_predict():
                
                self.handler.config.set_num_selected_features(n_features)
            
        except Exception as e:
            self.handler.logger.print_warning(f"Something went wrong on loading model from file: {e}")
            return None

        return the_model

    # load pipeline
    def load_pipeline_from_file(self, filename: str, init_dh: DatasetHandler = None) -> Pipeline:
        self.handler.logger.print_code("Loading pipeline from", filename)
        model = self.load_model_from_file(filename, init_dh)
        if model:
            return model.pipeline
        else:
            return None
    
    # Sets default (read: empty) values
    def load_empty_model(self) -> Model:
        return Model()

    def train_model(self, dh: DatasetHandler, cross_validation_filepath: str) -> None:
        try:
            self.model = self.get_model_from(dh, cross_validation_filepath)
        except ModelException as ex:
            raise ModelException(f"Model could not be trained. Check your training data or your choice of models: {str(ex)}")
        except Exception as e:
            self.handler.logger.print_dragon(exception=e)
            raise ModelException(f"Something unknown went wrong on training model: {str(e)}")

    def get_model_from(self, dh: DatasetHandler, cross_validation_filepath: str) -> Model:
        
        # Prepare for k-folded cross evaluation
        # Calculate the number of possible folds of the training data. The k-value is chosen such that
        # each fold contains at least one sample of the smallest class
        k_train = int(Helpers.find_smallest_class_number(dh.Y) * 1.0-self.handler.config.get_test_size())
        k = min(self.STANDARD_K_FOLDS, k_train)
        
        # If oversampler is turned on, we need to have at least two samples of the smallest class in each fold
        if self.handler.config.use_oversampling():
            k2 = int(float(k_train) / self.STANDARD_NUM_SAMPLES_PER_FOLD_FOR_OVERSAMPLER)
            if k2 < 2:
                self.handler.logger.print_warning(f"The smallest class is of size {k}, which is too small for using oversampler in cross-validation. Turning oversampler off!") 
                self.handler.config.set_oversampler(Oversampling.NOG)
            else:
                k = min(k, k2)
        if k < self.STANDARD_K_FOLDS:
            self.handler.logger.print_key_value_pair("Using non-standard k-value for cross-validation of algorithms", k)
        
        try:
            # Find the best model
            model = self.spot_check_machine_learning_models(dh, cross_validation_filepath, k=k)
            if model is None:
                raise ModelException(f"No model could be trained with the given settings: {str(ex)}")
            
            # Now train the picked model on training data, either with a grid search or ordinary fit.
            # We assume the algorithm is the last step in the pipeline.
            model_name = model.get_name()
            alg_name = model_name.split('-')[-1]
            t0 = time.time()
            if not model.algorithm.search_params.parameters:
                self.handler.logger.print_progress(f"Using ordinary fit for final training of model {model_name}...(consider adding grid search parameters)")
                model.pipeline = self.train_picked_model(model.pipeline, dh.X_train, dh.Y_train)
            else:
                self.handler.logger.print_progress(f"Using grid search for final training of model {model_name}...")

                # Doing a grid search, we must pass on search parameters to algorithm with '__' notation. 
                #search_params = \
                #    Helpers.add_missing_prefix_double_underscore_to_dict_keys(prefix=alg_name, the_dict=model.algorithm.search_params.parameters)
                search_params = \
                    Helpers.add_prefix_to_dict_keys(prefix=alg_name, the_dict=model.algorithm.search_params.parameters)
                self.handler.logger.print_info(f"Using grid search parameters: {str(search_params)} for model {model_name}")
                try:
                    # grid_cv_info is no used right now, but stays for debugging purposes
                    model.pipeline, grid_cv_info = self.model_parameter_grid_search(
                        model.pipeline,
                        search_params,
                        k,
                        dh.X_train,
                        dh.Y_train)
                    self.handler.logger.print_code("Optimized parameters after grid search", model.pipeline.get_params(deep=False))
                except ModelException as ex:
                    # In case all estimators failed to fit, use ordinary fit as fallback
                    self.handler.logger.print_info(f"Grid search failed: {str(ex)}. Using ordinary fit as fallback for: {model_name}. " + \
                        f"You might need to adjust grid search parameters for pipeline: {str(model.pipeline)}")
                    model.pipeline = self.train_picked_model(model.pipeline, dh.X_train, dh.Y_train)
            t1 = time.time()
            self.handler.logger.print_info(f"Final training of model {alg_name} took {str(round(t1-t0,2))} secs.")
        except Exception as ex:
            raise ModelException(f"Model from spot_check_machine_learning_models failed: {str(ex)}")

        return model

    # Train ml model
    def train_picked_model(self, model: Pipeline, X: pd.DataFrame, Y: pd.DataFrame) -> Pipeline:
        
        # Train model
        try:
            return model.fit(X, Y)
        except TypeError:
            return model.fit(X.to_numpy(), Y.to_numpy())
        except Exception as e:
            self.handler.logger.print_dragon(exception=e)
            raise ModelException(f"Something went wrong on training picked model: {str(e)}")
        
    
    # For pipelines with a specified fit search parameters list, do like this instead
    def model_parameter_grid_search(self, model: Pipeline, search_params: dict, n_splits: int, X: pd.DataFrame, Y: pd.DataFrame) -> Pipeline:

        try:
            # Make a stratified kfold
            kfold = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=True)

            # Create correct scoring mechanism
            scorer = self.handler.config.get_scoring_mechanism()

            # Create search grid and fit model
            search_verbosity = 4 if self.handler.config.io.verbose else 0
            search = self.execute_n_job(GridSearchCV, model, search_params, scoring=scorer, cv=kfold, refit=True, verbose=search_verbosity)
            try:
                search.fit(X, Y)
            except TypeError:
                search.fit(X.to_numpy(), Y.to_numpy())
            except Exception as e:
                self.handler.logger.print_dragon(exception=e)
                raise ModelException(f"Something went wrong on grid search training of picked model: {str(e)}")                

            # Choose best estimator from grid search
            return search.best_estimator_, pd.DataFrame.from_dict(search.cv_results_)
        
        except Exception as e:
            self.handler.logger.print_dragon(exception=e)
            raise ModelException(f"Something went wrong on training picked model with grid parameter search: {str(e)}")

    # Train and evaluate picked model (warning for overfitting)
    def train_and_evaluate_picked_model(self, pipeline: Pipeline, dh: DatasetHandler):

        exception = ""
        test_score = -1.0
        try:
            # First train model on whole of test data (no k-folded cross validation here).
            # Handle problems with sparse input by conversion to numpy, if needed
            try:
                pipeline.fit(dh.X_train, dh.Y_train)
            except TypeError:
                pipeline.fit(dh.X_train.to_numpy(), dh.Y_train.to_numpy())

            # Evaluate on test_data with correct scorer
            if dh.X_validation is not None and dh.Y_validation is not None:
                scorer = self.handler.config.get_scoring_mechanism()
                
                if not isinstance(scorer, str):
                    try:
                        test_score = scorer(pipeline, dh.X_validation, dh.Y_validation)
                    except TypeError:
                        test_score = scorer(pipeline, dh.X_validation.to_numpy(), dh.Y_validation.to_numpy())
                        
                elif scorer == 'roc_auc_ovo':
                    test_score = self.generate_roc_auc_score(pipeline, dh)
                else:
                    raise MissingScorerException("Scorer {0} is not supported".format(scorer))
                    
        except Exception as ex:
            test_score = np.nan
            if not GIVE_EXCEPTION_TRACEBACK:
                exception = str("{0}: {1}".format(type(ex).__name__, ','.join(ex.args)))
            else:
                exception = str(traceback.format_exc())

        return pipeline, test_score, exception
    
    # Help function for generating roc_auc_score in the general case
    def generate_roc_auc_score(self, pipeline: Pipeline, dh: DatasetHandler):
        
        # Multilabel case does not raise an ValueError
        try:
            try:
                score = roc_auc_score(dh.Y_validation, pipeline.predict_proba(dh.X_validation), multi_class='ovo')
            except TypeError:
                score = roc_auc_score(dh.Y_validation.to_numpy(), pipeline.predict_proba(dh.X_validation.to_numpy()), multi_class='ovo')
        # Binary case
        except ValueError:
            try:
                score = roc_auc_score(dh.Y_validation, pipeline.predict_proba(dh.X_validation)[:, 1], multi_class='ovo')
            except TypeError:
                score = roc_auc_score(dh.Y_validation.to_numpy(), pipeline.predict_proba(dh.X_validation.to_numpy())[:, 1], multi_class='ovo')
        return score
        

    # Help function for running other functions in parallel
    def execute_n_job(self, func: function, *args: tuple, **kwargs: dict) -> Any:
            n_jobs = psutil.cpu_count()
            success = False
            result = None
            while not success:
                try:
                    result = func(*args, **kwargs, n_jobs=n_jobs)
                except MemoryError as ex:
                    if n_jobs == 1:
                        raise MemoryError(f"n_jobs is 1 but not enough memory for {func.__name__}") from ex
                    new_njobs = max(int(n_jobs / 2), 1)
                    self.handler.logger.print_warning(f"MemoryError in {func.__name__}, scaling down n_jobs from {n_jobs} to {new_njobs}")
                    n_jobs = new_njobs
                except Exception as ex:
                    raise Exception(f"Could not call function {func.__name__} with supplied positional arguments " + \
                                    f"{str(ex)} and keyword arguments {str(kwargs)}") from ex
                else:
                    success = True
            return result
    
    # While more code, this should (hopefully) be easier to read
    def should_run_computation(self, current_reduction: Reduction, current_algorithm: Algorithm) -> bool:
        # Some algorithms and RFE do not get along
        if current_reduction == Reduction.RFE and not current_algorithm.rfe_compatible:
            return False
        else:
            return True

    # Spot Check Algorithms.
    # We do an extensive search of the best algorithm in comparison with the best
    # preprocessing.
    def spot_check_machine_learning_models(self, dh: DatasetHandler,  cross_validation_filepath: str, k: int=10) -> Model:
        
        # Save standard progress text
        standardProgressText = "Check and train algorithms for best model"
        self.handler.logger.print_info("Spot-checking ML algorithms")
        
        # Prepare a list of feature reduction transforms to loop over
        reductions = self.handler.config.get_callable_reductions(*dh.X.shape)
        
        # Prepare list of algorithms to loop over
        algorithms = self.handler.config.get_callable_algorithms(
            size=dh.X.shape[0], 
            max_iterations=self.handler.config.get_max_iterations()
        )

        # Prepare list of preprocessors
        preprocessors = self.handler.config.get_callable_preprocessors()

        # Prepare over- and undersampling methods
        oversampler = self.handler.config.mode.oversampler
        undersampler = self.handler.config.mode.undersampler

        # Prepare some guidance
        progress_key = "training_model"
        number_of_tries = len(algorithms) * len(reductions) * len(preprocessors)
        
        self.handler.logger.start_inline_progress(
            progress_key, 
            "Checking & Training models", 
            number_of_tries, 
            "Percent models checked")
        
        # Evaluate each model in turn in combination with all reduction and preprocessing methods
        best_cv_score = 0.0
        best_stdev = 1.0
        best_test_score = 0.0
        trained_pipeline = None
        best_algorithm = None
        best_preprocessor = None
        best_reduction = None
        best_num_components = dh.X_train.shape[1]
        
        # Store evaluation results in a list of lists
        listOfResults = []
        
        # Make evaluation of model
        try:
            kfold = StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
        except Exception as e:
            self.handler.logger.print_dragon(exception=e)
            raise ModelException(f"StratifiedKfold raised an exception with message: {e}")
        
        # Due to the stochastic nature of the algorithms, make sure we do some repetitions until successful cross validation training
        success = False
        repetitions = 0
        while repetitions < self.SPOT_CHECK_REPETITIONS and not success:
            if repetitions > 0:
                self.handler.logger.reset_inline_progress(progress_key)
                self.handler.logger.print_warning("All algorithms failed. Restarting spot check.")
            
            repetitions += 1
            tries = -1 # Starting on negative number with ensure the first update will give 0 percent progress
        
            # Loop over pre-processing methods
            for preprocessor, preprocessor_callable in preprocessors:
                
                # Loop over feature reduction transforms
                for reduction, reduction_callable in reductions:

                    # For RFE only
                    best_rfe_feature_selection = dh.X.shape[1]
                    first_rfe_round = True if reduction == Reduction.RFE else False

                    # Loop over the algorithms
                    for algorithm, algorithm_callable in algorithms:
                        
                        # Keep track so we can update the progress bar correctly
                        tries += 1        
                        
                        # Some combinations of REF and algorithms are error prone and should be skipped
                        if not self.should_run_computation(reduction, algorithm):
                            continue

                        # Divide data in training and test parts according to settings X -> X_train, X_validation etc...
                        dh.split_dataset_for_training_and_validation()
                        
                        # Update progressbar percent and label
                        self.handler.logger.print_progress(message=f"{standardProgressText} ({preprocessor.name}-{reduction.name}-{algorithm.name})")
                        if not first_rfe_round:
                            self.handler.logger.update_inline_progress(progress_key, tries, "Percent models checked")
                        else:
                            first_rfe_round = False

                        # Add RFE feature selection if selected, i.e., the option of reducing the number of variables recursively.
                        # With RFE, we make a binary search for the optimal number of features.
                        max_features_selection = dh.X.shape[1]
                        min_features_selection = 0 if reduction == Reduction.RFE else max_features_selection
                        num_components = max_features_selection
                        
                        # Loop over feature selections span: break this loop when min and max reach the same value
                        rfe_score = 0.0                                             # Save the best values so far.
                        num_features = max_features_selection                       # Start with all features.
                        first_feature_selection = True                              # Make special first round: use all features
                        
                        while first_feature_selection or min_features_selection < max_features_selection:
                            
                            # Update limits for binary search, and end loop if needed
                            if not first_feature_selection:
                                num_features = ceil((min_features_selection+max_features_selection) / 2)
                                if num_features == max_features_selection:          
                                    break
                            else:
                                first_feature_selection = False
                                num_features = max_features_selection

                            # Calculate the time for this setting
                            t0 = time.time()
                            
                            try:
                                # Create pipeline and cross validate
                                current_pipeline, cv_results, failure = \
                                    self.create_pipeline_and_cv(reduction, algorithm, preprocessor, reduction_callable, \
                                        algorithm_callable, preprocessor_callable, oversampler, undersampler, \
                                            kfold, dh, num_features)
                                
                                # Train and evaluate on test data
                                if dh.X_validation is not None and dh.Y_validation is not None:
                                    current_pipeline, tmp_test_score, failure = \
                                        self.train_and_evaluate_picked_model(current_pipeline, dh)
                                else:
                                    tmp_test_score = 0.0

                                # Get used number of features after reduction (components)
                                num_components = self.get_components_from_pipeline(reduction, current_pipeline, num_features)
                            except ModelException:
                                # If any exceptions happen, continue to next step in the loop
                                self.handler.logger.print_warning(f"ModelException: {str(ex)}")
                                break
                            except Exception as ex:
                                self.handler.logger.print_warning(f"Exception: {str(ex)}")

                            # Stop the stopwatch
                            t = time.time() - t0

                            # For current settings, calculate score
                            temp_cv_score = cv_results.mean()
                            temp_cv_stdev = cv_results.std()

                            # Evaluate if feature selection changed accuracy or not. 
                            # Notice: Better or same score with less variables are both seen as an improvement,
                            # since the chance of finding an improvement increases when number of variables decrease
                            if  temp_cv_score >= rfe_score:
                                rfe_score = temp_cv_score
                                max_features_selection = num_features   # We need to reduce more features
                            else:
                                min_features_selection = num_features   # We have reduced too much already  

                            # Save result if it is the overall best (inside RFE-while)
                            # Notice the difference from above, here we demand a better score.
                            try:
                                if self.is_best_run_yet(temp_cv_score, temp_cv_stdev, best_cv_score, best_stdev, \
                                                        tmp_test_score, best_test_score):
                                    trained_pipeline = current_pipeline
                                    best_reduction = reduction
                                    best_algorithm = algorithm
                                    best_preprocessor = preprocessor
                                    best_cv_score = temp_cv_score
                                    best_stdev = temp_cv_stdev
                                    best_test_score = tmp_test_score
                                    best_rfe_feature_selection = num_features
                                    best_num_components = num_components
                            except UnstableModelException as ex:
                                if not failure:
                                    failure = f"{','.join(ex.args)}"
                            else:
                                success = True

                            listOfResults.append([ 
                                preprocessor.name,
                                reduction.name,
                                str(algorithm.name) + " - " + str(algorithm.full_name) + " (" + str(algorithm.lib.full_name) + ")",
                                min(num_components, num_features),
                                temp_cv_score,
                                temp_cv_stdev,
                                tmp_test_score,
                                t,
                                failure])
                    
        updates = {"feature_selection": reduction, "algorithm": best_algorithm, \
            "preprocessor" : best_preprocessor, "num_selected_features": best_rfe_feature_selection}
        self.handler.config.update_attributes(type="mode", updates=updates)
        
        best_model = self.model
        best_model.preprocess = best_preprocessor
        best_model.reduction = best_reduction
        best_model.algorithm = best_algorithm
        best_model.pipeline = trained_pipeline
        best_model.n_features_out = best_num_components
        
        # Prepare and print a pandas Dataframe for storing test evaluation results
        self.handler.logger.clear_last_printed_result_line()
        self.handler.logger.print_test_performance(listOfResults, cross_validation_filepath)
        
        self.handler.logger.end_inline_progress(progress_key)
        # Return best model for start making predictions
        return best_model

    def calculate_current_features(self, current_score: float, best_score: float, num_features: int, max_features: int, min_features: int):
        # Evaluate if feature selection changed accuracy or not. 
        # Notice: Better or same score with less variables are both seen as an improvement,
        # since the chance of finding an improvement increases when number of variables decrease
        if  current_score >= best_score: # Reduce more features
            return current_score, num_features, min_features
        
        # Too much reduced
        return best_score, max_features, num_features

    def is_best_run_yet(self, train_score: float, train_stdev: float, best_train_score: float, best_stdev: float, \
                        test_score: float, best_test_score: float) -> bool:
        """ Calculates if this round is better than any prior """
        
        # First check if performance seems unstable
        if abs(train_score - test_score) > 2.0 * train_stdev:
            raise UnstableModelException(f"Performance metric difference for cross evaluation and final test exceeds 2*stdev")
        
        # If performance is stable, check if it is better than last time.
        # Use these rules to detemine:
        # 1. If it is better on test and at least comparable for train data, it is better
        if test_score > best_test_score and train_score >= best_train_score: 
            return True
        # 2. If at least comparable on test and train data and with lower standard deviation, it is better
        elif test_score >= best_test_score and train_score >= best_train_score and train_stdev < best_stdev:
            return True
        # 3. If strictly better on test and the difference between train and test is less than or equal to 1*stdev
        elif test_score > best_test_score and abs(train_score - test_score) <= 1*train_stdev:
            return True
        else:
            return False

    # Build pipeline and perform cross validation
    def create_pipeline_and_cv(self, reduction: Reduction, algorithm: Algorithm, preprocessor: Preprocess, feature_reducer: Transform, \
        estimator: Estimator, scaler: Transform, oversampler: Oversampling, undersampler: Undersampling, kfold: StratifiedKFold, \
        dh: DatasetHandler, num_features: int):

        exception = ""
        try:
                    
            # Build pipeline of model and preprocessor.
            pipe = self.get_pipeline(reduction, feature_reducer, algorithm, estimator, preprocessor, scaler, \
                oversampler, undersampler, dh.X_train.shape[1], num_features, min(5, kfold.get_n_splits()))
                        
            # Use parallel processing for k-folded cross evaluation
            cv_results = self.get_cross_val_score(pipeline=pipe, dh=dh, kfold=kfold, algorithm=algorithm)

        except Exception as ex:
            cv_results = np.array([np.nan])
            if not GIVE_EXCEPTION_TRACEBACK:
                exception = str("{0}: {1}".format(type(ex).__name__, ','.join(ex.args)))
            else:
                exception = str(traceback.format_exc())

        return pipe, cv_results, exception
    
    # Build the pipeline
    def get_pipeline(self, reduction: Reduction, feature_reducer: Transform, algorithm: Algorithm, \
                estimator: Estimator, preprocessor: Preprocess, scaler: Transform, \
                oversampler: Oversampling, undersampler: Undersampling, max_features: int, \
                rfe_features: int = None, max_k_neighbors: int = 2) -> Union[Pipeline, Estimator]:
       
        try:
            # First, put estimator/algorithm in pipeline
            steps = [(algorithm.name, estimator)]
            
            # Secondly, add the other steps to the pipeline BEFORE the algorithm
            steps.insert(0, (preprocessor.name, scaler))
            
            # RFE object must unfortunately be updated with the correct estimator at this point
            if reduction == Reduction.RFE:
                the_feature_reducer = RFE(estimator=estimator, n_features_to_select=rfe_features)
            else:
                the_feature_reducer = feature_reducer
            
            # Then add the feature reduction part of the pipeline between the preprocessor and the estimator
            steps.insert(1, (reduction.name, the_feature_reducer))

            # Finally, put oversampling and undersampling techniques before everything else
            steps.insert(0, (oversampler.name, oversampler.get_callable_oversampler()))
            steps.insert(1, (undersampler.name, undersampler.get_callable_undersampler()))
            
            # Check that all steps in pipeline implements "fit"
            steps = [step for step in steps if hasattr(step[1] , "fit") and callable(getattr(step[1] , "fit"))] 
        
        except Exception as ex:
            raise PipelineException(f"Could not build Pipeline correctly: {str(ex)}") from ex
        
        return Pipeline(steps=steps, verbose=False)

    # Find number of components in reduction step in Pipeline
    def get_components_from_pipeline(self, reduction, pipeline, num_features) -> int:

        for step in pipeline.steps:
            try:
                if reduction == Reduction[step[0]]:
                    try:
                        if isinstance(step[1].n_components_, int):
                            return step[1].n_components_
                    except AttributeError as ex:
                        if isinstance(step[1].n_components, int):
                            return step[1].n_components
                    except Exception as ex:
                        print(f"Matched reduction {str(step)} has no component attribute of type integer: {str(ex)}")
                        return num_features
            except Exception as ex:
                pass

        return num_features
    
    # Make necessary modifications to algorithm when rfe is on
    def rfe_modify_algorithm_step(self, estimator: Estimator, estimator_name: str, n_features: int, n_features_to_select: int) -> list[tuple]:
        
        # Apply RFE feature selection to current model and number of features.
        if n_features > n_features_to_select:
            rfe = RFE(estimator, n_features_to_select=n_features_to_select)
            rfe_step = ('rfe', rfe)
        else:
            rfe_step = ('rfe', None)
        steps = [rfe_step, (estimator_name, estimator)]
            
        return steps

    # Make cross val score evaluation
    def get_cross_val_score(self, pipeline: Pipeline, dh: DatasetHandler, kfold: StratifiedKFold, algorithm: Algorithm) -> np.ndarray:
       
        # Now make kfolded cross evaluation. Notice that fit_params are not used right now (just placeholder for future revisions)
        scorer_mechanism = self.handler.config.get_scoring_mechanism()

        fit_params = {}
        
        for key, function_name in algorithm.fit_params.items():
            if hasattr(self, function_name) and callable(func := getattr(self, function_name)):
                fit_params[key] = func(dh.X_train, dh.Y_train)

        # We want to executed the job with as many threads as possible, but as a final alternative use
        # only one. Exception typically arises when input data is sparse, and a possible remedy to convert 
        # it to dense numpy arrays.
        try:
            cv_results = self.execute_n_job(cross_val_score, pipeline, dh.X_train, dh.Y_train, cv=kfold, \
                scoring=scorer_mechanism, fit_params=fit_params, error_score='raise') 
        except Exception as ex:
            #print("cross_val_score Exception 1:", str(ex))
            try:
                cv_results = self.execute_n_job(cross_val_score, pipeline, dh.X_train.to_numpy(), \
                    dh.Y_train.to_numpy(), cv=kfold, scoring=scorer_mechanism, fit_params=fit_params, \
                    error_score='raise') 
            except Exception as ex:
                #print("cross_val_score Exception 2:",str(ex))
                try:
                    cv_results = cross_val_score(pipeline, dh.X_train.to_numpy(), dh.Y_train.to_numpy(), \
                        cv=kfold, scoring=scorer_mechanism, fit_params=fit_params, error_score='raise')
                except Exception as ex:
                    #print("cross_val_score Exception 3:",str(ex))
                    raise ModelException(f"Unexpected error in cross_val_score: {str(ex)}") from ex
        
        return cv_results
    
    def save_model_to_file(self, filename: Path):
        """ Save ml model and corresponding configuration """

        save_config = self.handler.config.get_clean_config()

        # Handle KERAS models differently because of problems with Access denied errors
        try:
            if self.model.algorithm.lib == Library.KERAS:
                
                # Prepare the data to save
                keras_name = self.model.pipeline.steps[-1][0]
                data = [
                    save_config,
                    self.model.text_converter,
                    (self.model.oversampler, self.model.undersampler, self.model.preprocess, self.model.reduction, self.model.algorithm),
                    Pipeline(steps=self.model.pipeline.steps[:-1]),
                    keras_name,
                    self.model.n_features_out
                ]   
                
                # Save the data minus the KERAS model
                dill.dump(data, open(filename,'wb'))

                # Save the KERAS model separately
                self.model.pipeline.steps[-1][-1].model_.save(str(filename) + "." + keras_name)

            else: # Non KERAS models
                
                # Prepare the data to save
                data = [
                    save_config,
                    self.model.text_converter,
                    (self.model.oversampler, self.model.undersampler, self.model.preprocess, self.model.reduction, self.model.algorithm),
                    self.model.pipeline,
                    None,                       # Placeholder for Keras model name (unused)
                    self.model.n_features_out
                ]   
                
                # Save the data
                dill.dump(data, open(filename,'wb'))

        except Exception as e:
            self.handler.logger.print_warning(f"Something went wrong on saving {self.model.algorithm.lib.get_full_name()} model to file: {e}")

@dataclass
class PredictionsHandler:
    LIMIT_MISPREDICTED = 50
    CROSS_TRAINED_MODEL = "cross trained"
    RETRAINED_MODEL = "retrained"
    NO_MODEL = "no model"
    
    handler: JBGHandler
    could_predict_proba: bool = False
    probabilites: np.ndarray = field(init=False)
    predictions: np.ndarray = field(init=False)
    rates: np.ndarray = field(init=False)
    num_mispredicted: int = field(init=False)
    X_mispredicted: pd.DataFrame = field(init=False)
    X_most_mispredicted: pd.DataFrame = field(init=False)
    dark_numbers: pd.DataFrame = field(init=False)
    dark_numb_conf_matrix: pd.DataFrame = field(init=False)
    model: str = field(init=False)
    class_report: dict = field(init=False)

    def get_prediction_results(self, keys: pd.Series) -> list:
        """ 
            Creates a list combining the values from the set prediction,
            using a given key. It is used to simplify saving the data
        """
        return_list = []
        
        try:
            for k,y,r,p in zip(keys.values, self.predictions, self.rates, self.probabilites):
                item = {
                    "key": int(k),
                    "prediction": y,
                    "rate": float(r),
                    "probabilities": self.get_probablities_as_string(p)
                }

                return_list.append(item)
        except AttributeError:
            return []
        
        return return_list

    def save_predictions(self, dh: DatasetHandler, mh: ModelHandler) -> dict:
        """ Saves the predictions in the database, to be fetched later """
        try:
            results = self.get_prediction_results(dh.unpredicted_keys)
        except AttributeError as e: 
            raise HandlerException(e)

        class_labels = mh.get_class_labels(Y=dh.Y)

        try:
            return self.handler.datalayer.save_prediction_data(
                results,
                class_rate_type=self.get_rate_type(),
                model_name=mh.model.get_name(),
                class_labels=class_labels
            )
            
        except Exception as e:
            self.handler.logger.print_dragon(exception=e)
            raise HandlerException(e)

    def get_probablities_as_string(self, item) -> str:
        """ Gets a probabilities list as a comma-delimited string """
        try:
            iter(item)
            return ",".join([str(elem) for elem in item])
        except TypeError:
            # This only happens if the model couldn't predict, so uses the mean
            return item

    def get_mispredicted_dataframe(self) -> pd.DataFrame:
        try:
            return self.X_most_mispredicted
        except AttributeError:
            return None

    def report_results(self, Y) -> None:
        """ Prints the various informations """
        rates = None
        
        if self.could_predict_proba:
            rates = self.get_rates(as_string = False)
        
        # Evaluate predictions (optional)
        accuracy = accuracy_score(Y, self.predictions)
        con_matrix = confusion_matrix(Y, self.predictions)
        class_labels = sorted(set(Y.tolist() + self.predictions.tolist()))
        class_matrix = classification_report(Y, self.predictions, zero_division='warn', output_dict=True)
        self.handler.logger.print_prediction_report(
            accuracy_score=accuracy,
            confusion_matrix=con_matrix,
            class_labels= class_labels,
            classification_matrix=class_matrix,
            sample_rates= rates)


    # Evaluates mispredictions
    def evaluate_mispredictions(self, misplaced_filepath: str) -> None:
        try:
            if self.X_most_mispredicted.empty or not self.handler.config.should_display_mispredicted():
                return 
        except AttributeError: # In some cases X_most_mispredicted is not even defined
            return
        
        self.handler.logger.print_key_value_pair(f"Total number of mispredicted elements", self.num_mispredicted, print_always=True)
        
        ids = ', '.join([str(id) for id in self.X_most_mispredicted.index.tolist()])
        most_mispredicted_query = f"{self._get_rid_of_TOP_stuffs(self.handler.read_data_query)} WHERE {self.handler.config.get_id_column_name()} IN ({ids})"
        
        self.handler.logger.print_code("Get the most misplaced data by SQL query", most_mispredicted_query)
        Helpers.save_matrix_as_csv(
            self.X_mispredicted,
            misplaced_filepath,
            self.handler.config.get_id_column_name()
        )
        self.handler.logger.print_code("Open the following csv-data file to get the full list", Helpers.create_download_link(misplaced_filepath, title = ""))
    
    # Clean up data query string from the original TOP specification
    def _get_rid_of_TOP_stuffs(self, query):

        top = "TOP("
        parentesis = ")"

        start = query.find(top)
        stop = query.find(parentesis)

        query = query[:start-1] + query[stop+1:]

        return query

    # Compte dark number of predictions for all classes
    #def calculate_dark_numbers(self, Y_real: pd.Series, Y_predicted: pd.Series, Y_pred_proba: pd.Series) -> pd.DataFrame:
    #
    #    confusion_matrix, dark_numbers = compute_dark_numbers(Y_real, Y_predicted, Y_pred_proba, type="all")
    #    return confusion_matrix, dark_numbers
    
    # Make predictions on dataset
    def make_predictions(self, model: Pipeline, X: pd.DataFrame, classes: pd.Series, Y: pd.DataFrame = None) -> bool:
        could_predict_proba = False
        try:
            predictions = model.predict(X)
        except TypeError:
            predictions = model.predict(X.to_numpy())
        except ValueError as e:
            self.handler.logger.abort_cleanly(message=f"It seems like you need to regenerate your prediction model: {e}")
        try:
            try:
                probabilities = model.predict_proba(X)
            except TypeError:
                probabilities = model.predict_proba(X.to_numpy())
            rates = np.amax(probabilities, axis=1)
            could_predict_proba = True
        except Exception as e:
            self.handler.logger.print_warning(f"Probablity prediction not available for current model: {e}")
            probabilities = np.array([[-1.0]*len(classes)]*X.shape[0])
            rates = np.array([-1.0]*X.shape[0])
        
        self.could_predict_proba = could_predict_proba
        self.probabilites = probabilities
        self.predictions = predictions
        self.rates = rates

        if Y is not None:
            self.set_classification_report(Y)
        # This is backup if predictions couldn't be made (IE predict_proba == False)
        self.calculate_probability()
        
        return could_predict_proba

    # Gets the rate type
    def get_rate_type(self) -> RateType:
        if self.could_predict_proba:
            return RateType.I

        if not self.handler.config.should_train():
            return RateType.U
        
        return RateType.A
    
    # Calculate the probability if the machine could not
    def calculate_probability(self) -> None:
        if not self.handler.config.should_train():
            return

        if self.could_predict_proba:
            return 
        
        prob = []
        
        for prediction in self.predictions:
            try:
                # This should probably be a list of floats, not a float
                prob = prob + [self.class_report[prediction]['precision']]
            except KeyError as e:
                self.handler.logger.print_warning(f"probability collection failed for key {prediction} with error {e}")
    
        self.probabilites = prob
        
    
    # Returns a list of mean and standard deviation
    def get_rates(self, as_string: bool = False) -> list:
        mean = np.mean(self.rates)
        std = np.std(self.rates)
        
        if as_string:
            return [str(mean), str(std)]

        return [mean, std]

    def set_classification_report(self, Y: pd.DataFrame) -> dict:
        report = classification_report(Y, self.predictions, output_dict = True)
        self.class_report = report

    
    # Function for finding the n most mispredicted data rows
    # TODO: Clean up a bit more
    def most_mispredicted(self, X_original: pd.DataFrame, full_pipe: Pipeline, ct_pipe: Pipeline, X: pd.DataFrame, Y: pd.Series) -> None:
        
        # Calculate predictions for both total model and cross trained model
        self.num_mispredicted = 0
        self.model = self.NO_MODEL
        
        # For storing mismatching informations
        X_not = [] 
        model_not = pd.Series(index = Y.index)
        Y_not = pd.Series(index = Y.index)  

        # Make predictions using both models
        try:
            Y_pred_ct, ct_model = pd.Series(ct_pipe.predict(X), index = Y.index), self.CROSS_TRAINED_MODEL
            Y_pred_ft, ft_model = pd.Series(full_pipe.predict(X), index = Y.index), self.RETRAINED_MODEL
        except TypeError:
            Y_pred_ct, ct_model = pd.Series(ct_pipe.predict(X.to_numpy()), index = Y.index), self.CROSS_TRAINED_MODEL 
            Y_pred_ft, ft_model = pd.Series(full_pipe.predict(X.to_numpy()), index = Y.index), self.RETRAINED_MODEL 
        
        for Y_pred_what, what_model in [(Y_pred_ct, ct_model), (Y_pred_ft, ft_model)]:
            self.handler.logger.print_key_value_pair(f"Accuracy score for {what_model} model", accuracy_score(Y, Y_pred_what), print_always=True)

        # Find the data rows where the real category is different from the predictions on at least one of the models.
        # The cross trained model has precedence over the retrained model in case both predictions are incorrect.
        # Iterate over the indexes (they are now not in order).
        for i in Y.index:
            mismatch_ct = Y.loc[i] != Y_pred_ct.loc[i]
            mismatch_ft = Y.loc[i] != Y_pred_ft.loc[i]
            try:
                X_not.append(mismatch_ct or mismatch_ft)
                model_not.loc[i] = ct_model if mismatch_ct else ft_model if mismatch_ft else self.model
                Y_not.loc[i] = Y_pred_ct.loc[i] if mismatch_ct else Y_pred_ft.loc[i] if mismatch_ft else Y.loc[i]
            except Exception as e:
                self.handler.logger.print_warning(f"Append of data row with index: {i} failed: {e}. Indicates duplicate indicies in data!")
                break

        # Update on the number of mispredicted data rows
        model_not = model_not.loc[X_not]
        self.num_mispredicted = sum(elem == True for elem in X_not)
        if self.num_mispredicted > 0:
            self.model = str(model_not.value_counts())
        
        # Quick return if possible
        else:
            self.X_most_mispredicted = pd.DataFrame()
            self.X_mispredicted = pd.DataFrame()
            return
        
        # Select the found mispredicted data from the computations
        self.X_mispredicted = X.loc[X_not]

        # Predict probabilities for predictions in both models and pick the one to display
        try:
            try:
                Y_prob_ct = ct_pipe.predict_proba(self.X_mispredicted)
                Y_prob_ft = full_pipe.predict_proba(self.X_mispredicted)
                ndim = Y_prob_ct.ndim
            except TypeError:
                Y_prob_ct = ct_pipe.predict_proba(self.X_mispredicted.to_numpy())
                Y_prob_ft = full_pipe.predict_proba(self.X_mispredicted.to_numpy())
                ndim = Y_prob_ct.ndim
            Y_prob = []
            
            # Handle the case of only one mispredicted data row
            if ndim == 1:
                Y_prob.append(Y_prob_ct[:] if model_not.iloc[0] == self.CROSS_TRAINED_MODEL else Y_prob_ft[:])
            else:
                # Handle the case of more than one mispredicted data row
                i = 0
                for model in model_not.to_numpy():
                    Y_prob.append(Y_prob_ct[i,:] if model == self.CROSS_TRAINED_MODEL else Y_prob_ft[i,:])
                    i += 1
            Y_prob = np.array(Y_prob)
            could_predict_proba = True
        except Exception as e:
            self.handler.logger.print_key_value_pair("Could not predict probabilities for most mispredicted", e)
            could_predict_proba = False

        #  Re-insert original data columns into mispredicted data DataFrame but drop the class column
        self.X_mispredicted = X_original.loc[X_not]
        
        # Add other columns to mispredicted data
        self.X_mispredicted.insert(0, "Actual", Y.loc[X_not].to_numpy())
        self.X_mispredicted.insert(1, "Predicted", Y_not.loc[X_not].to_numpy())
        self.X_mispredicted.insert(2, "Model", value=model_not.to_numpy())

        # Add probabilities and sort only if they could be calculated above, otherwise
        # return a random sample of mispredicted
        try:
            the_classes = ct_pipe.classes_
        except AttributeError as e:
            self.handler.logger.print_key_value_pair("No classes_ attribute in model, using original classes as fallback", e)
            the_classes = [y for y in set(Y) if y is not None]

        if not could_predict_proba:
            for item in the_classes:
                self.X_mispredicted.insert(0, f"P({item})", "N/A")
            n_limit = min(self.LIMIT_MISPREDICTED, self.X_mispredicted.shape[0])

            self.X_most_mispredicted = self.X_mispredicted.sample(n=n_limit)
            return
        
        Y_prob = Y_prob if len(Y_prob.shape) > 1 else np.array([Y_prob]) # Handle only one mispredicted sample
        Y_prob_max = np.amax(Y_prob, axis = 1)
        for i in reversed(range(Y_prob.shape[1])):
            self.X_mispredicted.insert(0, f"P({the_classes[i]})", Y_prob[:,i])
        self.X_mispredicted.insert(0, "__Sort__", Y_prob_max)

        # Sort the dataframe on the first column and remove it
        self.X_mispredicted = self.X_mispredicted.sort_values("__Sort__", ascending = False)
        self.X_mispredicted = self.X_mispredicted.drop("__Sort__", axis = 1)

        # Keep only the top n_limit rows and return
        self.X_most_mispredicted = self.X_mispredicted.head(self.LIMIT_MISPREDICTED)
        
        return
    
    # Dark number stuffs
    def get_dark_numbers(self, X: pd.DataFrame, Y: pd.DataFrame, type: str = "all", models: list = [None], \
                         model_names: list = [None], combine_models=True) -> None:

        # These DataFrames contain the results
        self.dark_numbers = pd.DataFrame()
        self.dark_numb_conf_matrix = pd.DataFrame()

        # Combine models to compute a worst case scenario
        if combine_models:
            Y_pred_worst = Y.copy(deep=True)
            Y_prob_pred_worst = pd.Series([0.0 for i in range(Y.size)], index=Y.index)

        # Use these as labels
        labels = np.sort(Y.unique())
        
        # Compute dark number for all models
        for model, model_name in zip(models, model_names):

            # Make prediction for current model
            Y_pred = pd.Series(model.predict(X), index=Y.index)

            # Update the confusion matrix
            self._update_confusion_matrix(model_name, Y, Y_pred, labels)
            
            # Predict probabilities for current predictions
            Y_prob_pred = pd.Series([max(row) for row in model.predict_proba(X)], index=Y.index)

            # Compute and update the dark numbers matrix
            self._update_dark_numbers(model_name, Y, Y_pred, Y_prob_pred, type)

            # Combine models processed so far to get worst case scenario by replacing wrong predictions with
            # the currently worst one
            if combine_models:
                Y_pred_worst = Y_pred_worst.mask((Y_pred != Y) & (Y_prob_pred >= Y_prob_pred_worst), Y_pred)
                Y_prob_pred_worst = Y_prob_pred_worst.mask((Y_pred != Y) & (Y_prob_pred >= Y_prob_pred_worst), Y_prob_pred)      
        
        # In case we have combined the models, we need to repeat what we did above
        if combine_models and not self.dark_numbers.empty:
            
            # Update the confusion matrix
            self._update_confusion_matrix("Combined", Y, Y_pred_worst, labels)

            # Compute and update the dark numbers matrix
            self._update_dark_numbers("Combined", Y, Y_pred_worst, Y_prob_pred_worst, type)
    
        return None
    
    def _update_confusion_matrix(self, model_name, Y, Y_pred, labels):

        model_confusion_matrix = pd.DataFrame(confusion_matrix(Y, Y_pred, labels=None), index=labels, columns=labels)

        # Put together the results for the confusion matrix
        model_name_col = pd.Series([model_name]+["" for i in range(model_confusion_matrix.shape[0]-1)], \
                                    name="Model type", index=labels)

        if self.dark_numb_conf_matrix.empty:
            self.dark_numb_conf_matrix = pd.concat([model_name_col, model_confusion_matrix.copy(deep=True)], axis = 1)
        else:
            self.dark_numb_conf_matrix = \
                pd.concat([self.dark_numb_conf_matrix, pd.concat([model_name_col, model_confusion_matrix], axis=1)], axis=0)

        return None

    def _update_dark_numbers(self, model_name, Y, Y_pred, Y_prob_pred, type):
        
        # Compute dark numbers
        model_dark_numbers = compute_dark_numbers(Y, Y_pred, Y_prob_pred, type=type)
        
        # Put together the results for the dark numbers
        model_name_col = pd.Series([model_name]+["" for i in range(model_dark_numbers.shape[0]-1)], name="Model type")
        
        if self.dark_numbers.empty:
            self.dark_numbers = pd.concat([model_name_col, model_dark_numbers.copy(deep=True)], axis = 1)
        else:
            self.dark_numbers = pd.concat([self.dark_numbers, pd.concat([model_name_col, model_dark_numbers], axis=1)], axis=0)

        return None

    def evaluate_dark_numbers(self, dark_numbers_filepath: str, dark_numbers_conf_matr_filepath: str) -> None:
        
        self.handler.logger.display_matrix(f"Dark numbers confusion matrices", self.dark_numb_conf_matrix)
        self.handler.logger.display_matrix(f"Dark numbers calculations", self.dark_numbers, precision = 4)

        Helpers.save_matrix_as_csv(
            self.dark_numb_conf_matrix,
            dark_numbers_conf_matr_filepath
        )

        self.handler.logger.print_code("Get dark numbers confusion matrices", Helpers.create_download_link(dark_numbers_conf_matr_filepath, title = ""))

        Helpers.save_matrix_as_csv(
            self.dark_numbers,
            dark_numbers_filepath
        )

        self.handler.logger.print_code("Get dark numbers calculations", Helpers.create_download_link(dark_numbers_filepath, title = ""))

        return None
    

def main():
    import JBGLogger, SQLDataLayer, Config, sys

    if len(sys.argv) > 1:
        config = Config.Config.load_config_from_module(sys.argv)
    else:
        config = Config.Config()

    logger = JBGLogger.JBGLogger(not config.io.verbose)
    
    datalayer = SQLDataLayer.DataLayer(config=config, logger=logger)
    
    handler = JBGHandler(datalayer, config, logger)
    
if __name__ == "__main__":
    main()
