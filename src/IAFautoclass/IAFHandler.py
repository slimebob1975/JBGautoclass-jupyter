from __future__ import annotations

import pickle
import time
import psutil
import typing
from dataclasses import dataclass, field
from datetime import datetime
from math import ceil
from typing import Callable, Protocol, Union

import langdetect
import numpy as np
import pandas
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lexicalrichness import LexicalRichness
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from stop_words import get_stop_words
from skclean.detectors import KDN


from Config import Algorithm, Preprocess, Reduction, RateType, get_model_name, Estimator, Transform
from IAFExceptions import DatasetException, ModelException, HandlerException
import Helpers

# Sklearn issue a lot of warnings sometimes, we suppress them here
import warnings
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Logger(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""
    def print_info(self, *args) -> None:
        """printing info"""

    def print_progress(self, message: str = None, percent: float = None) -> None:
        """Printing progress"""

    def print_components(self, component, components, exception = None) -> None:
        """ Printing Reduction components"""

    def print_formatted_info(self, message: str) -> None:
        """ Printing info with """

    def print_percentage_checked(self, text: str, old_percent, percent_checked) -> None:
        """ Uses print() to update the line rather than new line"""

    def investigate_dataset(self, dataset: pandas.DataFrame, class_name: str, show_class_distribution: bool = True, show_statistics: bool = True) -> bool:
        """ Print information about dataset """
    
    def print_warning(self, *args) -> None:
        """ print warning """

    def print_table_row(self, items: list[str], divisor: str = None) -> None:
        """ Prints row of items, with optional divisor"""

    def abort_cleanly(self, message: str) -> None:
        """ Exits the process """

class DataLayer(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""

    def get_dataset(self, num_rows: int, train: bool, predict: bool):
        """ Get the dataset, query and number of rows"""

    def save_data(self, results: list, class_rate_type: RateType, model: Model, config: Config)-> int:
        """ Saves classification for X_unknown in classification database """

    def get_sql_command_for_recently_classified_data(self, num_rows: int) -> str:
        """ What name says """

class Config(Protocol):
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

    def use_RFE(self) -> bool:
        """ Gets whether RFE is used or not """

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

    def use_imb_pipeline(self) -> bool:
        """ Returns true if smote or undersampler is not None """

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
        


@dataclass
class IAFHandler:
    datalayer: DataLayer
    config: Config
    logger: Logger
    progression: dict

    handlers: dict = field(default_factory=dict, init=False)
    queries: dict = field(default_factory=dict)
    

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


    def get_dataset(self):
        """ By putting this here, Datalayer does not need to be supplied to Dataset Handler"""
        data, query =  self.datalayer.get_dataset(self.config.get_max_limit(), self.config.should_train(), self.config.should_predict())
        
        self.queries["read_data"] = query

        return data
   
    def save_classification_data(self) -> None:
        # Save new classifications for X_unknown in classification database
        self.logger.print_progress(message="Save new classifications in database")

        try:
            dh = self.get_handler("dataset")
            ph = self.get_handler("predictions")
            mh = self.get_handler("model")
        except HandlerException as e:
            raise e
        
        try:
            results = ph.get_prediction_results(dh.unpredicted_keys)
        except AttributeError as e: 
            raise HandlerException(e)

        try:
            results_saved = self.datalayer.save_data(
                results,
                class_rate_type=ph.get_rate_type(),
                model=mh.model,
                config=self.config
            )
            
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise HandlerException(e)
        
        saved_query = self.datalayer.get_sql_command_for_recently_classified_data(results_saved)
        self.logger.print_info(f"Added {results_saved} rows to classification table. Get them with SQL query:\n\n{saved_query}")

    # Updates the progress and notifies the logger
    # Currently duplicated over Classifier and IAFHandler, but that's for later
    def update_progress(self, percent: float, message: str = None) -> float:
        self.progression["progress"] += percent

        if message is None:
            self.logger.print_progress(percent = self.progression["progress"])
        else:
            self.logger.print_progress(message=message, percent = self.progression["progress"])

        return self.progression["progress"]

    

@dataclass
class DatasetHandler:
    STANDARD_LANG = "sv"
    LIMIT_IS_CATEGORICAL = 30
    
    handler: IAFHandler
    dataset: pandas.DataFrame = field(init=False)
    classes: list[str] = field(init=False)
    keys: pandas.Series = field(init=False)
    unpredicted_keys: pandas.Series = field(init=False)
    X_original: pandas.DataFrame = field(init=False)
    X: pandas.DataFrame = field(init=False)
    Y: pandas.Series = field(init=False)
    X_train: pandas.DataFrame = field(init=False)
    X_validation: pandas.DataFrame = field(init=False)
    Y_train: pandas.DataFrame = field(init=False)
    Y_validation: pandas.DataFrame = field(init=False)
    Y_unknown: pandas.DataFrame = field(init=False)
    X_unknown: pandas.DataFrame = field(init=False)
    X_transformed: pandas.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        """ Empty for now """
    
    # Sets the unpredicted keys
    def set_unpredicted_keys(self, keys: pandas.Series) -> None:
        self.unpredicted_keys = keys

    # Function for reading in data to classify from database
    def read_in_data(self) -> bool:
        
        try:
            data = self.handler.get_dataset()
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(e)

        if data is None:
            return False
        
        # Set the column names of the data array
        column_names = self.handler.config.get_column_names()
        class_column = self.handler.config.get_class_column_name()
        id_column = self.handler.config.get_id_column_name()
        
        # TODO: validate_dataset should probably do a report of potentional issues, or lead into the function that does
        dataset = self.validate_dataset(data, column_names, class_column)
        
        dataset = self.shuffle_dataset(dataset)
        
        self.dataset, self.keys = self.split_keys_from_dataset(dataset, id_column)
        
        # Extract unique class labels from dataset
        self.classes = list(set(self.dataset[class_column].tolist()))

        # Pick out the classification column. This is the "right hand side" of the problem.
        self.Y = self.dataset[class_column]

        
        if self.handler.config.should_train():
            self.handler.logger.investigate_dataset(self.dataset, class_column) # Returns True if the investigation/printing was not suppressed
        
        return True

    def validate_dataset(self, data: list, column_names: list, class_column: str) -> pandas.DataFrame:
        dataset = pandas.DataFrame(data, columns = column_names)
        
        # Make sure the class column is a categorical variable by setting it as string
        try:
            dataset.astype({class_column: 'str'}, copy=False)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(f"Could not convert class column {class_column} to string variable: {e}")
            
        # Make an extensive search through the data for any inconsistencies (like NaNs and NoneType). 
        # Also convert datetime numerical variables to ordinals, i.e., convert them to the number of days 
        # or similar from a certain starting point, if any are left after the conversion above.
        #self.handler.logger.print_formatted_info(message="Consistency check")
        percent_checked = 0
        index_length = float(len(dataset.index))
        try:
            for index in dataset.index:
                old_percent_checked = percent_checked
                percent_checked = round(100.0*float(index)/index_length)
                # TODO: fix message
                #self.handler.logger.print_percentage_checked("Data checked of fetched", old_percent_checked, percent_checked)
                
                for key in dataset.columns:
                    item = dataset.at[index,key]
                    column_is_text = self.handler.config.column_is_text(key)
                    column_is_numeric = self.handler.config.column_is_numeric(key)
                    
                    if column_is_text or column_is_numeric:
                        checked_item = self.sanitize_value(item, column_is_text)
                    
                        # Save new value
                        if checked_item != item:
                            dataset.at[index,key] = checked_item
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(f"Something went wrong in inconsistency check at {key}: {item} ({e})")

        return dataset

    def shuffle_dataset(self, dataset: pandas.DataFrame) -> pandas.DataFrame:
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
        self.handler.logger.print_formatted_info("Shuffle data")
        num_un_pred = self.get_num_unpredicted_rows(dataset) 
        num_lines = len(dataset.index)
        dataset['rnd'] = np.concatenate([np.random.rand(num_lines - num_un_pred), [num_lines]*num_un_pred])
        dataset.sort_values(by = 'rnd', inplace = True )
        dataset.drop(['rnd'], axis = 1, inplace = True )

        return dataset
    
    def split_keys_from_dataset(self, dataset: pandas.DataFrame, id_column: str) -> tuple[pandas.DataFrame, pandas.Series]:
        # Use the unique id column from the data as the index column and take a copy, 
        # since it will not be used in the classification but only to reconnect each 
        # data row with classification results later on
        try:
            keys = dataset[id_column].copy(deep = True).apply(Helpers.get_rid_of_decimals)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(f"Could not convert to integer: {e}")
            
        
        try:
            dataset.set_index(keys.astype('int64'), drop=False, append=False, inplace=True, verify_integrity=False)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
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

    # Collapse all data text columns into a new column, which is necessary
    # for word-in-a-bag-technique
    def convert_textdata_to_numbers(self, model: Model):
        """ This divides the columns, recreating them as self.X (dataframe) in the end """
        # Continue with "left hand side":
        # Prepare empty DataFrames
        text_dataset = pandas.DataFrame()
        num_dataset = pandas.DataFrame()
        categorical_dataset = pandas.DataFrame()
        binarized_dataset = pandas.DataFrame()

        text_data = self.handler.config.is_text_data()

        # While we could use model.* directly, I prefer using local variable and then return to be updated
        label_binarizers = model.label_binarizers
        count_vectorizer = model.count_vectorizer
        tfid_transformer = model.tfid_transformer

        # Create the numerical dataset
        for column in self.handler.config.get_numerical_column_names():
            num_dataset = pandas.concat([num_dataset, self.dataset[column]], axis = 1)

        if text_data:
            # Make sure to find the categorical data automatically
            for column in self.handler.config.get_text_column_names():
                if self.is_categorical_data(self.dataset[column]) or column in label_binarizers.keys():
                    categorical_dataset = pandas.concat([categorical_dataset, self.dataset[column]], axis = 1)
                    picked = "picked"
                else:
                    text_dataset = pandas.concat([text_dataset, self.dataset[column]], axis = 1)
                    picked = "not picked"
                
                self.handler.logger.print_info(f"Text data {picked} for categorization: ", column)

             # For concatenation, we need to make sure all text data are 
            # really treated as text, and categorical data as categories
            self.handler.logger.print_info("Text Columns:", str(text_dataset.columns))
            if len(text_dataset.columns) > 0:

                text_dataset = text_dataset.applymap(str)
                
                # Concatenating text data such that the result is another DataFrame  
                # with a single column
                text_dataset = text_dataset.agg(' '.join, axis = 1)
                
                # Convert text data to numbers using word-in-a-bag technique
                text_dataset, count_vectorizer, tfid_transformer = self.word_in_a_bag_conversion(text_dataset, model)

            # Continue with handling the categorical data.
            # Binarize the data, resulting in more columns.
            if len(categorical_dataset.columns) > 0:
                categorical_dataset = categorical_dataset.applymap(str)
                generate_new_label_binarizers = (len(label_binarizers) == 0)
                for column in categorical_dataset.columns:
                    lb = None
                    if generate_new_label_binarizers:
                        lb = LabelBinarizer()
                        lb.fit(categorical_dataset[column])
                        label_binarizers[column] = lb
                    else:
                        lb = label_binarizers[column]
                    lb_results = lb.transform(categorical_dataset[column])
                    try:
                        if lb_results.shape[1] > 1:
                            lb_results_df = pandas.DataFrame(lb_results, columns=lb.classes_)
                            self.handler.logger.print_info(f"Column {column} was categorized with categories: {lb.classes_}")
                        else:
                            lb_results_df = pandas.DataFrame(lb_results, columns=[column])
                            self.handler.logger.print_info(f"Column {column} was binarized")
                    except ValueError as e:
                        self.handler.logger.print_warning(f"Column {column} could not be binarized: {e}")
                    binarized_dataset = pandas.concat([binarized_dataset, lb_results_df], axis = 1 )

        self.X = self.create_X([text_dataset, num_dataset, binarized_dataset])

        if text_data:
            self.handler.logger.print_formatted_info("After conversion of text data to numerical data")
            self.handler.logger.investigate_dataset( self.X, None, False, False )

        return label_binarizers, count_vectorizer, tfid_transformer
    
    def create_X(self, frames: list[pandas.DataFrame], index: pandas.Int64Index = None) -> pandas.DataFrame:
        """ Creates a dataframe from text and numerical data """
        if index is None:
            index = self.dataset.index
        
        X = pandas.DataFrame()
        for df in frames:
            X = self.concat_with_index(X, df, index)
        
        return X

    def concat_with_index(self, X: pandas.DataFrame, concat: pandas.DataFrame, index: pandas.Int64Index) -> pandas.DataFrame:
        """ The try/except ensures that the dataframe added has the right number of rows """
        try:
            concat.set_index(index, drop=False, append=False, inplace=True, verify_integrity=False)
        except ValueError:
            return X
        
        return pandas.concat([concat, X], axis = 1)

    def set_training_data(self) -> None:
        # We need to know the number of data rows that are currently not
        # classified (assuming that they are last in the dataset because they
        # were sorted above). Separate keys from data elements that are
        # already classified from the rest.
        num_un_pred = self.get_num_unpredicted_rows()
        self.unpredicted_keys = self.keys[-num_un_pred:]

        # Original training data are stored for later reference
        if num_un_pred > 0:
            self.X_original = self.dataset.head(-num_un_pred)
        else:
            self.X_original = self.dataset.copy(deep=True)

    # Use the bag of words technique to convert text corpus into numbers
    def word_in_a_bag_conversion(self, dataset: pandas.DataFrame, model: Model ) -> tuple:

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
        
        return pandas.DataFrame(X), count_vectorizer, tfid_transformer

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
                self.handler.logger.print_info(f"Detected language is: {my_language}")

        # Calculate the lexical richness
        try:
            lex = LexicalRichness(' '.join(dataset)) 
            self.handler.logger.print_info("#Words, #Terms and TTR for original text is {0}, {1}, {2:5.2f} %".format(lex.words,lex.terms,100*float(lex.ttr)))
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
            self.handler.logger.print_info("Using standard stop words: ", str(my_stop_words))
            if (self.handler.config.should_hex_encode()):
                for word in my_stop_words:
                    word = Helpers.cipher_encode_string(str(word))

            # Collect text specific stop words (already encrypted if encryption is on)
            text_specific_stop_words = []
            threshold = self.handler.config.get_stop_words_threshold()
            if threshold < 1.0:
                try:
                    stop_vectorizer = CountVectorizer(min_df = threshold)
                    stop_vectorizer.fit_transform(dataset)
                    text_specific_stop_words = stop_vectorizer.get_feature_names()
                    self.handler.logger.print_info("Using specific stop words: ", text_specific_stop_words)
                except ValueError as e:
                    self.handler.logger.print_warning(f"Specified stop words threshold at {threshold} generated no stop words.")
                
                my_stop_words = sorted(set(my_stop_words + text_specific_stop_words))
                self.handler.logger.print_info("Total list of stop words:", my_stop_words)

        # Use the stop words and count the words in the matrix        
        count_vectorizer = CountVectorizer(stop_words = my_stop_words)
        count_vectorizer.fit(dataset)

        return count_vectorizer

        
    # Feature selection (reduction) function for PCA or Nystroem transformation of data.
    # (RFE feature selection is built into the model while training and does not need to be
    # considered here.)
    # NB: This used to take number_of_components, from mode.num_selected_features
    def perform_feature_selection(self, model: Model ):
        # Early return if we shouldn't use feature election, or the selection is RFE
        if self.handler.config.feature_selection_in([Reduction.NON, Reduction.RFE]):
            return

        feature_selection_transform = model.transform
            
        t0 = time.time()
        
        # For only predictions, use the saved transform associated with trained model
        if feature_selection_transform is not None:
            self.X = feature_selection_transform.transform(self.X)

        # In the case of training, we compute a new transformation
        else:
            feature_selection = self.handler.config.get_feature_selection()
            self.X, feature_selection_transform = feature_selection.call_transformation_theory(
                    logger=self.handler.logger, X=self.X, num_selected_features=self.handler.config.get_num_selected_features()
            )
            # Below is old version which uses call_transformation rather than call_transformation_theory(). 
            # Will be deleted once accuracy of latter confirmed
            #num_selected_features = self.handler.config.get_num_selected_features()
            #if feature_selection.has_function():
            #    self.handler.logger.print_info(f"{feature_selection.full_name} transformation of dataset under way...")
            #    self.X, feature_selection_transform = feature_selection.call_transformation(
            #        logger=self.handler.logger, X=self.X, num_selected_features=num_selected_features
            #    )
            


        t = time.time() - t0
        self.handler.logger.print_info(f"Feature reduction took {str(round(t,2))}  sec.")
        return feature_selection_transform
        
    # Split dataset into training and validation parts
    def split_dataset(self) -> bool:
        if not self.handler.config.should_train():
            self.X_unknown = self.X
            self.Y_unknown = self.Y
            
            return False
        
        self.handler.logger.print_progress(message="Split dataset for machine learning")
        num_lower = self.get_num_unpredicted_rows()
        
        # First, split X and Y in two parts: the upper part, to be used in training,
        # and the lower part, to be classified later on
        [X_upper, X_lower] = np.split(self.X, [self.X.shape[0]-num_lower], axis = 0)
        [Y_upper, Y_lower] = np.split(self.Y, [self.Y.shape[0]-num_lower], axis = 0)

        # Split-out validation dataset from the upper part (do not use random order here)
        testsize = self.handler.config.get_test_size()
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split( 
            X_upper, Y_upper, test_size = testsize, shuffle = False, random_state = None, stratify = None
            )

        self.Y_unknown = Y_lower
        self.X_unknown = X_lower
        
        pandas.testing.assert_series_equal(Y_upper, pandas.concat([self.Y_train, self.Y_validation], axis = 0) )
        return True
        
    # Find out if a DataFrame column contains categorical data or not
    def is_categorical_data(self, column: pandas.Series) -> bool:
        if not (self.handler.config.should_train() and self.handler.config.use_categorization()):
            return False
        
        return column.value_counts().count() <= self.LIMIT_IS_CATEGORICAL or self.handler.config.is_categorical(column.name)
        
    # Calculate the number of unclassified rows in data matrix
    def get_num_unpredicted_rows(self, dataset: pandas.DataFrame = None) -> int:
        if dataset is None:
            dataset = self.dataset
        num = 0
        for item in dataset[self.handler.config.get_class_column_name()]:
            if item == None or not str(item).strip(): # Corresponds to empty string and SQL NULL
                num += 1
        return num

@dataclass
class Model:
    label_binarizers: dict = field(default_factory=dict)
    count_vectorizer: CountVectorizer = field(default=None)
    tfid_transformer: TfidfTransformer = field(default=None)
    algorithm: Algorithm = field(default=None)
    preprocess: Preprocess = field(default=None)
    model: Pipeline = field(default=None)
    transform: typing.Any = field(default=None)

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
        if self.algorithm is None or self.preprocess is None:
            return ""

        return get_model_name(self.algorithm, self.preprocess)
                

@dataclass
class ModelHandler:
    handler: IAFHandler
    model: Model = field(init=False)
    
    use_feature_selection: bool = field(init=False)
    text_data: bool = field(init=False)
    
    def __post_init__(self) -> None:
        """ Empty for now """
    
    def load_model(self, filename: str = None) -> None:
        """ Called explicitly in run() """
        if filename is None:
            self.model = self.load_empty_model()
        else:
            self.model = self.load_model_from_file(filename)

    # Load ml model
    def load_model_from_file(self, filename: str) -> Model:
        try:
            _, label_binarizers, count_vectorizer, tfid_transformer, feature_selection_transform, model_name, model = pickle.load(open(filename, 'rb'))
        except Exception as e:
            self.handler.logger.print_warning(f"Something went wrong on loading model from file: {e}")
            return None
        
        model_class = Model(
            label_binarizers=label_binarizers,
            count_vectorizer=count_vectorizer,
            tfid_transformer=tfid_transformer,
            transform=feature_selection_transform,
            algorithm=model_name[0],
            preprocess=model_name[1],
            model=model
        )
        
        if self.handler.config.should_predict():
            self.handler.config.set_num_selected_features(model.n_features_in_)

        return model_class

    # load pipeline
    def load_pipeline_from_file(self, filename: str) -> Pipeline:
        self.handler.logger.print_info(f"loading pipeline from {filename}")
        model = self.load_model_from_file(filename)

        return model.model
    
    # Sets default (read: empty) values
    def load_empty_model(self) -> Model:
        return Model(label_binarizers={}, count_vectorizer=None, tfid_transformer=None, transform=None)

    def train_model(self, X_train: pandas.DataFrame, Y_train: pandas.DataFrame) -> None:
        # TODO: fix message
        #self.handler.logger.print_progress(message="Check and train algorithms for best model")
        
        try:
            self.model = self.get_model_from(X_train, Y_train)
    
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise ModelException(f"Something went wrong on training model: {str(e)}")

        self.save_model_to_file(self.handler.config.get_model_filename())

    def get_model_from(self, X_train: pandas.DataFrame, Y_train: pandas.DataFrame) -> Model:
        k = min(10, Helpers.find_smallest_class_number(Y_train))
        if k < 10:
            self.handler.logger.print_info(f"Using non-standard k-value for spotcheck of algorithms: {k}")
        
        model = self.spot_check_ml_algorithms(X_train, Y_train, k)
        
        model.model.fit(X_train, Y_train)

        return model

    # Train ml model
    def train_picked_model(self, model: Pipeline, X: pandas.DataFrame, Y: pandas.DataFrame) -> Pipeline:
        # Train model
        try:
            model.fit(X, Y)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise ModelException(f"Something went wrong on training picked model: {str(e)}")
            

        return model

    # While more code, this should (hopefully) be easier to read
    def should_run_computation(self, current_algorithm: Algorithm, current_preprocessor: Preprocess) -> bool:
        chosen_algorithm = self.handler.config.get_algorithm()
        chosen_preprocessor = self.handler.config.get_preprocessor()

        # RLRN and RFE do not get along
        if current_algorithm == Algorithm.RLRN and self.handler.config.use_RFE():
            return False
        
        # If both of these are ALL, it doesn't matter where in the set we are
        if chosen_algorithm == Algorithm.ALL and chosen_preprocessor == Preprocess.ALL:
            return True

        # If both the current one are equal to the chosen ones, carry on
        if current_algorithm == chosen_algorithm and current_preprocessor == chosen_preprocessor:
            return True

        # Two edge cases: A is All and P is Current, or A is Current and P is All
        if chosen_algorithm == Algorithm.ALL and current_preprocessor == chosen_preprocessor:
            return True

        if current_algorithm == chosen_algorithm and chosen_preprocessor == Preprocess.ALL:
            return True

        return False

    # Spot Check Algorithms.
    # We do an extensive search of the best algorithm in comparison with the best
    # preprocessing.
    def spot_check_ml_algorithms(self, X_train: pandas.DataFrame, Y_train: pandas.DataFrame, k:int=10) -> Model:
        # Save standard progress text
        standardProgressText = "Check and train algorithms for best model"
        self.handler.logger.print_info("Spot check ml algorithms...")
        
        algorithms = Algorithm.list_callable_algorithms(
            size=X_train.shape[0], 
            max_iterations=self.handler.config.get_max_iterations()
        )

        preprocessors = Preprocess.list_callable_preprocessors(is_text_data=self.handler.config.is_text_data())
        # Evaluate each model in turn in combination with all preprocessing methods
        best_mean = 0.0
        best_stdev = 1.0
        trained_pipeline = None
        best_algorithm = None
        best_preprocessor = None
        
        # Make evaluation of model
        try:
            #if k < 2: # What to do?
            kfold = StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise ModelException(f"StratifiedKfold raised an exception with message: {e}")
        
        best_feature_selection = X_train.shape[1]
        first_round = True
        # TODO: fix message
        self.handler.logger.print_table_row(items=["Name","Prep.","#Feat.","Mean","Std","Time","Failure"], divisor="=")

        numMinorTasks = len(algorithms) * len(preprocessors)
        percentAddPerMinorTask = (1.0-self.handler.progression["percentPerMajorTask"]*self.handler.progression["majorTasks"]) / float(numMinorTasks)
        
        # Loop over the algorithms
        for algorithm, algorithm_callable in algorithms:
            # Loop over pre-processing methods
            for preprocessor, preprocessor_callable in preprocessors:
                if not self.should_run_computation(algorithm, preprocessor):
                    #self.handler.logger.print_progress(message=f"Skipping ({algorithm.name}-{preprocessor.name}) due to config")
                    continue
                # Update progressbar percent and label
                self.handler.logger.print_progress(message=f"{standardProgressText} ({algorithm.name}-{preprocessor.name})")
                if not first_round: 
                    """ Empty for now, while we have the messages disabled """
                    # TODO: fix message
                    #self.handler.update_progress(percent=percentAddPerMinorTask)
                else:
                    first_round = False


                # Add feature selection if selected, i.e., the option of reducing the number of variables used.
                # Make a binary search for the optimal dimensions.
                max_features_selection = X_train.shape[1]
                min_features_selection = 0 if self.handler.config.use_RFE() else max_features_selection
                
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
                        current_pipeline, cv_results, failure = \
                            self.create_pipeline_and_cv(algorithm, preprocessor, algorithm_callable, preprocessor_callable, kfold, X_train, Y_train, num_features)
                    except ModelException:
                        # If any exceptions happen, continue to next step in the loop
                        break

                    # Stop the stopwatch
                    t = time.time() - t0

                    # For current settings, calculate score
                    temp_score = cv_results.mean()
                    temp_stdev = cv_results.std()

                    # Print results to screen
                    #if False: # TODO: fix message
                    #if self.handler.config.is_verbose(): # TODO: print prettier
                    print("{0:>4s}-{1:<6s}{2:6d}{3:8.3f}{4:8.3f}{5:11.3f} {6:<30s}".
                            format(algorithm.name,preprocessor.name,num_features,temp_score,temp_stdev,t,failure))

                    # Evaluate if feature selection changed accuracy or not. 
                    #rfe_score, max_features_selection, min_features_selection = self.calculate_current_features(temp_score, rfe_score, num_features, max_features_selection, min_features_selection)
                    # Notice: Better or same score with less variables are both seen as an improvement,
                    # since the chance of finding an improvement increases when number of variables decrease
                    if  temp_score >= rfe_score:
                        rfe_score = temp_score
                        max_features_selection = num_features   # We need to reduce more features
                    else:
                        min_features_selection = num_features   # We have reduced too much already  

                    # Save result if it is the overall best (inside RFE-while)
                    # Notice the difference from above, here we demand a better score.
                    if self.is_best_run_yet(temp_score, temp_stdev, best_mean, best_stdev):
                    #if temp_score > best_mean or (temp_score == best_mean and temp_stdev < best_std):
                        trained_pipeline = current_pipeline
                        best_algorithm = algorithm
                        best_preprocessor = preprocessor
                        best_mean = temp_score
                        best_stdev = temp_stdev
                        best_feature_selection = num_features

        updates = {"algorithm": best_algorithm, "preprocessor" : best_preprocessor, "num_selected_features": best_feature_selection}
        self.handler.config.update_attributes(type="mode", updates=updates)
        

        best_model = self.model
        best_model.algorithm = best_algorithm
        best_model.preprocess = best_preprocessor
        best_model.model = trained_pipeline
        
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

    def is_best_run_yet(self, current_mean: float, current_stdev: float, best_mean: float, best_stdev: float) -> bool:
        """ Calculates if this round is better than any prior """

        if current_mean < best_mean: # Obviously if current is less it's worse
            return False
        if current_mean > best_mean: # If it is better, it is better
            return True

        # This means that the current and best means are equal, so compare the standard deviations, lower standard deviation is better
        return current_stdev < best_stdev

    def create_pipeline_and_cv(self, algorithm: Algorithm, preprocessor: Preprocess, estimator: Estimator, transform: Transform, kfold: StratifiedKFold, X: pandas.DataFrame, y: pandas.DataFrame, num_features: int):
        """ The flow for each algorithm-preprocessor pair, broken out to simplify testing
        
          Parameters
        ----------
        algorithm : Algorithm
            Which algorithm is the base for the pipeline

        preprocess : Preprocess
            Which preprocess is used (if None, value is Preprocess.NON)

        estimator : Estimator
            An object with the fit() method

        transform : Transform
            An object with the fit() and transform() methods

        kfold : StratifiedKfold
            Cross-validation generator

        X : DataFrame
            The data which is processed

        y : DataFrame
            Classes for the processed data

        num_features : int
            What number of features is being used for the processing

        Returns
        -------
        pipeline : Pipeline or Estimator
           The pipeline the dataset gets transformed and/or fitted with.

        cv_results : ndarray
            The results from the cross-validation scoring
        """
        exception = None
        message = ""
        try:
            # Apply feature selection to current model and number of features.
            modified_estimator = self.modify_algorithm(estimator, num_features, X, y)
        
            # Build pipeline of model and preprocessor.
            current_pipeline = self.get_pipeline(algorithm, modified_estimator, transform)
            
            # Now make kfolded cross evaluation
            cv_results = self.get_cross_val_score(current_pipeline, X, y, kfold, algorithm)
        #except ValueError as exception:
        #    raise ModelException(f"Creating pipeline or getting cross val score failed in {algorithm.name}-{preprocessor.name}")
            # This warning kept to not forget it
            #self.handler.logger.print_warning(f"Pipeline ({algorithm.name}-{preprocessor.name}) raised a ValueError in cross_val_score. Skipping to next")
        #except IndexError as exception:
        #    raise ModelException(f"Creating pipeline failed in {algorithm.name}-{preprocessor.name} because {exception}")
        except Exception as ex:
            cv_results = np.array([np.nan])
            exception = "{0}: {1!r}".format(type(ex).__name__, ex.args)
            #raise(ex)

        return current_pipeline, cv_results, str(exception)
 
    
    def modify_algorithm(self, estimator: Estimator, n_features_to_select: int, X: pandas.DataFrame, y: pandas.DataFrame) -> Estimator:
        """ Modifies the algorithm (callable) based on config 
        
          Parameters
        ----------
        estimator : Estimator
            An object with the fit() method

        n_features_to_select : int

        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample (depending on the object's type)

        Returns
        -------
        estimator : Estimator
            The modified or non-modified estimator 
        """

        modified_estimator = estimator
        
        # Apply feature selection to current model and number of features.
        if self.handler.config.use_RFE():
            rfe = RFE(estimator, n_features_to_select=n_features_to_select)
            modified_estimator = rfe.fit(X, y)
            
        return modified_estimator
    
    
    def get_pipeline(self, algorithm: Algorithm, estimator: Estimator, transform: Transform) -> Union[Pipeline, Estimator]:
        """ Decides on which pipeline to use based on configuration and current algorithm 
        
          Parameters
        ----------
        algorithm : Algorithm
            Which algorithm is the base for the pipeline

        estimator : Estimator
            An object with the fit() method

        transform : Transform
            An object with the fit() and transform() methods

        Returns
        -------
        pipeline : Pipeline or Estimator
           The pipeline the dataset gets transformed and/or fitted with.
        """
        possible_steps = [
            (algorithm.name, estimator)
        ]
        possible_steps.insert(0, ("smote", self.handler.config.get_smote()))
        possible_steps.insert(1, ("undersample", self.handler.config.get_undersampler()))
        possible_steps.insert(2, ("preprocessor", transform))
        
        steps = [step  for step in possible_steps if hasattr(step[1] , "fit") and callable(getattr(step[1] , "fit"))] # List of 1 to 4 elements
        
        if len(steps) == 1: # No pipeline if the only one is the estimator
            return estimator

        # Robust algorithms all use ImbPipeline, Smote and/or Undersample (Config options) do too
        if algorithm.use_imb_pipeline() or self.handler.config.use_imb_pipeline(): 
            return ImbPipeline(steps=steps)

        return Pipeline(steps=steps)


    def get_cross_val_score(self, pipeline: Pipeline, X: pandas.DataFrame, y: pandas.DataFrame, kfold: StratifiedKFold, algorithm: Algorithm) -> np.ndarray:
        """ Allows to change call to cross_val_score based on algorithm. 
        
          Parameters
        ----------
        pipeline : Pipeline
            Which pipeline is being scored

        X : Dataframe
            The data the pipeline is being scored on

        y : DataFrame
            The classes for the data

        kfold : StratifiedKFold
            Cross-validation generator

        algorithm : Algorithm
            Which algorithm is being evaluated

        Returns
        -------
        score : ndarray
           The result from the scoring
        """
        # Now make kfolded cross evaluation
        scorer_mechanism = self.handler.config.get_scoring_mechanism()
        fit_params = {}
        
        for key, function_name in algorithm.fit_params.items():
            if hasattr(self, function_name) and callable(func := getattr(self, function_name)):
                fit_params[key] = func(X, y)

        return cross_val_score(pipeline, X, y=y, n_jobs=psutil.cpu_count(), cv=kfold, scoring=scorer_mechanism, fit_params=fit_params, error_score='raise') 

    # Save ml model and corresponding configuration
    def save_model_to_file(self, filename):
        try:
            save_config = self.handler.config.get_clean_config()
            data = [
                save_config,
                self.model.label_binarizers,
                self.model.count_vectorizer,
                self.model.tfid_transformer,
                self.model.transform,
                (self.model.algorithm, self.model.preprocess),
                self.model.model
            ]
            pickle.dump(data, open(filename,'wb'))

        except Exception as e:
            self.handler.logger.print_warning(f"Something went wrong on saving model to file: {e}")



@dataclass
class PredictionsHandler:
    LIMIT_MISPREDICTED = 20
    
    handler: IAFHandler
    could_predict_proba: bool = False
    probabilites: np.ndarray = field(init=False)
    predictions: np.ndarray = field(init=False)
    rates: np.ndarray = field(init=False)
    num_mispredicted: int = field(init=False)
    X_most_mispredicted: pandas.DataFrame = field(init=False)
    model: str = field(init=False)
    class_report: dict = field(init=False)

    def get_prediction_results(self, keys: pandas.Series) -> list:
        """ 
            Creates a list combining the values from the set prediction,
            using a given key. It is used to simplify saving the data
        """
        return_list = []
        
        try:
            for k,y,r,p in zip(keys.values, self.predictions, self.rates, self.probabilites):
                item = {
                    "key": k,
                    "prediction": y,
                    "rate": r,
                    "probabilities": self.get_probablities_as_string(p)
                }

                return_list.append(item)
        except AttributeError:
            return []
        
        return return_list

    def get_probablities_as_string(self, item) -> str:
        """ Gets a probabilities list as a comma-delimited string """
        try:
            iter(item)
            return ",".join([str(elem) for elem in item])
        except TypeError:
            # This only happens if the model couldn't predict, so uses the mean
            return item

    def get_mispredicted_dataframe(self) -> pandas.DataFrame:
        return self.X_most_mispredicted

    # Evaluate predictions
    def evaluate_predictions(self, Y, message="Unknown") -> None:
        self.handler.logger.print_progress(message="Evaluate predictions")
                
        self.handler.logger.print_info(f"Evaluation performed with evaluation data: " + message)
        self.handler.logger.print_info(f"Accuracy score for evaluation data: {accuracy_score(Y, self.predictions)}")
        self.handler.logger.print_info(f"Confusion matrix for evaluation data: \n\n{confusion_matrix(Y, self.predictions)}")
        self.handler.logger.print_info(f"Classification matrix for evaluation data: \n\n{classification_report(Y, self.predictions, zero_division='warn')}")

    # Evaluates mispredictions
    def evaluate_mispredictions(self, read_data_query: str, misplaced_filepath: str) -> None:
        if self.X_most_mispredicted.empty or not self.handler.config.should_display_mispredicted():
            return
        
        self.handler.logger.print_info(f"Total number of mispredicted elements: {self.num_mispredicted}")
        joiner = self.handler.config.get_id_column_name() + " = \'"
        most_mispredicted_query = read_data_query + "WHERE " +  joiner \
            + ("\' OR " + joiner).join([str(number) for number in self.X_most_mispredicted.index.tolist()]) + "\'"
        
        self.handler.logger.print_formatted_info(f"Most mispredicted during training (using {self.model})")
        self.handler.logger.print_info(str(self.X_most_mispredicted))
        self.handler.logger.print_info(f"Get the most misplaced data by SQL query:\n {most_mispredicted_query}")
        self.handler.logger.print_info(f"Or open the following csv-data file: \n\t {misplaced_filepath}")
        
        self.X_most_mispredicted.to_csv(path_or_buf = misplaced_filepath, sep = ';', na_rep='N/A', \
                                float_format=None, columns=None, header=True, index=True, \
                                index_label=self.handler.config.get_id_column_name(), mode='w', encoding='utf-8', \
                                compression='infer', quoting=None, quotechar='"', line_terminator=None, \
                                chunksize=None, date_format=None, doublequote=True, decimal=',', \
                                errors='strict')
    
    # Make predictions on dataset
    def make_predictions(self, model: Pipeline, X: pandas.DataFrame, classes: pandas.Series) -> bool:
        could_predict_proba = False
        try:
            predictions = model.predict(X)
        except ValueError as e:
            self.handler.logger.abort_cleanly(message=f"It seems like you need to regenerate your prediction model: {e}")
        try:
            probabilities = model.predict_proba(X)
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
        # TODO: range(len) is generally not ideal
        for i in range(len(self.predictions)):
            try:
                # This should probably be a list of a float, not a float
                prob = prob + [self.class_report[self.predictions[i]]['precision']]
            except KeyError as e:
                self.handler.logger.print_warning(f"probability collection failed for key {self.predictions[i]} with error {e}")
    
        self.handler.logger.print_info("Probabilities:", str(prob))
        self.probabilites = prob
        
    
    # Returns a list of mean and standard deviation
    def get_rates(self, as_string: bool = False) -> list:
        mean = np.mean(self.rates)
        std = np.std(self.rates)
        
        if as_string:
            return [str(mean), str(std)]

        return [mean, std]

    # Create classification report
    def get_classification_report(self, Y_validation: pandas.DataFrame, model: Model) -> list:
        self.class_report = classification_report(Y_validation, self.predictions, output_dict = True)

        return [self.class_report, model, self.handler.config.get_num_selected_features()]
    
    # Function for finding the n most mispredicted data rows
    # TODO: Clean up a bit more
    def most_mispredicted(self, X_original: pandas.DataFrame, model: Pipeline, ct_model: Pipeline, X_transformed: pandas.DataFrame, Y: pandas.DataFrame) -> None:
        # Calculate predictions for both total model and cross trained model
        for what_model, the_model in [("model retrained on all data", model), ("model cross trained on training data", ct_model)]:
            Y_pred = pandas.DataFrame(the_model.predict(X_transformed), index = Y.index)

            # Find the data rows where the real category is different from the predictions
            # Iterate over the indexes (they are now not in order)
            X_not = []
            for i in Y.index:
                try:
                    X_not.append((Y.loc[i] != Y_pred.loc[i]).bool())
                except Exception as e:
                    self.handler.logger.print_warning(f"Append of data row with index: {i} failed: {e}. Probably duplicate indicies in data!")
                    break

            # Quick end of loop if possible
            num_mispredicted = sum(elem == True for elem in X_not)
            if num_mispredicted != 0:
                break
        
        self.num_mispredicted = num_mispredicted
        self.model = what_model
            
        # Quick return if possible
        if num_mispredicted == 0:
            self.X_mispredicted = pandas.DataFrame()
            self.model = "no model produced mispredictions"
            
            return
        
        #
        self.handler.logger.print_info(f"Accuracy score for {what_model}: {accuracy_score(Y, Y_pred)}")

        # Select the found mispredicted data
        X_mispredicted = X_transformed.loc[X_not]

        # Predict probabilites
        try:
            Y_prob = the_model.predict_proba(X_mispredicted)
            could_predict_proba = True
        except Exception as e:
            self.handler.logger.print_warning(f"Could not predict probabilities: {e}")
            could_predict_proba = False

        #  Re-insert original data columns but drop the class column
        X_mispredicted = X_original.loc[X_not]
        X_mispredicted = X_mispredicted.drop(self.handler.config.get_class_column_name(), axis = 1)

        # Add other columns to mispredicted data
        X_mispredicted.insert(0, "Actual", Y.loc[X_not])
        X_mispredicted.insert(0, "Predicted", Y_pred.loc[X_not].values)
        
        # Add probabilities and sort only if they could be calculated above, otherwise
        # return a random sample of mispredicted
        try:
            the_classes = the_model.classes_
        except AttributeError as e:
            self.handler.logger.print_info(f"No classes_ attribute in model, using original classes as fallback: {e}")
            the_classes = [y for y in set(Y) if y is not None]

        if not could_predict_proba:
            for i in range(len(the_classes)): # TODO: possibly rewrite this for loop
                X_mispredicted.insert(0, "P(" + the_classes[i] + ")", "N/A")
            n_limit = min(self.LIMIT_MISPREDICTED, X_mispredicted.shape[0])

            self.X_most_mispredicted = X_mispredicted.sample(n=n_limit)
            return
        
        Y_prob_max = np.amax(Y_prob, axis = 1)
        for i in reversed(range(Y_prob.shape[1])):
            X_mispredicted.insert(0, "P(" + the_classes[i] + ")", Y_prob[:,i])
        X_mispredicted.insert(0, "__Sort__", Y_prob_max)

        # Sort the dataframe on the first column and remove it
        X_mispredicted = X_mispredicted.sort_values("__Sort__", ascending = False)
        X_mispredicted = X_mispredicted.drop("__Sort__", axis = 1)

        # Keep only the top n_limit rows and return
        self.X_most_mispredicted = X_mispredicted.head(self.LIMIT_MISPREDICTED)
        
        return


def main():
    handler = IAFHandler(DataLayer, Config, Logger)

    # Should hopefully not give errors
    handler.add_handler(name="dataset")

    # Gives exception
    #handler.add_handler(name="model")


if __name__ == "__main__":
    main()
