from __future__ import annotations

import base64
import pickle
import time
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
from regex import R
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelBinarizer
from stop_words import get_stop_words

from Config import Algorithm, Preprocess, Reduction, get_model_name
from IAFExceptions import DatasetException, ModelException, HandlerException


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

    def investigate_dataset(self, dataset: pandas.DataFrame, show_class_distribution: bool = True, show_statistics: bool = True) -> bool:
        """ Print information about dataset """
    
    def print_warning(self, *args) -> None:
        """ print warning """

    def print_table_row(self, items: list[str], divisor: str = None) -> None:
        """ Prints row of items, with optional divisor"""

class DataLayer(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""

    def get_dataset(self, num_rows: int, train: bool, predict: bool):
        """ Get the dataset, query and number of rows"""

class Config(Protocol):
    # Methods to hide implementation of Config
    def set_num_selected_features(self, num_features: int) -> None:
        """ Updates the config with the number """

    def get_model_filename(self) -> str:
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

    def update_values(self, updates: dict,  type: str = None) -> bool:
        """ Updates several values inside the config """

    def get_scoring_mechanism(self):
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
    

    def get_handler(self, name: str):
        if name in self.handlers:
            return self.handlers[name]

        raise HandlerException(f"{name.capitalize()}Handler does not exist")


    def get_dataset(self):
        """ By putting this here, Datalayer does not need to be supplied to Dataset Handler"""
        # return data, non_ordered_query, num_lines
        return self.datalayer.get_dataset(self.config.get_max_limit(), self.config.should_train(), self.config.should_predict())

    # Updates the progress and notifies the logger
    # Currently duplicated over Classifier and IAFHandler, but that's for later
    def update_progress(self, percent: float, message: str = None) -> float:
        self.progression.progress += percent

        if message is None:
            self.handler.logger.print_progress(percent = self.progression.progress)
        else:
            self.handler.logger.print_progress(message=message, percent = self.progression.progress)

        return self.progression.progress

    

@dataclass
class DatasetHandler:
    STANDARD_LANG = "sv"
    LIMIT_IS_CATEGORICAL = 30
    
    handler: IAFHandler
    dataset: pandas.DataFrame = field(init=False)
    classes: list[str] = field(init=False)
    queries: dict = field(default_factory=dict)
    keys: pandas.Series = field(init=False)
    unpredicted_keys: pandas.Series = field(init=False)
    X_original: pandas.DataFrame = field(init=False)
    X: pandas.DataFrame = field(init=False)
    Y: pandas.DataFrame = field(init=False)
    X_train: pandas.DataFrame = field(init=False)
    X_validation: pandas.DataFrame = field(init=False)
    Y_train: pandas.DataFrame = field(init=False)
    Y_validation: pandas.DataFrame = field(init=False)
    Y_unknown: pandas.DataFrame = field(init=False)
    X_unknown: pandas.DataFrame = field(init=False)
    X_transformed: pandas.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        """ Empty for now """
    
    # Function for reading in data to classify from database
    def read_in_data(self) -> bool:
        
        try:
            data, query, num_lines = self.handler.get_dataset()
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(e)

        if data is None:
            return False


        # Set the column names of the data array
        column_names = self.handler.config.get_column_names()
        
        
        self.dataset = pandas.DataFrame(data, columns = column_names)
        
        # Make sure the class column is a categorical variable by setting it as string
        class_column = self.handler.config.get_class_column_name()
        try:
            self.dataset.astype({class_column: 'str'}, copy=False)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(f"Could not convert class column {class_column} to string variable: {e}")
            
        # Extract unique class labels from dataset
        unique_classes = list(set(self.dataset[class_column].tolist()))

        #TODO: Clean up the clean-up
        # Make an extensive search through the data for any inconsistencies (like NaNs and NoneType). 
        # Also convert datetime numerical variables to ordinals, i.e., convert them to the number of days 
        # or similar from a certain starting point, if any are left after the conversion above.
        self.handler.logger.print_formatted_info(message="Consistency check")
        change = False
        percent_checked = 0
        index_length = float(len(self.dataset.index))
        try:
            for index in self.dataset.index:
                old_percent_checked = percent_checked
                percent_checked = round(100.0*float(index)/index_length)
                # TODO: \r is to have the Data checked updated, or some-such. Can it be moved?
                if self.handler.config.is_verbose() and percent_checked > old_percent_checked:
                    print("Data checked of fetched: " + str(percent_checked) + " %", end='\r')
                for key in self.dataset.columns:
                    item = self.dataset.at[index,key]
                    print(f"{item=}, {index=}, {key=}")
                    # Set NoneType objects  as zero or empty strings
                    if (key in self.handler.config.get_numerical_column_names() or \
                        key in self.handler.config.get_text_column_names()) and item == None:
                        if key in self.handler.config.get_numerical_column_names():
                            item = 0
                        else:
                            item = ""
                        change = True

                    # Convert numerical datetime values to ordinals
                    elif key in self.handler.config.get_numerical_column_names() and is_datetime(item):
                        item = datetime.toordinal(item)
                        change = True

                    # Set remaining numerical values that cannot be casted as integer or floating point numbers to zero, i.e., do not
                    # take then into account
                    elif key in self.handler.config.get_numerical_column_names() and \
                        not (is_int(item) or is_float(item)):
                        item = 0
                        change = True

                    # Set text values that cannot be casted as strings to empty strings
                    elif key in self.handler.config.get_text_column_names() and type(item) != str and not is_str(item):
                        item = ""
                        change = True

                    # Remove line breaks from text strings
                    if key in self.handler.config.get_text_column_names():
                        item = item.replace('\n'," ").replace('\r'," ").strip()
                        change = True

                    # Save new value
                    if change:
                        self.dataset.at[index,key] = item
                        change = False
        except Exception as e:
            print(e)
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(f"Something went wrong in inconsistency check at {key}: {item} ({e})")
    
        # Shuffle the upper part of the data, such that all already classified material are
        # put in random order keeping the unclassified material in the bottom
        # Notice: perhaps this operation is now obsolete, since we now use a NEWID() in the 
        # data query above
        self.handler.logger.print_formatted_info("Shuffle data")
        num_un_pred = self.get_num_unpredicted_rows()
        self.dataset['rnd'] = np.concatenate([np.random.rand(num_lines - num_un_pred), [num_lines]*num_un_pred])
        self.dataset.sort_values(by = 'rnd', inplace = True )
        self.dataset.drop(['rnd'], axis = 1, inplace = True )

        # Use the unique id column from the data as the index column and take a copy, 
        # since it will not be used in the classification but only to reconnect each 
        # data row with classification results later on
        try:
            keys = self.dataset[self.handler.config.get_id_column_name()].copy(deep = True).apply(get_rid_of_decimals)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(f"Could not convert integer: {e}")
            
        
        try:
            self.dataset.set_index(keys.astype('int64'), drop=False, append=False, inplace=True, verify_integrity=False)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(f"Could not set index for dataset: {e}")
        
        self.dataset = self.dataset.drop([self.handler.config.get_id_column_name()], axis = 1)


        self.keys = keys
        self.queries["read_data"] = query
        self.classes = unique_classes

        if self.handler.config.should_train():
            self.handler.logger.investigate_dataset(self.dataset) # Returns True if the investigation/printing was not suppressed
        
        return True

    # Collapse all data text columns into a new column, which is necessary
    # for word-in-a-bag-technique
    def convert_textdata_to_numbers(self, model: Model): #, label_binarizers:dict = {}, count_vectorizer: CountVectorizer = None, tfid_transformer: TfidfTransformer = None ) -> tuple:

        # Pick out the classification column. This is the 
        # "right hand side" of the problem.
        self.Y = self.dataset[self.handler.config.get_class_column_name()]

        # Continue with "left hand side":
        # Prepare two empty DataFrames
        text_dataset = pandas.DataFrame()
        num_dataset = pandas.DataFrame()
        categorical_dataset = pandas.DataFrame()
        binarized_dataset = pandas.DataFrame()

        # While we could use model.* directly, I prefer using local variable and then return to be updated
        label_binarizers = model.label_binarizers
        count_vectorizer = model.count_vectorizer
        tfid_transformer = model.tfid_transformer

        # First, separate all the text data from the numerical data, and
        # make sure to find the categorical data automatically
        if self.handler.config.is_text_data():
            text_columns = self.handler.config.get_text_column_names()
            for column in text_columns:
                if (self.handler.config.should_train() and self.handler.config.use_categorization() and  \
                    self.is_categorical_data(self.dataset[column])) or column in label_binarizers.keys():
                    categorical_dataset = pandas.concat([categorical_dataset, self.dataset[column]], axis = 1)
                    self.handler.logger.print_info("Text data picked for categorization: ", column)
                else:
                    text_dataset = pandas.concat([text_dataset, self.dataset[column]], axis = 1)
                    self.handler.logger.print_info("Text data NOT picked for categorization: ", column)
                 
        if self.handler.config.is_numerical_data():
            num_columns = self.handler.config.get_numerical_column_names()
            for column in num_columns:
                num_dataset = pandas.concat([num_dataset, self.dataset[column]], axis = 1)

        # For concatenation, we need to make sure all text data are 
        # really treated as text, and categorical data as categories
        if self.handler.config.is_text_data():
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

        self.X = pandas.DataFrame()
        if self.handler.config.is_text_data() and text_dataset.shape[1] > 0:
            text_dataset.set_index(self.dataset.index, drop=False, append=False, inplace=True, \
                                   verify_integrity=False)
            self.X = pandas.concat([text_dataset, self.X], axis = 1)
        if self.handler.config.is_numerical_data() and num_dataset.shape[1] > 0:
            num_dataset.set_index(self.dataset.index, drop=False, append=False, inplace=True, \
                                  verify_integrity=False)
            self.X = pandas.concat([num_dataset, self.X], axis = 1)
        if self.handler.config.is_text_data() and binarized_dataset.shape[1] > 0:
            binarized_dataset.set_index(self.dataset.index, drop=False, append=False, inplace=True, \
                                        verify_integrity=False)
            self.X = pandas.concat([binarized_dataset, self.X], axis = 1)

        if self.handler.config.is_text_data():
            self.handler.logger.print_formatted_info("After conversion of text data to numerical data")
            self.handler.logger.investigate_dataset( self.X, False, False )

        return label_binarizers, count_vectorizer, tfid_transformer
    
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
    def word_in_a_bag_conversion(self, dataset: pandas.DataFrame, model: Model = None ) -> tuple:

        # Start working with datavalues in array
        X = dataset.values

        # Find actual languange if stop words are used
        if model.count_vectorizer == None:
            my_language = None
            if self.handler.config.use_stop_words():
                try:
                    my_language = langdetect.detect(' '.join(X))
                except Exception as e:
                    my_language = self.STANDARD_LANG
                    self.handler.logger.print_warning(f"Language could not be detected automatically: {e}. Fallback option, use: {my_language}.")
                else:
                    self.handler.logger.print_info(f"Detected language is: {my_language}")

            
            # Calculate the lexical richness
            try:
                lex = LexicalRichness(' '.join(X)) 
                self.handler.logger.print_info("#Words, #Terms and TTR for original text is {0}, {1}, {2:5.2f} %".format(lex.words,lex.terms,100*float(lex.ttr)))
            except Exception as e:
                self.handler.logger.print_warning(f"Could not calculate lexical richness: {e}")

        # Mask all material by encryption (optional)
        if (self.handler.config.should_hex_encode()):
            X = do_hex_base64_encode_on_data(X)

        # Text must be turned into numerical feature vectors ("bag-of-words"-technique).
        # If selected, remove stop words
        if model.count_vectorizer == None:
            my_stop_words = None
            if self.handler.config.use_stop_words():

                # Get the languange specific stop words and encrypt them if necessary
                my_stop_words = get_stop_words(my_language)
                self.handler.logger.print_info("Using standard stop words: ", str(my_stop_words))
                if (self.handler.config.should_hex_encode()):
                    for word in my_stop_words:
                        word = cipher_encode_string(str(word))

                # Collect text specific stop words (already encrypted if encryption is on)
                text_specific_stop_words = []
                threshold = self.handler.config.get_stop_words_threshold()
                if threshold < 1.0:
                    try:
                        stop_vectorizer = CountVectorizer(min_df = threshold)
                        stop_vectorizer.fit_transform(X)
                        text_specific_stop_words = stop_vectorizer.get_feature_names()
                        self.handler.logger.print_info("Using specific stop words: ", text_specific_stop_words)
                    except ValueError as e:
                        self.handler.logger.print_warning(f"Specified stop words threshold at {threshold} generated no stop words.")
                    
                    my_stop_words = sorted(set(my_stop_words + text_specific_stop_words))
                    self.handler.logger.print_info("Total list of stop words:", my_stop_words)

            # Use the stop words and count the words in the matrix        
            count_vectorizer = CountVectorizer(stop_words = my_stop_words)
            count_vectorizer.fit(X)

        # Do the word in a bag now
        X = count_vectorizer.transform(X)

        # Also generate frequencies instead of occurences to normalize the information.
        if model.tfid_transformer == None:
            tfid_transformer = TfidfTransformer(use_idf = False)
            tfid_transformer.fit(X) 

        # Generate the sequences
        X = (tfid_transformer.transform(X)).toarray()

        return pandas.DataFrame(X), count_vectorizer, tfid_transformer

    # Feature selection (reduction) function for PCA or Nystroem transformation of data.
    # (RFE feature selection is built into the model while training and does not need to be
    # considered here.)
    # NB: This used to take number_of_compponents, from mode.num_selected_features
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
            num_selected_features = self.handler.config.get_num_selected_features()
            if feature_selection.has_transformation_function():
                self.handler.logger.print_info(f"{feature_selection.value} transformation of dataset under way...")
                self.X, feature_selection_transform = feature_selection.call_transformation(
                    self.handler.logger, self.X, num_selected_features)


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
        X_train, X_validation, Y_train, Y_validation = train_test_split( 
            X_upper, Y_upper, test_size = testsize, shuffle = False, random_state = None, stratify = None
            )

        self.X_train = X_train
        self.X_validation = X_validation
        self.Y_train = Y_train
        self.Y_validation = Y_validation
        self.Y_unknown = Y_lower
        self.X_unknown = X_lower
        
        return True
        
    # Find out if a DataFrame column contains categorical data or not
    def is_categorical_data(self, column):
        return column.value_counts().count() <= self.LIMIT_IS_CATEGORICAL or self.handler.config.is_categorical(column.name)


     # Calculate the number of unclassified rows in data matrix
    def get_num_unpredicted_rows(self):
        num = 0
        for item in self.dataset[self.handler.config.get_class_column_name()]:
            if item == None or not str(item).strip(): # Corresponds to empty string and SQL NULL
                num += 1
        return num

# Convert dataset to unreadable hex code
def do_hex_base64_encode_on_data(X):

    XX = X.copy()

    with np.nditer(XX, op_flags=['readwrite'], flags=["refs_ok"]) as iterator:
        for x in iterator:
            xval = str(x)
            xhex = cipher_encode_string(xval)
            x[...] = xhex 

    return XX

def cipher_encode_string(a):

    aa = a.split()
    b = ""
    for i in range(len(aa)):
        b += (str(
                base64.b64encode(
                    bytes(aa[i].encode("utf-8").hex(),"utf-8")
                )
            ) + " ")

    return b.strip()

def get_rid_of_decimals(x) -> int:
    return int(round(float(x)))

# Help routines for determining consistency of input data
def is_float(val):
    try:
        float(val)
    except ValueError:
        return False
    return True

def is_int(val):
    try:
        int(val)
    except ValueError:
        return False
    return True

def is_bool(val):
    return type(val)==bool

def is_str(val):
    is_other_type = \
        is_float(val) or \
        is_int(val) or \
        is_bool(val) or \
        is_datetime(val)
    return not is_other_type 
    
def is_datetime(val):
    try:
        if isinstance(val, datetime): #Simple, is already instance of datetime
            return True
    except ValueError:
        pass
    # Harder: test the value for many different datetime formats and see if any is correct.
    # If so, return true, otherwise, return false.
    the_formats = ['%Y-%m-%d %H:%M:%S','%Y-%m-%d','%Y-%m-%d %H:%M:%S.%f','%Y-%m-%d %H:%M:%S,%f', \
                    '%d/%m/%Y %H:%M:%S','%d/%m/%Y','%d/%m/%Y %H:%M:%S.%f', '%d/%m/%Y %H:%M:%S,%f']
    for the_format in the_formats:
        try:
            date_time = datetime.strptime(str(val), the_format)
            return isinstance(date_time, datetime)
        except ValueError:
            pass
    return False

@dataclass
class Model:
    label_binarizers: dict = field(default_factory=dict)
    count_vectorizer: CountVectorizer = field(default=None)
    tfid_transformer: TfidfTransformer = field(default=None)
    algorithm: Algorithm = field(default=None)
    preprocess: Preprocess = field(default=None)
    model: Pipeline = field(default=None)
    transform: typing.Any = field(default=None)

    def update_fields(self, fields: list[str], update_function: Callable = None) -> bool:
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
        self.model = self.load_model()

    # Loads model based on config
    def load_model(self) -> Model:
        if self.handler.config.should_train():
            return self.load_empty_model()
        
        return self.load_model_from_file(self.handler.config.get_model_filename())
            

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
        model = self.load_model_from_file(filename)

        return model.model
    
    # Sets default (read: empty) values
    def load_empty_model(self) -> Model:
        return Model(label_binarizers={}, count_vectorizer=None, tfid_transformer=None, transform=None)

    def train_model(self, X_train: pandas.DataFrame, Y_train: pandas.DataFrame) -> None:
        self.handler.logger.print_progress(message="Check and train algorithms for best model")
        
        try:
            self.model = self.get_model_from(X_train, Y_train)
    
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise ModelException(f"Something went wrong on training picked model: {str(e)}")


        self.save_model_to_file(self.handler.config.get_model_filename())

    def get_model_from(self, X_train: pandas.DataFrame, Y_train: pandas.DataFrame) -> Model:
        k = min(10, find_smallest_class_number(Y_train))
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

    def prepare_models_preprocessors(self, size) -> tuple[list, list]:
        # Add all algorithms in a list
        self.handler.logger.print_info("Spot check ml algorithms...")
        models = []
        for algo in Algorithm:
            if algo.has_algorithm_function():
                models.append((algo, algo.call_algorithm(max_iterations=self.handler.config.mode.max_iterations, size=size)))
        

        # Add different preprocessing methods in another list
        preprocessors = []
        for preprocessor in Preprocess:
            # This checks that the preprocessor has the function, and also that BIN is only added if there is text_data
            if preprocessor.has_preprocess_function() and (preprocessor.name != "BIN" or self.handler.config.is_text_data()):
                preprocessors.append((preprocessor, preprocessor.call_preprocess()))
                
        
        preprocessors.append((Preprocess.NON, None)) # In case they choose no preprocessor in config

        return models, preprocessors

    def get_feature_selection(self, max_features_selection: int) -> tuple(int, int):
        max_features_selection = max_features_selection
        
        # Make sure all numbers are propely set for feature selection interval
        if self.handler.config.use_feature_selection():
            min_features_selection = self.handler.config.get_num_selected_features()

            if min_features_selection > 0:
                max_features_selection = min_features_selection

            return max_features_selection, min_features_selection

        
        return max_features_selection, max_features_selection # No or minimal number of features are eliminated
    
    # Spot Check Algorithms.
    # We do an extensive search of the best algorithm in comparison with the best
    # preprocessing.
    def spot_check_ml_algorithms(self, X_train: pandas.DataFrame, Y_train: pandas.DataFrame, k:int=10) -> Model:

        # Save standard progress text
        standardProgressText = "Check and train algorithms for best model"
        models, preprocessors = self.prepare_models_preprocessors(size=X_train.shape[0])

        # Evaluate each model in turn in combination with all preprocessing methods
        names = []
        best_mean = 0.0
        best_std = 1.0
        trained_model = None
        temp_model = None
        algorithm = None
        pprocessor = None
        
        scorer_mechanism = self.handler.config.get_scoring_mechanism()

        smote = self.handler.config.get_smote()
        undersampler = self.handler.config.get_undersampler()

        # Make evaluation of model
        try:
            kfold = StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise ModelException(f"StratifiedKfold raised an exception with message: {e}")
        
        best_feature_selection = X_train.shape[1]
        first_round = True
        self.handler.logger.print_table_row(items=["Name","Prep.","#Feat.","Mean","Std","Time"], divisor="=")

        numMinorTasks = len(models) * len(preprocessors)
        percentAddPerMinorTask = (1.0-self.handler.progression["percentPerMajorTask"]*self.handler.progression["majorTasks"]) / float(numMinorTasks)
        
        # Loop over the models
        for name, model in models:
            # Loop over pre-processing methods
            for preprocessor_name, preprocessor in preprocessors:
                if not self.should_run_computation(name, preprocessor_name):
                    #self.handler.logger.print_progress(message=f"Skipping ({name.name}-{preprocessor_name.name}) due to config")
                    continue
                # Update progressbar percent and label
                self.handler.logger.print_progress(message=f"{standardProgressText} ({name.name}-{preprocessor_name.name})")
                if not first_round: 
                    self.handler.update_progress(percent=percentAddPerMinorTask)
                else:
                    first_round = False


                # Add feature selection if selected, i.e., the option of reducing the number of variables used.
                # Make a binary search for the optimal dimensions.
                max_features_selection, min_features_selection = self.get_feature_selection(X_train.shape[1])
                

                # Loop over feature selections span: break this loop when min and max reach the same value
                score = 0.0                                                 # Save the best values
                stdev = 1.0                                                 # so far.
                num_features = max_features_selection                       # Start with all features.
                first_feature_selection = True                              # Make special first round: use all features
                counter = 0
                while first_feature_selection or min_features_selection < max_features_selection:
                    counter += 1

                    # Update limits for binary search and break loop if we are done
                    if not first_feature_selection:
                        num_features = ceil((min_features_selection+max_features_selection) / 2)
                        if num_features == max_features_selection:          
                            break
                    else:
                        first_feature_selection = False
                        num_features = max_features_selection

                    # Calculate the time for this setting
                    t0 = time.time()

                    # Apply feature selection to current model and number of features.
                    # If feature selection is not applicable, set a flag to the loop is 
                    # ending after one iteration
                    temp_model = model
                    if self.handler.config.use_feature_selection() and self.handler.config.get_feature_selection() == Reduction.RFE:     
                        try:
                            rfe = RFE(temp_model, n_features_to_select=num_features)
                            temp_model = rfe.fit(X_train, Y_train)
                        except ValueError as e:
                            break

                    # Both algorithm and preprocessor should be used. Move on.
                    # Build pipline of model and preprocessor.
                    names.append((name,preprocessor_name))
                    if smote is None and undersampler is None:
                        if preprocessor is not None:
                            pipe = make_pipeline(preprocessor, temp_model)
                        else:
                            pipe = temp_model

                    # For use SMOTE and undersampling, different Pipeline is used
                    else:
                        steps = [('smote', smote ), ('under', undersampler), \
                                ('preprocessor', preprocessor), ('model', temp_model)]
                        pipe = ImbPipeline(steps=steps)
                        

                    # Now make kfolded cross evaluation
                    cv_results = None
                    try:
                        cv_results = cross_val_score(pipe, X_train, Y_train, cv=kfold, scoring=scorer_mechanism) 
                    except ValueError as e:
                        self.handler.logger.print_warning(f"Model {names[-1]} raised a ValueError in cross_val_score. Skipping to next")
                    else:
                        # Stop the stopwatch
                        t = time.time() - t0

                        # For current settings, calculate score
                        temp_score = cv_results.mean()
                        temp_stdev = cv_results.std()

                        # Print results to screen
                        if self.handler.config.is_verbose(): # TODO: print prettier
                            print("{0:>4s}-{1:<6s}{2:6d}{3:8.3f}{4:8.3f}{5:11.3f} s.".
                                  format(name,preprocessor_name,num_features,temp_score,temp_stdev,t))

                        # Evaluate if feature selection changed accuracy or not. 
                        # Notice: Better or same score with less variables are both seen as an improvement,
                        # since the chance of finding an improvement increases when number of variables decrease
                        if  temp_score >= score:
                            score = temp_score
                            stdev = temp_stdev
                            max_features_selection = num_features   # We need to reduce more features
                        else:
                            min_features_selection = num_features   # We have reduced too much already  

                         # Save result if it is the overall best
                         # Notice the difference from above, here we demand a better score.
                        if temp_score > best_mean or \
                           (temp_score == best_mean and temp_stdev < best_std):
                            trained_model = pipe
                            algorithm = name
                            pprocessor = preprocessor_name
                            best_mean = temp_score
                            best_std = temp_stdev
                            best_feature_selection = num_features

        updates = {"algorithm": algorithm, "preprocessor" : pprocessor, "num_selected_features": best_feature_selection}
        self.handler.config.update_values(type="mode", updates=updates)
        
        model = self.model
        model.algorithm = algorithm
        model. preprocess = pprocessor
        model.model = trained_model
        # Return best model for start making predictions
        return model

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

def find_smallest_class_number(Y):
    class_count = {}
    for elem in Y:
        if elem not in class_count:
            class_count[elem] = 1
        else:
            class_count[elem] += 1
    return max(1, min(class_count.values()))

@dataclass
class PredictionsHandler:
    LIMIT_MISPREDICTED = 20
    
    handler: IAFHandler
    could_predict_proba: bool = field(init=False)
    probabilites: np.ndarray = field(init=False)
    predictions: np.ndarray = field(init=False)
    rates: np.ndarray = field(init=False)
    num_mispredicted: int = field(init=False)
    X_most_mispredicted: pandas.DataFrame = field(init=False)
    model: str = field(init=False)
    class_report: dict = field(init=False)

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
    def make_predictions(self, model: Pipeline, X: pandas.DataFrame) -> bool:
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
            probabilities = np.array([[-1.0]*len(model.classes_)]*X.shape[0])
            rates = np.array([-1.0]*X.shape[0])

        self.could_predict_proba = could_predict_proba
        self.probabilites = probabilities
        self.predictions = predictions
        self.rates = rates

        return could_predict_proba

    # Gets the rate type
    # TODO: Maybe enum? I = "", U = "Unknown", A =""
    def get_rate_type(self) -> str:
        if self.could_predict_proba:
            return "I"

        if not self.handler.config.should_train():
            return "U"
        
        return "A"

    # Calculate the probability if the machine could not
    def calculate_probability(self) -> None:
        if not self.handler.config.should_train():
            return

        prob = []
        for i in range(len(self.predictions)):
            try:
                prob = prob + [self.class_report[self.predictions[i]]['precision']]
            except KeyError as e:
                self.handler.logger.print_warning(f"probability collection failed for key {self.predictions[i]} with error {e}")
    
        self.handler.logger.print_info("Probabilities:", str(prob))
        self.probabilites = prob

        
    
    # Returns a list of mean and standard deviation
    def get_probabilities(self, as_string: bool = False) -> list:
        mean = np.mean(self.probabilites)
        std = np.std(self.probabilites)
        
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
        if not could_predict_proba:
            for i in range(len(the_model.classes_)):
                X_mispredicted.insert(0, "P(" + the_model.classes_[i] + ")", "N/A")
            n_limit = min(self.LIMIT_MISPREDICTED, X_mispredicted.shape[0])

            self.X_most_mispredicted = X_mispredicted.sample(n=n_limit)
            return
        
        Y_prob_max = np.amax(Y_prob, axis = 1)
        for i in reversed(range(Y_prob.shape[1])):
            X_mispredicted.insert(0, "P(" + the_model.classes_[i] + ")", Y_prob[:,i])
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
