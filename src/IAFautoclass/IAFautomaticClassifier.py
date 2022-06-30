#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# General code for automatic classification of texts and numbers in databases
#
# Implemented by Robert Granat, IAF, March - May 2021
# Updated by Robert Granat, August 2021 - May 2022.
#
# Major revisions:
# 
# * Jan 26 - Febr 27, 2022: Rewritten the code as a Python Class
#
# Standard imports
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Special imports that might need installations below
# Make sure pip is the latest version for the current user
checked_pip = False

def check_pip_installation():
    print("Upgrading pip if necessary...")
    try:
        os.system('python.exe -m pip install -q --upgrade pip')
        return True
    except:
        return False

# Check that importlib is available
try:
    try:
        import importlib
    except:
        checked_pip = check_pip_installation()
        os.system('pip install importlib')
        import importlib
  
    # Install special libs that might cause problems for a new user
    packages = ['pyodbc','wheel','scipy','numpy','pickle','pandas','sklearn', \
        'matplotlib','stop_words', 'langdetect', 'lexicalrichness', 'textblob', \
        'anytree', 'imblearn', 'ipywidgets']

    for package in packages: 
        try:
            importlib.import_module(package)
        except:
            print("Please wait...installing missing package: {0}".format(package))
            if not checked_pip:
                checked_pip = check_pip_installation()
            os.system("pip install " + package)
            try:
                importlib.import_module(package)
            except:
                sys.exit("Please restart script to enable import of added modules!")
                            
except Exception as e:
    sys.exit("Aborting. Package installation failed: {0}".format(str(ex)))
    

# General imports
import base64
import importlib
import pickle
import time
import warnings
from datetime import datetime, timedelta
from math import ceil
#import ipywidgets as widgets
from pathlib import Path

import langdetect
import numpy as np
import pandas
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lexicalrichness import LexicalRichness
from pandas import concat
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import RFE
from sklearn.kernel_approximation import Nystroem
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, make_scorer, matthews_corrcoef)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.random_projection import GaussianRandomProjection
from stop_words import get_stop_words

# Imports of local help class for communication with SQL Server
import Config
import DataLayer
import IAFLogger

# Sklearn issue a lot of warnings sometimes, we suppress them here
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

class IAFautomaticClassiphyer:

    # Internal constants
    STANDARD_LANG = "sv"
    PCA_VARIANCE_EXPLAINED = 0.999
    LOWER_LIMIT_REDUCTION = 100
    NON_LINEAR_REDUCTION_COMPONENTS = 2
    LIMIT_MISPREDICTED = 20
    MAX_HEAD_COLUMNS = 10
    
    # Constructor with arguments
    def __init__(self, config: Config.Config, logger: IAFLogger.IAFLogger, save_config_to_file: bool = False):
        self.config = config

        self.logger = logger

        self.numMajorTasks = 12
        self.percentPermajorTask = 0.03
        self.progress = 0.0
        self.logger.print_progress(message="Starting up ...", percent=self.progress)
        
        self.majorTask = 0
        self.use_progress_bar = True
        
         # Init some internal variables
        self.scriptpath = os.path.dirname(os.path.realpath(__file__))
        self.scriptname = sys.argv[0][2:]

        self.datalayer = DataLayer.DataLayer(config.connection, self.scriptpath, config.io.verbose)

        # Setting path for misplaced data output file
        if self.config.mode.mispredicted:
            self.misplaced_filepath = self.get_output_filename("misplaced")
        else:
            self.misplaced_filepath = None
            
        # Prepare variables for unique class lables and misplaced data elements
        self.unique_classes = None
        self.X_most_mispredicted = None
        
        # Extract parameters that are not textual
        self.text_data = self.config.connection.data_text_columns != ""
        self.force_categorization = self.config.mode.category_text_columns != ""
        self.numerical_data = self.config.connection.data_numerical_columns != ""
        self.use_feature_selection = self.config.mode.feature_selection != Config.Reduction.NON
        
         # Set the name and path of the model file depending on training or predictions
        self.model_path = Path(self.scriptpath) / self.config.io.model_path
        if self.config.mode.train:
            self.model_filename = self.model_path / (self.config.name + ".sav")
        else:
            self.model_filename = self.model_path / self.config.io.model_name
        
        # Redirect standard output to debug files
        if self.config.debug.on and self.config.io.redirect_output:
            output_file = self.get_output_filename("output")
            error_file = self.get_output_filename("error")

            self.logger.print_info(f"Redirecting standard output to {output_file} and standard error to {error_file}")
            try:
                sys.stdout = open(output_file,'w')
                sys.stderr = open(error_file,'w')
            except Exception as e:
                self.abort_cleanly(f"Something went wrong with redirection of standard output and error: {str(e)}")
        
        # Write configuration to file for easy start from command line
        if save_config_to_file:
            self.config.save_to_file()
    
        # Internal settings for panda
        pandas.set_option("max_columns", self.MAX_HEAD_COLUMNS)
        pandas.set_option("display.width", 80)

        # Get some timestamps
        self.date_now = datetime.now()
        self.clock1 = time.time()
        
    # Destructor
    def __del__(self):
        pass
    
    # Print the class 
    def __str__(self):
        return str(type(self))
    
    # Functions below

    # Function to simplify the paths/names of output files
    def get_output_filename(self, type: str) -> str:
        types = {
            "output": {
                "suffix": "txt",
                "prefix": "output_"
            },
            "error": {
                "suffix": "txt",
                "prefix": "error_"
            },
            "misplaced": {
                "suffix": "csv",
                "prefix": "misplaced_"
            }
        }

        type_dict = types[type]

        output_path = self.scriptpath + "\\output\\"
        output_name = self.config.name + "_" + self.config.connection.data_username

        return output_path + type_dict["prefix"] + output_name + "." + type_dict["suffix"]

    
    # Function for reading in data to classify from database
    def read_in_data(self):

        try:
            data, query, num_lines = self.datalayer.get_data(self.config.debug.num_rows, self.config.debug.on, self.config.io.redirect_output, self.config.mode.train, self.config.mode.predict)

            if data is None:
                return pandas.DataFrame(), None, None, None


            # Set the column names of the data array
            column_names = [self.config.connection.id_column] + [self.config.connection.class_column] + \
                    self.config.connection.data_text_columns.split(',') + \
                    self.config.connection.data_numerical_columns.split(',')
            try:
                column_names.remove("") # Remove any empty column name
            except Exception as e:
                pass
            dataset = pandas.DataFrame(data, columns = column_names)
            
            # Make sure the class column is a categorical variable by setting it as string
            try:
                dataset.astype({self.config.connection.class_column: 'str'}, copy=False)
            except Exception as e:
                self.logger.print_warning(f"Could not convert class column {self.config.connection.class_column} to string variable: {str(e)}")
                
            # Extract unique class labels from dataset
            unique_classes = list(set(dataset[self.config.connection.class_column].tolist()))

            #TODO: Clean up the clean-up
            # Make an extensive search through the data for any inconsistencies (like NaNs and NoneType). 
            # Also convert datetime numerical variables to ordinals, i.e., convert them to the number of days 
            # or similar from a certain starting point, if any are left after the conversion above.
            self.logger.print_formatted_info(message="Consistency check")
            change = False
            percent_checked = 0
            try:
                for index in dataset.index:
                    old_percent_checked = percent_checked
                    percent_checked = round(100.0*float(index)/float(len(dataset.index)))
                    # TODO: \r is to have the Data checked updated, or some-such. Can it be moved?
                    if self.config.io.verbose and not self.config.io.redirect_output and percent_checked > old_percent_checked:
                        print("Data checked of fetched: " + str(percent_checked) + " %", end='\r')
                    for key in dataset.columns:
                        item = dataset.at[index,key]

                        # Set NoneType objects  as zero or empty strings
                        if (key in self.config.connection.data_numerical_columns.split(",") or \
                            key in self.config.connection.data_text_columns.split(",")) and item == None:
                            if key in self.config.connection.data_numerical_columns.split(","):
                                item = 0
                            else:
                                item = ""
                            change = True

                        # Convert numerical datetime values to ordinals
                        elif key in self.config.connection.data_numerical_columns.split(",") and self.is_datetime(item):
                            item = datetime.toordinal(item)
                            change = True

                        # Set remaining numerical values that cannot be casted as integer or floating point numbers to zero, i.e., do not
                        # take then into account
                        elif key in self.config.connection.data_numerical_columns.split(",") and \
                            not (self.is_int(item) or self.is_float(item)):
                            item = 0
                            change = True

                        # Set text values that cannot be casted as strings to empty strings
                        elif key in self.config.connection.data_text_columns.split(",") and type(item) != str and not self.is_str(item):
                            item = ""
                            change = True

                        # Remove line breaks from text strings
                        if key in self.config.connection.data_text_columns.split(","):
                            item = item.replace('\n'," ").replace('\r'," ").strip()
                            change = True

                        # Save new value
                        if change:
                            dataset.at[index,key] = item
                            change = False
            except Exception as e:
                # TODO: Maybe set it's own error? This is caught by parent
                raise ValueError("Aborting! Reason: {0}. Something went wrong in inconsistency check at {1}: {2}.".format(str(e),key,item))

            # Shuffle the upper part of the data, such that all already classified material are
            # put in random order keeping the unclassified material in the bottom
            # Notice: perhaps this operation is now obsolete, since we now use a NEWID() in the 
            # data query above
            self.logger.print_formatted_info("Shuffle data")
            num_un_pred = self.get_num_unpredicted_rows(dataset)
            dataset['rnd'] = np.concatenate([np.random.rand(num_lines - num_un_pred), [num_lines]*num_un_pred])
            dataset.sort_values(by = 'rnd', inplace = True )
            dataset.drop(['rnd'], axis = 1, inplace = True )

            # Use the unique id column from the data as the index column and take a copy, 
            # since it will not be used in the classification but only to reconnect each 
            # data row with classification results later on
            keys = dataset[self.config.connection.id_column].copy(deep = True).apply(get_rid_of_decimals)
            try:
                dataset.set_index(keys.astype('int64'), drop=False, append=False, inplace=True, verify_integrity=False)
            except Exception as e:
                self.abort_cleanly(f"Could not set index for dataset: {str(e)}")
            
            dataset = dataset.drop([self.config.connection.id_column], axis = 1)

        except Exception as e:
            self.abort_cleanly(f"Load of dataset failed: {str(e)}")

        # Return both data and keys
        return dataset, keys, query, unique_classes

    # TODO: used in GUI
    # Get a list of pretrained models
    def get_trained_models(self):
        models = []
        for file in os.listdir(self.model_path):
            if file[-len(self.DEFAULT_MODEL_FILE_EXTENSION):] == self.DEFAULT_MODEL_FILE_EXTENSION:
                models.append(file)
        return models
                
    # Help routines for determining consistency of input data
    @staticmethod
    def is_float(val):
        try:
            num = float(val)
        except ValueError:
            return False
        return True
    
    @staticmethod
    def is_int(val):
        try:
            num = int(val)
        except ValueError:
            return False
        return True
    
    @staticmethod
    def is_bool(val):
        return type(val)==bool

    @staticmethod
    def is_str(val):
        is_other_type = \
            IAFautomaticClassiphyer.is_float(val) or \
            IAFautomaticClassiphyer.is_int(val) or \
            IAFautomaticClassiphyer.is_bool(val) or \
            IAFautomaticClassiphyer.is_datetime(val)
        return not is_other_type 
    
    @staticmethod
    def is_datetime(val):
        try:
            if isinstance(val,datetime): #Simple, is already instance of datetime
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

    # Calculate the number of unclassified rows in data matrix
    def get_num_unpredicted_rows(self, dataset):

        num = 0
        for item in dataset[self.config.connection.class_column]:
            if item == None or not str(item).strip(): # Corresponds to empty string and SQL NULL
                num += 1
        return num   

    # Find out if a DataFrame column contains categorical data or not
    def is_categorical_data(self, column):

        is_categorical = column.value_counts().count() <= self.config.LIMIT_IS_CATEGORICAL \
            or (self.force_categorization and column.name in self.config.mode.category_text_columns.split(","))

        return is_categorical

    # Collapse all data text columns into a new column, which is necessary
    # for word-in-a-bag-technique
    def convert_textdata_to_numbers(self, dataset, label_binarizers = {}, count_vectorizer = None, \
        tfid_transformer = None):

        # Pick out the classification column. This is the 
        # "right hand side" of the problem.
        Y = dataset[self.config.connection.class_column]

        # Continue with "left hand side":
        # Prepare two empty DataFrames
        text_dataset = pandas.DataFrame()
        num_dataset = pandas.DataFrame()
        categorical_dataset = pandas.DataFrame()
        binarized_dataset = pandas.DataFrame()

        # First, separate all the text data from the numerical data, and
        # make sure to find the categorical data automatically
        if self.text_data:
            text_columns = self.config.connection.data_text_columns.split(',')
            for column in text_columns:
                if (self.config.mode.train and self.config.mode.use_categorization and  \
                    self.is_categorical_data(dataset[column])) or column in label_binarizers.keys():
                    categorical_dataset = concat([categorical_dataset, dataset[column]], axis = 1)
                    self.logger.print_info("Text data picked for categorization: ", column)
                else:
                    text_dataset = concat([text_dataset, dataset[column]], axis = 1)
                    self.logger.print_info("Text data NOT picked for categorization: ", column)
                 
        if self.numerical_data:
            num_columns = self.config.connection.data_numerical_columns.split(',')
            for column in num_columns:
                num_dataset = concat([num_dataset, dataset[column]], axis = 1)

        # For concatenation, we need to make sure all text data are 
        # really treated as text, and categorical data as categories
        if self.text_data:
            self.logger.print_info("Text Columns:", text_dataset.columns)
            if len(text_dataset.columns) > 0:

                text_dataset = text_dataset.applymap(str)

                # Concatenating text data such that the result is another DataFrame  
                # with a single column
                text_dataset = text_dataset.agg(' '.join, axis = 1)

                # Convert text data to numbers using word-in-a-bag technique
                text_dataset, count_vectorizer, tfid_transformer = \
                    self.word_in_a_bag_conversion(text_dataset, count_vectorizer, tfid_transformer)
                text_dataset = pandas.DataFrame(text_dataset)

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
                            self.logger.print_info(f"Column {column} was categorized with categories: {lb.classes_}")
                        else:
                            lb_results_df = pandas.DataFrame(lb_results, columns=[column])
                            self.logger.print_info(f"Column {column} was binarized")
                    except ValueError as e:
                        self.logger.print_warning(f"Column {column} could not be binarized: {str(e)}")
                    binarized_dataset = concat([binarized_dataset, lb_results_df], axis = 1 )

        # Set together all the data before returning
        X = pandas.DataFrame()
        if self.text_data and text_dataset.shape[1] > 0:
            text_dataset.set_index(dataset.index, drop=False, append=False, inplace=True, \
                                   verify_integrity=False)
            X = concat([text_dataset, X], axis = 1)
        if self.numerical_data and num_dataset.shape[1] > 0:
            num_dataset.set_index(dataset.index, drop=False, append=False, inplace=True, \
                                  verify_integrity=False)
            X = concat([num_dataset, X], axis = 1)
        if self.text_data and binarized_dataset.shape[1] > 0:
            binarized_dataset.set_index(dataset.index, drop=False, append=False, inplace=True, \
                                        verify_integrity=False)
            X = concat([binarized_dataset, X], axis = 1)
            
        return X, Y, label_binarizers, count_vectorizer, tfid_transformer

    # Use the bag of words technique to convert text corpus into numbers
    def word_in_a_bag_conversion(self, dataset, count_vectorizer = None, tfid_transformer = None):

        # Start working with datavalues in array
        X = dataset.values

        # Find actual languange if stop words are used
        if count_vectorizer == None:
            my_language = None
            if self.config.mode.use_stop_words:
                try:
                    my_language = langdetect.detect(' '.join(X))
                except Exception as e:
                    my_language = self.STANDARD_LANG
                    self.logger.print_warning(f"Language could not be detected automatically: {str(e)}. Fallback option, use: {my_language}.")
                else:
                    self.logger.print_info(f"Detected language is: {my_language}")

            # Calculate the lexical richness
            try:
                lex = LexicalRichness(' '.join(X)) 
                self.logger.print_info("#Words, #Terms and TTR for original text is {0}, {1}, {2:5.2f} %"
                      .format(lex.words,lex.terms,100*float(lex.ttr)))
            except Exception as e:
                self.logger.print_warning(f"Could not calculate lexical richness: {str(e)}")

        # Mask all material by encryption (optional)
        if (self.config.mode.hex_encode):
            X = self.do_hex_base64_encode_on_data(X)

        # Text must be turned into numerical feature vectors ("bag-of-words"-technique).
        # If selected, remove stop words
        if count_vectorizer == None:
            my_stop_words = None
            if self.config.mode.use_stop_words:

                # Get the languange specific stop words and encrypt them if necessary
                my_stop_words = get_stop_words(my_language)
                self.config.print_info("Using standard stop words: ", my_stop_words)
                if (self.config.mode.hex_encode):
                    for word in my_stop_words:
                        word = self.cipher_encode_string(str(word))

                # Collect text specific stop words (already encrypted if encryption is on)
                text_specific_stop_words = []
                if self.config.mode.specific_stop_words_threshold < 1.0:
                    try:
                        stop_vectorizer = CountVectorizer(min_df = self.config.mode.specific_stop_words_threshold)
                        stop_vectorizer.fit_transform(X)
                        text_specific_stop_words = stop_vectorizer.get_feature_names()
                        self.logger.print_info("Using specific stop words: ", text_specific_stop_words)
                    except ValueError as e:
                        self.logger.print_warning(f"Specified stop words threshold at {self.config.mode.specific_stop_words_threshold} generated no stop words.")
                    
                    my_stop_words = sorted(set(my_stop_words + text_specific_stop_words))
                    self.logger.print_info("Total list of stop words:", my_stop_words)

            # Use the stop words and count the words in the matrix        
            count_vectorizer = CountVectorizer(stop_words = my_stop_words)
            count_vectorizer.fit(X)

        # Do the word in a bag now
        X = count_vectorizer.transform(X)

        # Also generate frequencies instead of occurences to normalize the information.
        if tfid_transformer == None:
            tfid_transformer = TfidfTransformer(use_idf = False)
            tfid_transformer.fit(X) 

        # Generate the sequences
        X = (tfid_transformer.transform(X)).toarray()

        return X, count_vectorizer, tfid_transformer

    # Convert dataset to unreadable hex code
    @staticmethod
    def do_hex_base64_encode_on_data(X):

        XX = X.copy()

        with np.nditer(XX, op_flags=['readwrite'], flags=["refs_ok"]) as iterator:
            for x in iterator:
                xval = str(x)
                xhex = IAFautomaticClassiphyer.cipher_encode_string(xval)
                x[...] = xhex 

        return XX

    @staticmethod 
    def cipher_encode_string(a):

        aa = a.split()
        b = ""
        for i in range(len(aa)):
            b += (str(
                    base64.b64encode(
                        bytes(aa[i].encode("utf-8").hex(),'utf-8')
                    )
                ) + " ")

        return b.strip()

    # Feature selection (reduction) function for PCA or Nystroem transformation of data.
    # (RFE feature selection is built into the model while training and does not need to be
    # considered here.)
    def perform_feature_selection(self, X, number_of_components = None, feature_selection_transform = None ):
        components = None
        if number_of_components == "None": # Convert from string to Nonetype Object
            number_of_components = None

        # In the case of training, we compute a new transformation
        if feature_selection_transform == None:
            
            # Keep track it the selected transform fails or not
            failed = False

            # PCA first. If not the number of features is specified by
            # the user, we rely on  Minka’s MLE to guess the optimal dimension.
            # Notice, MLE can only be used if the number of samples exceeds the number
            # of features, which is common, but not always. If MLE cannot be used or
            # fails, pick the number of components that explain a certain level of variance.
            
            if self.config.mode.feature_selection == Config.Reduction.PCA:
                self.logger.print_progress(message="PCA conversion of dataset under way...")
                
                components_options = []
                if number_of_components != None and number_of_components > 0:
                    components_options.append(number_of_components)
                    components = number_of_components
                if X.shape[0] >= X.shape[1]:
                    components_options.append('mle')
                    components = 'mle'
                components_options.append(self.PCA_VARIANCE_EXPLAINED)
                components_options.append(min(X.shape[0], X.shape[1]) - 1)
                
                # Make transformation
                for components in components_options:
                    self.logger.print_components("PCA", components)
                    try:
                        failed = True
                        feature_selection_transform = PCA(n_components=components)
                        feature_selection_transform.fit(X)
                        X = feature_selection_transform.transform(X)
                        failed = False
                    except Exception as e:
                        self.logger.print_components("PCA", components, str(e))
                    else:
                        self.logger.print_info("...new shape of data matrix is: ({X.shape[0]},{X.shape[1]})")
                        components = feature_selection_transform.n_components_
                        break

            # Nystroem transformation next
            elif self.config.mode.feature_selection == Config.Reduction.NYS:
                self.logger.print_progress(message="Nystroem conversion of dataset under way...")
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                else:
                    components = max(self.LOWER_LIMIT_REDUCTION, min(X.shape))
                    self.logger.print_components("Nystroem", components)
                # Make transformation
                try:
                    feature_selection_transform = Nystroem(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as e:
                    self.logger.print_components("Nystroem transform", components, str(e))
                    failed = True
                else:
                    self.logger.print_info(f"...new shape of data matrix is: ({X.shape[0]},{X.shape[1]})")
                    components = feature_selection_transform.n_components_
                    
            # Truncated SVD transformation next
            elif self.config.mode.feature_selection == Config.Reduction.TSVD:
                self.logger.print_progress(message="Truncated SVD reduction of dataset under way...")
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                else:
                    components = max(self.LOWER_LIMIT_REDUCTION, min(X.shape))
                    self.logger.print_components("Truncated SVD", components)
                # Make transformation
                try:
                    feature_selection_transform = TruncatedSVD(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as e:
                    self.logger.print_components("Truncated SVD reduction", components, str(e))
                    failed = True
                else:
                    self.logger.print_info(f"...new shape of data matrix is: ({X.shape[0]},{X.shape[1]})")
                    components = feature_selection_transform.n_components_

            # Fast independent component transformation next
            elif self.config.mode.feature_selection == Config.Reduction.FICA:
                self.logger.print_progress(message="Fast ICA reduction of dataset under way...")
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                else:
                    components = max(self.LOWER_LIMIT_REDUCTION, min(X.shape))
                    self.logger.print_components("Fast ICA", components)
                    
                # Make transformation
                try:
                    feature_selection_transform = FastICA(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as e:
                    self.logger.print_components("Fast ICA reduction", components, str(e))
                    failed = True
                else:
                    self.logger.print_info(f"...new shape of data matrix is: ({X.shape[0]},{X.shape[1]})")
                    components = feature_selection_transform.n_components_
                    
            # Random Gaussian Projection
            elif self.config.mode.feature_selection == Config.Reduction.GRP:
                self.logger.print_progress(message="Gaussian Random Projection reduction of dataset under way...")
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                else:
                    components = 'auto'
                    self.logger.print_components("GRP", components)
                # Make transformation
                try:
                    feature_selection_transform = GaussianRandomProjection(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as e:
                    self.logger.print_components("GRP reduction", components, str(e))
                    failed = True
                else:
                    self.logger.print_info(f"...new shape of data matrix is: ({X.shape[0]},{X.shape[1]})") 
                    components = feature_selection_transform.n_components_
            
            # Non Linear Transformations below
            # First: Isomap
            elif self.config.mode.feature_selection == Config.Reduction.ISO:
                self.logger.print_progress(message="Isometric Mapping reduction of dataset under way...")
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                else:
                    components = self.NON_LINEAR_REDUCTION_COMPONENTS
                    self.logger.print_components("ISO", components)
                # Make transformation
                try:
                    feature_selection_transform = Isomap(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as e:
                    self.logger.print_components("Isomap reduction", components, str(e))
                    failed = True
                else:
                    self.logger.print_info(f"...new shape of data matrix is: ({X.shape[0]},{X.shape[1]})") 
                
            # Second: LLE
            elif self.config.mode.feature_selection == Config.Reduction.LLE:
                self.logger.print_progress("Locally Linear Embedding reduction of dataset under way...")
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                else:
                    components = self.NON_LINEAR_REDUCTION_COMPONENTS
                    self.logger.print_components("LLE", components)
                # Make transformation
                try:
                    feature_selection_transform = LocallyLinearEmbedding(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as e:
                    self.logger.print_components("LLE reduction", components, str(e))
                    failed = True
                else:
                    self.logger.print_info(f"...new shape of data matrix is: ({X.shape[0]},{X.shape[1]})") 
                
            # If the wanted transform failed, mark it here
            if failed:
                feature_selection_transform = None
                components = X.shape[1]
                
        # For only predictions, use the saved transform associated with trained model
        else:
            X = feature_selection_transform.transform(X)

        return X, components, feature_selection_transform

    # Split dataset into training and validation parts
    def split_dataset(self, X, Y, num_lower):

        # First, split X and Y in two parts: the upper part, to be used in training,
        # and the lower part, to be classified later on
        [X_upper, X_lower] = np.split(X, [X.shape[0]-num_lower], axis = 0)
        [Y_upper, Y_lower] = np.split(Y, [Y.shape[0]-num_lower], axis = 0)

        # Split-out validation dataset from the upper part (do not use random order here)
        testsize = float(self.config.mode.test_size)
        X_train, X_validation, Y_train, Y_validation = \
                 train_test_split( X_upper, Y_upper, test_size = testsize, \
                 shuffle = False, random_state = None, stratify = None )

        # Return all parts of the data, including the unclassified
        return X_train, X_validation, X_lower, Y_train, Y_validation, Y_lower
    
    # Spot Check Algorithms.
    # We do an extensive search of the best algorithm in comparison with the best
    # preprocessing.
    def spot_check_ml_algorithms(self, X_train, Y_train, k=10):

        # Save standard progress text
        standardProgressText = "Check and train algorithms for best model"

        # Add all algorithms in a list
        self.logger.print_info("Spot check ml algorithms...")
        models = []
        for algo in Config.Algorithm:
            if algo.has_algorithm_function():
                models.append((algo, algo.call_algorithm(max_iterations=self.config.mode.max_iterations, size=X_train.shape[0])))
        

        # Add different preprocessing methods in another list
        preprocessors = []
        for preprocessor in Config.Preprocess:
            # This checks that the preprocessor has the function, and also that BIN is only added if self.text_data
            if preprocessor.has_preprocess_function() and (preprocessor.name != "BIN" or self.text_data):
                preprocessors.append((preprocessor, preprocessor.call_preprocess()))
                
        
        preprocessors.append((Config.Preprocess.NON, None)) # In case they choose no preprocessor in config
        

        # Evaluate each model in turn in combination with all preprocessing methods
        names = []
        best_mean = 0.0
        best_std = 1.0
        trained_model = None
        temp_model = None
        trained_model_name = None
        scorer_mechanism = None
        best_feature_selection = X_train.shape[1]
        first_round = True
        if self.config.io.verbose: # TODO: Can this be printed nicer?
                print("{0:>4s}-{1:<6s}{2:>6s}{3:>8s}{4:>8s}{5:>11s}".format("Name","Prep.","#Feat.","Mean","Std","Time"))
                print("="*45)
        numMinorTasks = len(models) * len(preprocessors)
        percentAddPerMinorTask = (1.0-self.percentPermajorTask*self.numMajorTasks) / float(numMinorTasks)

        # Loop over the models
        for name, model in models:
            # If the user has specified a specific algorithm, restrict computations accordingly
            if name != self.config.mode.algorithm and not self.config.mode.algorithm == Config.Algorithm.ALL:
                continue
            
            # Loop over pre-processing methods
            for preprocessor_name, preprocessor in preprocessors:
                # Update progressbar percent and label
                self.logger.print_progress(message=f"{standardProgressText} ({name.name}-{preprocessor_name.name})")
                if not first_round: 
                    self.update_progress(percent=percentAddPerMinorTask)
                else:
                    first_round = False


                # If the user has specified a specific preprocessor, restrict computations accordingly            
                if preprocessor_name != self.config.mode.preprocessor and not self.config.mode.preprocessor == Config.Preprocess.ALL:
                    continue

                # Add feature selection if selected, i.e., the option of reducing the number of variables used.
                # Make a binary search for the optimal dimensions.
                max_features_selection = X_train.shape[1]

                # Make sure all numbers are propely set for feature selection interval
                if self.use_feature_selection and self.config.mode.num_selected_features in ["", None]:
                    min_features_selection = 0
                elif self.use_feature_selection and self.config.mode.num_selected_features > 0:
                    min_features_selection = self.config.mode.num_selected_features
                    max_features_selection = self.config.mode.num_selected_features
                else:
                    min_features_selection = max_features_selection # No or minimal number of features are eliminated

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
                    if self.use_feature_selection and self.config.mode.feature_selection == Config.Reduction.RFE:     
                        try:
                            rfe = RFE(temp_model, n_features_to_select=num_features)
                            temp_model = rfe.fit(X_train, Y_train)
                        except ValueError as ex:
                            break

                    # Both algorithm and preprocessor should be used. Move on.
                    # Build pipline of model and preprocessor.
                    names.append((name,preprocessor_name))
                    if not self.config.mode.smote and not self.config.mode.undersample:
                        if preprocessor != None:
                            pipe = make_pipeline(preprocessor, temp_model)
                        else:
                            pipe = temp_model

                    # For use SMOTE and undersampling, different Pipeline is used
                    else:
                        steps = None
                        smote = None
                        undersampler = None
                        if self.config.mode.smote:
                            smote = SMOTE(sampling_strategy='auto')
                        if self.config.mode.undersample:
                            undersampler = RandomUnderSampler(sampling_strategy='auto')
                        steps = [('smote', smote ), ('under', undersampler), \
                                ('preprocessor', preprocessor), ('model', temp_model)]
                        pipe = ImbPipeline(steps=steps)

                    # Make evaluation of model
                    try:
                        kfold = StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
                    except Exception as e:
                        self.abort_cleanly(message=f"StratifiedKfold raised an exception with message: {str(e)}")

                    # Now make kfolded cross evaluation
                    cv_results = None
                    try:
                        if scorer_mechanism == None:
                            if self.config.mode.scoring == Config.Scoretype.mcc:
                                scorer_mechanism = make_scorer(matthews_corrcoef)
                            else:
                                scorer_mechanism = self.config.mode.scoring.name
                        cv_results = cross_val_score(pipe, X_train, Y_train, cv=kfold, scoring=scorer_mechanism) 
                    except Exception as e:
                        self.logger.print_warning(f"NOTICE: Model {names[-1]} raised an exception in cross_val_score with message: {str(e)}. Skipping to next")
                    else:
                        # Stop the stopwatch
                        t = time.time() - t0

                        # For current settings, calculate score
                        temp_score = cv_results.mean()
                        temp_stdev = cv_results.std()

                        # Print results to screen
                        if self.config.io.verbose: # TODO: print prettier
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
                            trained_model_name = (name, preprocessor_name)
                            best_mean = temp_score
                            best_std = temp_stdev
                            best_feature_selection = num_features

        # Return best model for start making predictions
        return trained_model_name, trained_model, best_feature_selection

    # Train ml model
    def train_picked_model(self, model, X, Y):

        # Train model
        try:
            model.fit(X, Y)
        except Exception as e:
            sys.exit("Something went wrong on training picked model {0}: {1}".format(str(model),str(ex))) #warning

        return model

    # Save ml model and corresponding configuration
    def save_model_to_file(self, label_binarizers, count_vectorizer, tfid_transformer, \
        feature_selection_transform, model_name, model, filename):
        
        try:
            save_config = self.config.get_clean_config()
            data = [save_config, label_binarizers, count_vectorizer, tfid_transformer, \
                feature_selection_transform, model_name, model]
            pickle.dump(data, open(filename,'wb'))

        except Exception as e:
            self.logger.print_warning(f"Something went wrong on saving model to file: {str(e)}")
    
    # Load ml model with corresponding configuration
    def load_model_from_file(self, filename):
        try:
            saved_config, label_binarizers, count_vectorizer, tfid_transformer, \
                feature_selection_transform, model_name, model = pickle.load(open(filename, 'rb'))
            saved_config.mode.train = self.config.mode.train
            saved_config.mode.predict = self.config.mode.predict
            saved_config.mode.mispredicted = self.config.mode.mispredicted
            saved_config.connection.data_catalog = self.config.connection.data_catalog
            saved_config.connection.data_table = self.config.connection.data_table
            saved_config.io.model_name = self.config.io.model_name
            saved_config.debug.num_rows = self.config.debug.num_rows
            self.config = saved_config
        except Exception as e:
            self.logger.print_warning(f"Something went wrong on loading model from file: {str(e)}")
        
        return label_binarizers, count_vectorizer, tfid_transformer, feature_selection_transform, \
            model_name, model
    
    # Make predictions on dataset
    def make_predictions(self, model, X):

        could_predict_proba = False
        try:
            predictions = model.predict(X)
        except ValueError as e:
            self.abort_cleanly(message=f"It seems like you need to regenerate your prediction model: {str(e)}")
        try:
            probabilities = model.predict_proba(X)
            rates = np.amax(probabilities, axis=1)
            could_predict_proba = True
        except Exception as e:
            self.logger.print_warning(f"Probablity prediction not available for current model: {str(e)}")
            probabilities = np.array([[-1.0]*len(model.classes_)]*X.shape[0])
            rates = np.array([-1.0]*X.shape[0])
        return predictions, could_predict_proba, rates, probabilities

    # Evaluate predictions
    def evaluate_predictions(self, predictions, Y, message="Unknown"):
        self.logger.print_info(f"Evaluation performed with evaluation data: " + message)
        self.logger.print_info(f"Accuracy score for evaluation data: {accuracy_score(Y, predictions)}")
        self.logger.print_info(f"Confusion matrix for evaluation data: \n\n{confusion_matrix(Y, predictions)}")
        self.logger.print_info(f"Classification matrix for evaluation data: \n\n{classification_report(Y, predictions, zero_division='warn')}")

    # Function for finding the n most mispredicted data rows
    def most_mispredicted(self, X_original, model, ct_model, X_transformed, Y, n_limit):

        # Calculate predictions for both total model and cross trained model
        for what_model, the_model in [("model retrained on all data",model), ("modell cross trained on training data",ct_model)]:
            Y_pred = pandas.DataFrame(the_model.predict(X_transformed), index = Y.index)

            # Find the data rows where the real category is different from the predictions
            # Iterate over the indexes (they are now not in order)
            X_not = []
            for i in Y.index:
                try:
                    X_not.append((Y.loc[i] != Y_pred.loc[i]).bool())
                except Exception as e:
                    self.logger.print_warning(f"Append of data row with index: {i} failed: {str(e)}. Probably duplicate indicies in data!")
                    break

            # Quick end of loop if possible
            num_mispredicted = sum(elem == True for elem in X_not)
            if num_mispredicted != 0:
                break
 
        # Quick return if possible
        if num_mispredicted == 0:
            return "no model produced mispredictions", num_mispredicted, pandas.DataFrame()
        
        #
        self.logger.print_info(f"Accuracy score for {what_model}: {accuracy_score(Y, Y_pred)}")

        # Select the found mispredicted data
        X_mispredicted = X_transformed.loc[X_not]

        # Predict probabilites
        try:
            Y_prob = the_model.predict_proba(X_mispredicted)
            could_predict_proba = True
        except Exception as e:
            self.logger.print_warning(f"Could not predict probabilities: {str(e)}")
            could_predict_proba = False

        #  Re-insert original data columns but drop the class column
        X_mispredicted = X_original.loc[X_not]
        X_mispredicted = X_mispredicted.drop(self.config.connection.class_column, axis = 1)

        # Add other columns to mispredicted data
        X_mispredicted.insert(0, "Actual", Y.loc[X_not])
        X_mispredicted.insert(0, "Predicted", Y_pred.loc[X_not].values)
        
        # Add probabilities and sort only if they could be calculated above, otherwise
        # return a random sample of mispredicted
        if not could_predict_proba:
            for i in range(len(the_model.classes_)):
                X_mispredicted.insert(0, "P(" + the_model.classes_[i] + ")", "N/A")
            n_limit = min(n_limit, X_mispredicted.shape[0])
            return what_model, num_mispredicted, X_mispredicted.sample(n=n_limit)
        else:
            Y_prob_max = np.amax(Y_prob, axis = 1)
            for i in reversed(range(Y_prob.shape[1])):
                X_mispredicted.insert(0, "P(" + the_model.classes_[i] + ")", Y_prob[:,i])
            X_mispredicted.insert(0, "__Sort__", Y_prob_max)

            # Sort the dataframe on the first column and remove it
            X_mispredicted = X_mispredicted.sort_values("__Sort__", ascending = False)
            X_mispredicted = X_mispredicted.drop("__Sort__", axis = 1)

            # Keep only the top n_limit rows and return
            return what_model, num_mispredicted, X_mispredicted.head(n_limit)

    
    # Find the number of data rows belonging to to smallest class in an array
    @staticmethod
    def find_smallest_class_number(Y):

        class_count = {}
        for elem in Y:
            if elem not in class_count:
                class_count[elem] = 1
            else:
                class_count[elem] += 1
        return max(1, min(class_count.values()))

    def abort_cleanly(self, message: str) -> None:
        self.logger.print_exit_error(message)
        sys.exit("Program aborted.")

    def update_progress(self, percent: float, message: str = None) -> float:
        self.progress += percent

        if message is None:
            self.logger.print_progress(percent = self.progress)
        else:
            self.logger.print_progress(message=message, percent = self.progress)

        return self.progress

    # The main classification function for the class
    def run(self):
        
         # Print a welcoming message for the audience
        self.logger.print_welcoming_message(config=self.config, date_now=self.date_now)

        # Print out what mode we use: training a new model or not
        if self.config.mode.train:
            self.logger.print_formatted_info("We will train our ml model")
        else:
            if os.path.exists(self.model_filename):
                self.logger.print_formatted_info("We will reload and use old model")
            else:
                self.abort_cleanly(f"No trained model exists at {self.model_filename}")
        
        # Create the classification table, if it does not exist already
        self.logger.print_progress(message="Create the classification table")
        self.datalayer.create_classification_table()
        self.update_progress(self.percentPermajorTask)

        try:
            # Set a flag in the classification database that execution has started
            self.logger.print_progress(message="Marking execution started in database...-please wait!")

            self.datalayer.mark_execution_started()
        except Exception as e:
            self.abort_cleanly(f"Mark of executionstart failed: {str(e)}")


        # Read in all data, with classifications or not
        self.logger.print_progress(message="Read in data from database")
        dataset, unique_keys, data_query, self.unique_classes = self.read_in_data()
        if dataset.empty:
            self.logger.print_progress(message="Process finished", percent=1.0)
            return -1
        
        self.update_progress(self.percentPermajorTask)

        # Investigate dataset (optional)
        if self.config.mode.train:
            self.logger.investigate_dataset(dataset) # Returns True if the investigation/printing was not suppressed

        # We need to know the number of data rows that are currently not
        # classified (assuming that they are last in the dataset because they
        # were sorted above). Separate keys from data elements that are
        # already classified from the rest.
        num_un_pred = self.get_num_unpredicted_rows(dataset)
        unpred_keys = unique_keys[-num_un_pred:]

        # Original training data are stored for later reference
        if num_un_pred > 0:
            X_original = dataset.head(-num_un_pred)
        else:
            X_original = dataset.copy(deep=True)

        # If we are making predictions, reload the necessary transforms and the model
        label_binarizers = {}
        count_vectorizer = None
        tfid_transformer = None
        feature_selection_transform = None
        if not self.config.mode.train:
            label_binarizers, count_vectorizer, tfid_transformer, feature_selection_transform, \
                trained_model_name, trained_model = self.load_model_from_file(self.model_filename)

        # Rearrange dataset such that all text columns are merged into one single
        # column, and convert text to numbers. Return the data in left-hand and
        # right hand side parts
        self.logger.print_progress(message="Rearrange dataset for possible textclassification, etc.")
        X, Y, label_binarizers, count_vectorizer, tfid_transformer = \
            self.convert_textdata_to_numbers(dataset, label_binarizers, count_vectorizer, tfid_transformer)

        if self.text_data:
            self.logger.print_info("After conversion of text data to numerical data:")
            self.logger.investigate_dataset( X, False, False )
        
        self.update_progress(self.percentPermajorTask)
        
        # In case of PCA or Nystreom or other feature reduction, here it goes
        if self.use_feature_selection and self.config.mode.feature_selection != Config.Reduction.RFE:
            t0 = time.time()
            X, self.config.mode.num_selected_features, feature_selection_transform = \
                self.perform_feature_selection( X, self.config.mode.num_selected_features, feature_selection_transform )

            t = time.time() - t0
            self.logger.print_info(f"Feature reduction took {str(round(t,2))}  sec.")

        self.update_progress(self.percentPermajorTask)

        # Split dataset for machine learning
        if self.config.mode.train:
            self.logger.print_progress(message="Split dataset for machine learning")
            X_train, X_validation, X_unknown, Y_train, Y_validation, Y_unknown = self.split_dataset(X, Y, num_un_pred)
        else:
            X_train = None
            X_validation = None
            Y_train = None
            Y_validation = None
            X_unknown = X
            Y_unknown = Y

        self.update_progress(self.percentPermajorTask)

        # Check algorithms for best model and train that model. K-value should be 10 or below.
        # Or just use the model previously trained.
        num_features = None
        if self.config.mode.train:
            self.logger.print_progress(message="Check and train algorithms for best model")
            k = min(10, self.find_smallest_class_number(Y_train))
            if k < 10:
                self.logger.print_info(f"Using non-standard k-value for spotcheck of algorithms: {k}")
            trained_model_name, trained_model, num_features = self.spot_check_ml_algorithms(X_train, Y_train, k)
            self.config.mode.algorithm = trained_model_name[0]
            self.config.mode.preprocessor = trained_model_name[1]
            self.config.mode.num_selected_features = num_features 
            trained_model = self.train_picked_model(trained_model, X_train, Y_train)
            self.save_model_to_file(label_binarizers, count_vectorizer, tfid_transformer, \
                feature_selection_transform, trained_model_name, trained_model, self.model_filename )
        elif self.config.mode.predict:
            num_features = trained_model.n_features_in_
        else:
            self.abort_cleanly("User must choose either to train a new model or use an old one for predictions")

        self.update_progress(percent=self.percentPermajorTask, message=f"Best model is: ({trained_model_name[0].value}/{trained_model_name[1].value}) with number of features: {num_features}")
        
        # Make predictions on known testdata
        if self.config.mode.train and X_validation.shape[0] > 0:
            self.logger.print_progress(message="Make predictions on known testdata")
            
            pred, could_proba, prob, probs = self.make_predictions(trained_model, X_validation)
            if could_proba:
               self.logger.print_info("Training Classification prob: Mean, Std.dev: ", str(np.mean(prob)), str(np.std(prob)))

        self.update_progress(percent=self.percentPermajorTask)
        
        
        # Evaluate predictions (optional)
        if self.config.mode.train and Y_validation.shape[0] > 0:
            self.logger.print_progress(message="Evaluate predictions")
            ml_algorithm = Config.get_model_name(trained_model_name[0], trained_model_name[1])
            self.evaluate_predictions(pred, Y_validation, "ML algorithm: " + ml_algorithm)
        

        # Get accumulated classification score report for all predictions
        if self.config.mode.train and Y_validation.shape[0] > 0:
            class_report = classification_report(Y_validation, pred, output_dict = True)
            self.logger.print_classification_report(class_report, trained_model_name, num_features)
            
        # RETRAIN best model on whole dataset: Xtrain + Xvalidation
        if self.config.mode.train and (X_train.shape[0] + X_validation.shape[0]) > 0:
            self.logger.print_progress(message="Retrain model on whole dataset")

            _label_binarizers, _count_vectorizer, _tfid_transformer, _feature_selection_transform, \
                _trained_model_name, cross_trained_model = self.load_model_from_file(self.model_filename)
            trained_model = \
                self.train_picked_model( trained_model, \
                    concat([pandas.DataFrame(X_train), pandas.DataFrame(X_validation)], axis = 0), \
                    concat([Y_train, Y_validation], axis = 0) )
            self.save_model_to_file(label_binarizers, count_vectorizer, tfid_transformer, \
                feature_selection_transform, trained_model_name, trained_model, self.model_filename )
            what_model, num_mispredicted, self.X_most_mispredicted = \
                self.most_mispredicted(X_original, trained_model, cross_trained_model, \
                concat([pandas.DataFrame(X_train), pandas.DataFrame(X_validation)], axis = 0), \
                concat([Y_train, Y_validation], axis = 0), \
                self.LIMIT_MISPREDICTED)
            
            self.logger.print_info(f"Total number of mispredicted elements: {num_mispredicted}")
            joiner = self.config.connection.id_column + " = \'"
            most_mispredicted_query = data_query + "WHERE " +  joiner \
                + ("\' OR " + joiner).join([str(number) for number in self.X_most_mispredicted.index.tolist()]) + "\'"
            if not self.X_most_mispredicted.empty and self.config.mode.mispredicted: 
                
                self.logger.print_formatted_info(f"Most mispredicted during training (using {what_model})")
                self.logger.print_info(str(self.X_most_mispredicted.head(self.LIMIT_MISPREDICTED)))
                self.logger.print_info(f"Get the most misplaced data by SQL query:\n {most_mispredicted_query}")
                self.logger.print_info(f"Or open the following csv-data file: \n\t {self.misplaced_filepath}")
                
                self.X_most_mispredicted.to_csv(path_or_buf = self.misplaced_filepath, sep = ';', na_rep='N/A', \
                                           float_format=None, columns=None, header=True, index=True, \
                                           index_label=self.config.connection.id_column, mode='w', encoding='utf-8', \
                                           compression='infer', quoting=None, quotechar='"', line_terminator=None, \
                                           chunksize=None, date_format=None, doublequote=True, decimal=',', \
                                           errors='strict')
        
        self.update_progress(percent=self.percentPermajorTask)

        # Now make predictions on non-classified dataset: X_unknown -> Y_unknown
        if self.config.mode.predict and X_unknown.shape[0] > 0:
            self.logger.print_progress(message="Make predictions on un-classified dataset")
            Y_unknown, could_proba, Y_prob, Y_probs = self.make_predictions(trained_model, X_unknown)
            self.logger.print_formatted_info("Predictions for the unknown data")
            self.logger.print_info("Predictions:", str(Y_unknown))
            if could_proba:
                self.logger.print_info("Classification Y_prob: Mean, Std.dev: ", str(np.mean(Y_prob)), str(np.std(Y_prob)))

            rate_type = 'I'
            if not could_proba:
                if self.config.mode.train:
                    Y_prob = []
                    rate_type = 'A'
                    for i in range(len(Y_unknown)):
                        try:
                            Y_prob = Y_prob + [class_report[Y_unknown[i]]['precision']]
                        except KeyError as e:
                            self.logger.print_warning(f"probability collection failed for key {Y_unknown[i]} with error {str(e)}")
                else:
                    rate_type = 'U'
                    
                self.logger.print_info("Probabilities:",str(Y_prob))
            self.update_progress(percent=self.percentPermajorTask)

            # Save new classifications for X_unknown in classification database
            self.logger.print_progress(message="Save new classifications in database")
            try:
                temp_model_name = Config.get_model_name(trained_model_name[0], trained_model_name[1])
                results_saved = self.datalayer.save_data( unpred_keys.values, Y_unknown, Y_prob, rate_type, trained_model.classes_, Y_probs, temp_model_name )
            except Exception as e:
                self.abort_cleanly(f"Save of predictions failed: {str(e)}")
            
            saved_query = self.datalayer.get_sql_command_for_recently_classified_data(results_saved)
            self.logger.print_info(f"Added {str(results_saved)} rows to classification table. Get them with SQL query:\n\n{saved_query}")

        self.update_progress(percent=self.percentPermajorTask)

        elapsed_time = time.time() - self.clock1
        date_again = str(datetime.now())
        self.logger.print_formatted_info(f"Ending program after {str(timedelta(seconds=round(elapsed_time)))} at {str(date_again)}")

        try:
            # Remove flag in database, signaling all was allright
            self.datalayer.mark_execution_ended()
        except Exception as e:
            self.abort_cleanly(f"Mark of executionend failed: {str(e)}")

        # Make sure progressbar is completed if not before
        self.logger.print_progress(message="Process finished", percent=1.0)

        # Close redirect of standard output in case of debugging
        if self.config.debug.on and self.config.io.redirect_output:
            sys.stdout.close()
            sys.stderr.close()
            
        # Return positive signal
        return 0
    

def get_rid_of_decimals(x):
    try:
        return int(round(float(x)))
    except Exception as e:
        sys.exit("Could not convert {0} to integer: {1}".format(x,str(ex)))

# Main program
def main(argv):

    if len(sys.argv) > 1:
        config = Config.load_config_from_module(argv)
    else:
       config = Config.Config()

    logger = IAFLogger.IAFLogger(not config.io.verbose)
    # Use the loaded configuration module argument
    # or create a classifier object with only standard settings
    myClassiphyer = IAFautomaticClassiphyer(config=config, logger=logger)

    # Run the classifier
    myClassiphyer.run()

# Start main
if __name__ == "__main__":
    main(sys.argv)
