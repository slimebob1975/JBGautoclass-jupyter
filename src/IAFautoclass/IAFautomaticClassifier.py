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
import sys
import os
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
                            
except Exception as ex:
    sys.exit("Aborting. Package installation failed: {0}".format(str(ex)))
    

# Imports of local help class for communication with SQL Server
import Config
import DataLayer

# General imports
import base64, getopt, pickle, scipy, numpy as np
import importlib, getpass
import time as time
import getpass
import copy
from datetime import datetime, timedelta
from numpy.linalg import lstsq, norm
from math import ceil
import matplotlib, pandas, sklearn
from pandas import read_csv, concat
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, Binarizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.kernel_approximation import Nystroem
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.preprocessing import LabelBinarizer
from sklearn.manifold import Isomap, LocallyLinearEmbedding
import time
import threading
from math import isnan
from stop_words import get_stop_words
import langdetect
from lexicalrichness import LexicalRichness
import warnings
#import ipywidgets as widgets
from pathlib import Path

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
    def __init__(self, config: Config.Config, progress_bar = None, progress_label = None, save_config_to_file: bool = False):
        self.config = config
        
        # For the external progressbar (hopefully iPython widgets)
        self.ProgressBar = progress_bar
        self.ProgressLabel = progress_label

        self.numMajorTasks = 12
        self.percentPermajorTask = 0.03
        if self.ProgressBar: self.ProgressBar.value = 0.0
        if self.ProgressLabel: self.ProgressLabel.value = "Starting up..."
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

            if self.config.io.verbose:
                print("Redirecting standard output to {0} and standard error to {1}".format(output_file,error_file))
            try:
                sys.stdout = open(output_file,'w')
                sys.stderr = open(error_file,'w')
            except Exception as ex:
                sys.exit("Something went wrong with redirection of standard output and error: {0}".format(str(ex)))
        
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
            if self.config.io.verbose: print("Reading in dataset from database...-please wait!")


            data, query, num_lines = self.datalayer.get_data(self.config.debug.num_rows, self.config.debug.on, self.config.io.redirect_output, self.config.mode.train, self.config.mode.predict)

            if data is None:
                return pandas.DataFrame(), None, None, None


            # Set the column names of the data array
            column_names = [self.config.connection.id_column] + [self.config.connection.class_column] + \
                    self.config.connection.data_text_columns.split(',') + \
                    self.config.connection.data_numerical_columns.split(',')
            try:
                column_names.remove("") # Remove any empty column name
            except Exception as ex:
                pass
            dataset = pandas.DataFrame(data, columns = column_names)
            
            # Make sure the class column is a categorical variable by setting it as string
            try:
                dataset.astype({self.config.connection.class_column: 'str'}, copy=False)
            except Exception as ex:
                print("Could not convert class column {0} to string variable: {1}". \
                      format(self.config.connection.class_column, str(ex)))
            
            # Extract unique class labels from dataset
            unique_classes = list(set(dataset[self.config.connection.class_column].tolist()))

            #TODO: Clean up the clean-up
            # Make an extensive search through the data for any inconsistencies (like NaNs and NoneType). 
            # Also convert datetime numerical variables to ordinals, i.e., convert them to the number of days 
            # or similar from a certain starting point, if any are left after the conversion above.
            if self.config.io.verbose: print("Consistency check")
            change = False
            percent_checked = 0
            try:
                for index in dataset.index:
                    old_percent_checked = percent_checked
                    percent_checked = round(100.0*float(index)/float(len(dataset.index)))
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
            except Exception as ex:
                raise ValueError("Aborting! Reason: {0}. Something went wrong in inconsistency check at {1}: {2}.".format(str(ex),key,item))

            # Shuffle the upper part of the data, such that all already classified material are
            # put in random order keeping the unclassified material in the bottom
            # Notice: perhaps this operation is now obsolete, since we now use a NEWID() in the 
            # data query above
            if self.config.io.verbose: print("\n\n---Shuffle data")
            num_un_pred = self.get_num_unpredicted_rows(dataset)
            dataset['rnd'] = np.concatenate([np.random.rand(num_lines - num_un_pred), [num_lines]*num_un_pred])
            dataset.sort_values(by = 'rnd', inplace = True )
            dataset.drop(['rnd'], axis = 1, inplace = True )

            # Use the unique id column from the data as the index column and take a copy, 
            # since it will not be used in the classification but only to reconnect each 
            # data row with classification results later on
            keys = dataset[self.config.connection.id_column].copy(deep = True).apply(get_rid_of_decimals)
            try:
                dataset.set_index(keys.astype('int64'), drop=False, append=False, inplace=True, \
                                  verify_integrity=False)
            except Exception as ex:
                sys.exit("Could not set index for dataset: {0}".format(str(ex)))
            dataset = dataset.drop([self.config.connection.id_column], axis = 1)

        except Exception as e:
            print("Load of dataset failed: " + str(e))
            if self.ProgressBar: self.ProgressBar.value = 1.0
            sys.exit("Program aborted.")

        # Return both data and keys
        return dataset, keys, query, unique_classes

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
                    if self.config.io.verbose: print("Text data picked for categorization: ",column)
                else:
                    text_dataset = concat([text_dataset, dataset[column]], axis = 1)
                    if self.config.io.verbose: print("Text data NOT picked for categorization: ",column)
                 
        if self.numerical_data:
            num_columns = self.config.connection.data_numerical_columns.split(',')
            for column in num_columns:
                num_dataset = concat([num_dataset, dataset[column]], axis = 1)

        # For concatenation, we need to make sure all text data are 
        # really treated as text, and categorical data as categories
        if self.text_data:
            if self.config.io.verbose: print("Text Columns:",text_dataset.columns)
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
                            if self.config.io.verbose: 
                                print("Column {0} was categorized with categories: {1}".format(column,lb.classes_))
                        else:
                            lb_results_df = pandas.DataFrame(lb_results, columns=[column])
                            if self.config.io.verbose:
                                print("Column {0} was binarized".format(column))
                    except ValueError as ex:
                        print("Column {0} could not be binarized: {1}".format(column,str(ex)))
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
                except Exception as ex:
                    my_language = self.STANDARD_LANG
                    print("Language could not be detected automatically: {0}. Fallback option, use: {1}.". \
                        format(str(ex),my_language))
                else:
                    if self.config.io.verbose: print("\nDetected language is: {0}".format(my_language))

            # Calculate the lexical richness
            try:
                lex = LexicalRichness(' '.join(X)) 
                if self.config.io.verbose: 
                    print("#Words, #Terms and TTR for original text is {0}, {1}, {2:5.2f} %"
                      .format(lex.words,lex.terms,100*float(lex.ttr)))
            except Exception as ex:
                print("Could not calculate lexical richness: {0}".format(str(ex)))

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
                if self.config.io.verbose: print("\nUsing standard stop words: ", my_stop_words)
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
                        if self.config.io.verbose: print("\nUsing specific stop words: ", text_specific_stop_words)
                    except ValueError as ex:
                        if self.config.io.verbose: print( \
                            "\nNotice: Specified stop words threshold at {0} generated no stop words." \
                            .format(self.config.mode.specific_stop_words_threshold))
                    my_stop_words = sorted(set(my_stop_words + text_specific_stop_words))
                    if self.config.io.verbose: print("\nTotal list of stop words:", my_stop_words)

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

    # Investigating dataset -- make some printouts to standard output
    def investigate_dataset(self, dataset, show_class_distribution = True):

        try:
            if self.config.io.verbose: print("Looking at dataset:")
            # 1. shape
            if self.config.io.verbose: print("\nShape:", dataset.shape)
        except Exception as ex:
           if self.config.io.verbose: print("Notice: an error occured in investigate_dataset: {0}".format(str(ex)))
        try:
            # 2. head
            if self.config.io.verbose: print("\nHead:",dataset.head(20))
        except Exception as ex:
            if self.config.io.verbose: print("Notice: an error occured in investigate_dataset: {0}".format(str(ex)))
        try:
            # 3. Data types
            if self.config.io.verbose: print("\nDatatypes:",dataset.dtypes)
        except Exception as ex:
            if self.config.io.verbose: print("Notice: an error occured in investigate_dataset: {0}".format(str(ex)))
        if show_class_distribution:
            try:
            # 4. Class distribution
                if self.config.io.verbose: print("Class distribution: ")
                if self.config.io.verbose: print(dataset.groupby(dataset.columns[0]).size())
            except Exception as ex:
                if self.config.io.verbose: print("Notice: an error occured in investigate_dataset: {0}".format(str(ex)))

    # Show statistics in standard output
    def show_statistics_on_dataset(self, dataset):

        # 1. Descriptive statistics
        pandas.set_option('display.width', 100)
        pandas.set_option('precision', 3)
        description = dataset.describe(datetime_is_numeric = True)
        if self.config.io.verbose: print("\nDescription:")
        if self.config.io.verbose: print(description)

        # 2. Correlations
        pandas.set_option('display.width', 100)
        pandas.set_option('precision', 3)
        description = dataset.corr('pearson')
        if self.config.io.verbose: print("\nCorrelation between attributes:")
        if self.config.io.verbose: print(description)

        # 3. Skew
        skew = dataset.skew()
        if self.config.io.verbose: print("\nSkew of Univariate descriptions")
        if self.config.io.verbose: print(skew)
        if self.config.io.verbose: print("\n")

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
                if self.config.io.verbose: print("\nPCA conversion of dataset under way...")
                if self.ProgressLabel: self.ProgressLabel.value = "PCA conversion of dataset under way..."
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
                    
                    print("Notice: PCA n_components is set to {0}".format(components))
                    try:
                        failed = True
                        feature_selection_transform = PCA(n_components=components)
                        feature_selection_transform.fit(X)
                        X = feature_selection_transform.transform(X)
                        failed = False
                    except Exception as ex:
                        print("Notice: PCA could not be used with n_components = {1}: {0}".format(str(ex),components))
                    else:
                        if self.config.io.verbose: print("...new shape of data matrix is: ({0},{1})\n".
                                          format(X.shape[0],X.shape[1]))
                        components = feature_selection_transform.n_components_
                        break

            # Nystroem transformation next
            elif self.config.mode.feature_selection == Config.Reduction.NYS:
                if self.ProgressLabel: self.ProgressLabel.value = "Nystroem conversion of dataset under way..."
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                else:
                    components = max(self.LOWER_LIMIT_REDUCTION, min(X.shape))
                    print("Notice: Nystroem n_components is set to: {0}".format(components))
                # Make transformation
                try:
                    feature_selection_transform = Nystroem(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as ex:
                    print("Notice: Nystroem transform could not be used: {0}".format(str(ex)))
                    failed = True
                else:
                    if self.config.io.verbose: print("...new shape of data matrix is: ({0},{1})\n".format(X.shape[0],X.shape[1]))
                    components = feature_selection_transform.n_components_
                    
            # Truncated SVD transformation next
            elif self.config.mode.feature_selection == Config.Reduction.TSVD:
                if self.ProgressLabel: self.ProgressLabel.value = "Truncated SVD reduction of dataset under way..."
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                else:
                    components = max(self.LOWER_LIMIT_REDUCTION, min(X.shape))
                    print("Notice: Truncated SVD n_components is set to: {0}".format(components))
                # Make transformation
                try:
                    feature_selection_transform = TruncatedSVD(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as ex:
                    print("Notice: Truncated SVD reduction could not be used: {0}".format(str(ex)))
                    failed = True
                else:
                    if self.config.io.verbose: print("...new shape of data matrix is: ({0},{1})\n".format(X.shape[0],X.shape[1]))
                    components = feature_selection_transform.n_components_

            # Fast independent component transformation next
            elif self.config.mode.feature_selection == Config.Reduction.FICA:
                if self.ProgressLabel: self.ProgressLabel.value = "Fast ICA reduction of dataset under way..."
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                else:
                    components = max(self.LOWER_LIMIT_REDUCTION, min(X.shape))
                    print("Notice: Fast ICA n_components is set to: {0}".format(components))
                # Make transformation
                try:
                    feature_selection_transform = FastICA(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as ex:
                    print("Notice: Fast ICA reduction could not be used: {0}".format(str(ex)))
                    failed = True
                else:
                    if self.config.io.verbose: print("...new shape of data matrix is: ({0},{1})\n".format(X.shape[0],X.shape[1]))
                    components = feature_selection_transform.n_components_
                    
            # Random Gaussian Projection
            elif self.config.mode.feature_selection == Config.Reduction.GRP:
                if self.ProgressLabel: self.ProgressLabel.value = "Gaussian Random Projection reduction of dataset under way..."
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                else:
                    components = 'auto'
                    print("Notice: GRP n_components is set to: {0}".format(components))
                # Make transformation
                try:
                    feature_selection_transform = GaussianRandomProjection(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as ex:
                    print("Notice: GRP reduction could not be used: {0}".format(str(ex)))
                    failed = True
                else:
                    if self.config.io.verbose: print("...new shape of data matrix is: ({0},{1})\n".format(X.shape[0],X.shape[1])) 
                    components = feature_selection_transform.n_components_
            
            # Non Linear Transformations below
            # First: Isomap
            elif self.config.mode.feature_selection == Config.Reduction.ISO:
                if self.ProgressLabel: self.ProgressLabel.value = "Isometric Mapping reduction of dataset under way..."
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                else:
                    components = self.NON_LINEAR_REDUCTION_COMPONENTS
                    print("Notice: ISO n_components is set to: {0}".format(components))
                # Make transformation
                try:
                    feature_selection_transform = Isomap(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as ex:
                    print("Notice: Isomap reduction could not be used: {0}".format(str(ex)))
                    failed = True
                else:
                    if self.config.io.verbose: print("...new shape of data matrix is: ({0},{1})\n".format(X.shape[0],X.shape[1])) 
                
            # Second: LLE
            elif self.config.mode.feature_selection == Config.Reduction.LLE:
                if self.ProgressLabel: self.ProgressLabel.value = "Locally Linear Embedding reduction of dataset under way..."
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                else:
                    components = self.NON_LINEAR_REDUCTION_COMPONENTS
                    print("Notice: LLE n_components is set to: {0}".format(components))
                # Make transformation
                try:
                    feature_selection_transform = LocallyLinearEmbedding(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as ex:
                    print("Notice: LLE reduction could not be used: {0}".format(str(ex)))
                    failed = True
                else:
                    if self.config.io.verbose: print("...new shape of data matrix is: ({0},{1})\n".format(X.shape[0],X.shape[1])) 
                
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
        if self.ProgressLabel:
            standardProgressText = self.ProgressLabel.value

        # Add all algorithms in a list
        if self.config.io.verbose: print("Spot check ml algorithms...")
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
        if self.config.io.verbose: # TODO: Bryt ut print/progressLabel/progressBar
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
                if self.ProgressLabel: self.ProgressLabel.value = f"{standardProgressText} ({name.name}-{preprocessor_name.name})"
                if not first_round: 
                    if self.ProgressBar: self.ProgressBar.value += percentAddPerMinorTask
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
                    except Exception as ex:
                        print("NOTICE: StratifiedKfold raised an exception with message: {1}. Killing the program".format(names[-1], str(ex)))
                        if self.ProgressBar: self.ProgressBar.value = 1.0
                        sys.exit()

                    # Now make kfolded cross evaluation
                    cv_results = None
                    try:
                        if scorer_mechanism == None:
                            if self.config.mode.scoring == Config.Scoretype.mcc:
                                scorer_mechanism = make_scorer(matthews_corrcoef)
                            else:
                                scorer_mechanism = self.config.mode.scoring.name
                        cv_results = cross_val_score(pipe, X_train, Y_train, cv=kfold, scoring=scorer_mechanism) 
                    except Exception as ex:
                        if self.config.io.verbose: print("NOTICE: Model {0} raised an exception in cross_val_score with message: {1}. Skipping to next".
                                          format(names[-1], str(ex)))
                    else:
                        # Stop the stopwatch
                        t = time.time() - t0

                        # For current settings, calculate score
                        temp_score = cv_results.mean()
                        temp_stdev = cv_results.std()

                        # Print results to screen
                        if self.config.io.verbose:
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
        except Exception as ex:
            sys.exit("Something went wrong on training picked model {0}: {1}".format(str(model),str(ex)))

        return model

    # Save ml model and corresponding configuration
    def save_model_to_file(self, label_binarizers, count_vectorizer, tfid_transformer, \
        feature_selection_transform, model_name, model, filename):
        
        try:
            save_config = self.config.get_clean_config()
            data = [save_config, label_binarizers, count_vectorizer, tfid_transformer, \
                feature_selection_transform, model_name, model]
            pickle.dump(data, open(filename,'wb'))

        except Exception as ex:
            print("Something went wrong on saving model to file: {0}".format(str(ex)))
    
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
        except Exception as ex:
            print("Something went wrong on loading model from file: {0}".format(str(ex)))

        return label_binarizers, count_vectorizer, tfid_transformer, feature_selection_transform, \
            model_name, model
    
    # Make predictions on dataset
    def make_predictions(self, model, X):

        could_predict_proba = False
        try:
            predictions = model.predict(X)
        except ValueError as e:
            print("It seems like you need to regenerate your prediction model: " + str(e))
            if self.ProgressBar: self.ProgressBar.value = 1.0
            sys.exit("Program aborted.")
        try:
            probabilities = model.predict_proba(X)
            rates = np.amax(probabilities, axis=1)
            could_predict_proba = True
        except Exception as e:
            if self.config.io.verbose:
                print("Probablity prediction not available for current model: " + \
                        str(e))
            probabilities = np.array([[-1.0]*len(model.classes_)]*X.shape[0])
            rates = np.array([-1.0]*X.shape[0])
        return predictions, could_predict_proba, rates, probabilities

    # Evaluate predictions
    def evaluate_predictions(self, predictions, Y, message="Unknown"):
        print("\nEvaluation performed with evaluation data: " + message)
        print("\nAccuracy score for evaluation data: ", accuracy_score(Y, predictions))
        print("\nConfusion matrix for evaluation data: \n\n", confusion_matrix(Y, predictions))
        print("\nClassification matrix for evaluation data: \n\n", classification_report(Y, predictions, zero_division='warn'))

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
                except Exception as ex:
                    print("Append of data row with index: {0} failed: {1}. Probably duplicate indicies in data!".format(str(i),str(ex)))
                    break

            # Quick end of loop if possible
            num_mispredicted = sum(elem == True for elem in X_not)
            if num_mispredicted != 0:
                break
 
        # Quick return if possible
        if num_mispredicted == 0:
            return "no model produced mispredictions", num_mispredicted, pandas.DataFrame()
        
        #
        if self.config.io.verbose:
            print("\nAccuracy score for {0}: {1}".format(what_model, accuracy_score(Y, Y_pred)))

        # Select the found mispredicted data
        X_mispredicted = X_transformed.loc[X_not]

        # Predict probabilites
        try:
            Y_prob = the_model.predict_proba(X_mispredicted)
            could_predict_proba = True
        except Exception as ex:
            print("Could not predict probabilities: {0}".format(str(ex)))
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
        return max(1, min(class_count.values()));
    
    # The main classification function for the class
    def run(self):
        
         # Print a welcoming message for the audience
        if self.config.io.verbose:
            self._print_welcoming_message()
        
        # Create the classification table, if it does not exist already
        if self.ProgressLabel: self.ProgressLabel.value = "Create the classification table"
        self.datalayer.create_classification_table()
        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        try:
            # Set a flag in the classification database that execution has started
            self.datalayer.mark_execution_started()
        except Exception as e:
            print("Mark of executionstart failed: " + str(e))
            sys.exit("Program aborted.")

        # Read in all data, with classifications or not
        if self.ProgressLabel: self.ProgressLabel.value = "Read in data from database"
        dataset, unique_keys, data_query, self.unique_classes = self.read_in_data()
        if dataset.empty:
            if self.ProgressLabel: self.ProgressLabel.value = "Process finished"
            if self.ProgressBar: self.ProgressBar.value = 1.0
            return -1
            
        elif self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Investigate dataset (optional)
        if self.config.mode.train:
            if self.ProgressLabel: self.ProgressLabel.value = "Investigate dataset (see console)"
            if self.config.io.verbose: self.investigate_dataset(dataset)

            # Give some statistical overview of the training material
            if self.config.io.verbose: self.show_statistics_on_dataset(dataset)
            if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

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
        if self.ProgressLabel: self.ProgressLabel.value = "Rearrange dataset for possible textclassification, etc."
        X, Y, label_binarizers, count_vectorizer, tfid_transformer = \
            self.convert_textdata_to_numbers(dataset, label_binarizers, count_vectorizer, tfid_transformer)

        if self.text_data:
            if self.config.io.verbose: 
                print("After conversion of text data to numerical data:")
                self.investigate_dataset( X, False )
        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # In case of PCA or Nystreom or other feature reduction, here it goes
        if self.use_feature_selection and self.config.mode.feature_selection != Config.Reduction.RFE:
            t0 = time.time()
            X, self.config.mode.num_selected_features, feature_selection_transform = \
                self.perform_feature_selection( X, self.config.mode.num_selected_features, feature_selection_transform )

            t = time.time() - t0
            if self.config.io.verbose: print("Feature reduction took " + str(round(t,2)) + " sec.\n")

        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Split dataset for machine learning
        if self.config.mode.train:
            if self.ProgressLabel: self.ProgressLabel.value = "Split dataset for machine learning"
            X_train, X_validation, X_unknown, Y_train, Y_validation, Y_unknown = self.split_dataset(X, Y, num_un_pred)
        else:
            X_train = None
            X_validation = None
            Y_train = None
            Y_validation = None
            X_unknown = X
            Y_unknown = Y

        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Check algorithms for best model and train that model. K-value should be 10 or below.
        # Or just use the model previously trained.
        num_features = None
        if self.config.mode.train:
            if self.ProgressLabel: self.ProgressLabel.value = "Check and train algorithms for best model"
            k = min(10, self.find_smallest_class_number(Y_train))
            if k < 10:
                if self.config.io.verbose: print("Using non-standard k-value for spotcheck of algorithms: {0}".format(k))
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
            sys.exit("Aborting. User must choose either to train a new model or use old one for predictions!")
        if self.config.io.verbose: print("Best model is: {0} with number of features: {1}".format(trained_model_name, num_features))
        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Make predictions on known testdata
        if self.config.mode.train and X_validation.shape[0] > 0:
            if self.ProgressLabel: self.ProgressLabel.value = "Make predictions on known testdata"
            pred, could_proba, prob, probs = self.make_predictions(trained_model, X_validation)
            if could_proba:
                #print("Training Probabilities:",prob)
                print("Training Classification prob: Mean, Std.dev: ", \
                        np.mean(prob),np.std(prob))
        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Evaluate predictions (optional)
        if self.config.mode.train and Y_validation.shape[0] > 0:
            if self.ProgressLabel: self.ProgressLabel.value = "Evaluate predictions"
            ml_algorithm = Config.get_model_name(trained_model_name[0], trained_model_name[1])
            self.evaluate_predictions(pred, Y_validation, "ML algorithm: " + ml_algorithm)
        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Get accumulated classification score report for all predictions
        if self.config.mode.train and Y_validation.shape[0] > 0:
            class_report = classification_report(Y_validation, pred, output_dict = True)
            print("Classification report for {0} with #features: {1}".format(trained_model_name, num_features))
            for key in class_report.keys():
                print("{0}: {1}".format(key, class_report[key]))

        # RETRAIN best model on whole dataset: Xtrain + Xvalidation
        if self.config.mode.train and (X_train.shape[0] + X_validation.shape[0]) > 0:
            if self.ProgressLabel: self.ProgressLabel.value = "Retrain model on whole dataset"
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
            if self.config.io.verbose:
                print("Total number of mispredicted elements: {0}".format(num_mispredicted))
            joiner = self.config.connection.id_column + " = \'"
            most_mispredicted_query = data_query + "WHERE " +  joiner \
                + ("\' OR " + joiner).join([str(number) for number in self.X_most_mispredicted.index.tolist()]) + "\'"
            if not self.X_most_mispredicted.empty and self.config.mode.mispredicted: 
                if self.config.io.verbose:
                    print("\n---Most mispredicted during training (using {0}):".format(what_model))
                    print(self.X_most_mispredicted.head(self.LIMIT_MISPREDICTED))
                    print("\nGet the most misplaced data by SQL query:\n {0}".format(most_mispredicted_query))
                    print("\nOr open the following csv-data file: \n\t {0}".format(self.misplaced_filepath))
                self.X_most_mispredicted.to_csv(path_or_buf = self.misplaced_filepath, sep = ';', na_rep='N/A', \
                                           float_format=None, columns=None, header=True, index=True, \
                                           index_label=self.config.connection.id_column, mode='w', encoding='utf-8', \
                                           compression='infer', quoting=None, quotechar='"', line_terminator=None, \
                                           chunksize=None, date_format=None, doublequote=True, decimal=',', \
                                           errors='strict')
        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Now make predictions on non-classified dataset: X_unknown -> Y_unknown
        if self.config.mode.predict and X_unknown.shape[0] > 0:
            if self.ProgressLabel: self.ProgressLabel.value = "Make predictions on un-classified dataset"
            Y_unknown, could_proba, Y_prob, Y_probs = self.make_predictions(trained_model, X_unknown)
            print("\n---Predictions for the unknown data---\n")
            if self.config.io.verbose: print("Predictions:", Y_unknown)
            if could_proba:
                #print("Probabilities:",Y_prob)
                print("Classification Y_prob: Mean, Std.dev: ", np.mean(Y_prob), np.std(Y_prob))

            rate_type = 'I'
            if not could_proba:
                if self.config.mode.train:
                    Y_prob = []
                    rate_type = 'A'
                    for i in range(len(Y_unknown)):
                        try:
                            Y_prob = Y_prob + [class_report[Y_unknown[i]]['precision']]
                        except KeyError as e:
                            print("Warning: probability collection failed for key {0} with error {1}".format(Y_unknown[i]),str(e))
                else:
                    rate_type = 'U'
                    
                if self.config.io.verbose: print("Probabilities:",Y_prob)
            if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

            # Save new classifications for X_unknown in classification database
            if self.ProgressLabel: self.ProgressLabel.value = "Save new classifications in database"
            try:
                if self.config.io.verbose: print("Save predictions to database...-please wait!")
                temp_model_name = Config.get_model_name(trained_model_name[0], trained_model_name[1])
                results_saved = self.datalayer.save_data( unpred_keys.values, Y_unknown, Y_prob, rate_type, trained_model.classes_, Y_probs, temp_model_name )
            except Exception as e:
                print("Save of predictions failed: {0}".format(str(e)))
                if self.ProgressBar: self.ProgressBar.value = 1.0
                sys.exit("Program aborted.")
            
            print("Added {0} rows to classification table. Get them with SQL query:\n\n{1}".
                  format(results_saved,self.datalayer.get_sql_command_for_recently_classified_data(results_saved)))

        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        elapsed_time = time.time() - self.clock1
        date_again = str(datetime.now())
        if self.config.io.verbose: print("\n--- Ending program after " + str(timedelta(seconds=round(elapsed_time))) + " at " + str(date_again) + " ---\n")

        try:
            # Remove flag in database, signaling all was allright
            self.datalayer.mark_execution_ended()
        except Exception as e:
            print("Mark of executionend failed: " + str(e))
            if self.ProgressBar: self.ProgressBar.value = 1.0
            sys.exit("Program aborted.")

        # Make sure progressbar is completed if not before
        if self.ProgressLabel: self.ProgressLabel.value = "Process finished"
        if self.ProgressBar: self.ProgressBar.value = 1.0

        # Close redirect of standard output in case of debugging
        if self.config.debug.on and self.config.io.redirect_output:
            sys.stdout.close()
            sys.stderr.close()
            
        # Return positive signal
        return 0
    
    # Some welcoming message to the audience
    def _print_welcoming_message(self):
        
        # Print welcoming message
        print("\n *** WELCOME TO IAF AUTOMATIC CLASSIFICATION SCRIPT ***\n")
        print("Execution started at: {0:>30s} \n".format(str(self.date_now)))

        # Print configuration settings
        print(self.config)

        # Print out what mode we use: training a new model or not
        if self.config.mode.train:
            print(" -- We will train our ml model --")
        else:
            try:
                f = open(self.model_filename,'br')
            except IOError:
                sys.exit("Aborting. No trained model exists at {0}".format(self.model_filename))
            else:
                print(" -- We will reload and use old model --")

def get_rid_of_decimals(x):
    try:
        return int(round(float(x)))
    except Exception as ex:
        sys.exit("Could not convert {0} to integer: {1}".format(x,str(ex)))

# Main program
def main(argv):

    if len(sys.argv) > 1:
        config = Config.load_config_from_module(argv)
    else:
       config = Config.Config()

    # Use the loaded configuration module argument
    # or create a classifier object with only standard settings
    myClassiphyer = IAFautomaticClassiphyer(config=config)

    # Run the classifier
    myClassiphyer.run()

# Start main
if __name__ == "__main__":
    main(sys.argv)
